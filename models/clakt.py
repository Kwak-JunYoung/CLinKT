# https://github.com/arghosh/AKT/blob/master/akt.py
import torch
import torch.nn as nn
from torch.nn import Sequential
from torch.nn import Module, Embedding, Linear, ReLU, Dropout, ModuleList, Sequential
from .modules import AKTTransformerLayer
import torch.nn.functional as F
from .rpe import SinusoidalPositionalEmbeddings

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class AKT(Module):
    def __init__(
        self,
        device,
        num_skills,
        num_questions,
        seq_len,
        diff_as_loss_weight,
        embedding_size,
        num_blocks,
        kq_same,
        model_type="akt",
        num_attn_heads=8,
        final_fc_dim=512,
        d_ff=2048,
        reg_l=1e-5,
        dropout=0.2,
        de_type="none",
        separate_qr=False,
    ):
        super(AKT, self).__init__()

        """
        params:
            num_skills: # of skills
            num_questions: # of questions
            embedding_size: embedding dim
            num_blocks: # of attn blocks
            seq_len: max length of sequenc
            kq_same: key랑 query랑 같은지
            num_attn_heads: number of heads if multi-headed attention
            final_fc_dim: dimension of final fully connected net before prediction
            d_ff: dimension for fully connected net inside the basic block
            
        """
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.embedding_size = embedding_size
        self.num_blocks = num_blocks
        self.seq_len = seq_len
        self.kq_same = kq_same
        self.model_type = model_type
        self.num_attn_heads = num_attn_heads
        self.final_fc_dim = final_fc_dim
        self.d_ff = d_ff
        self.reg_l = reg_l
        self.dropout = dropout
        self.separate_qr = separate_qr
        self.diff_as_loss_weight = diff_as_loss_weight
        self.device_info = device

        if self.num_questions > 0:
            self.difficult_param = Embedding(
                self.num_questions, 1, padding_idx=0)  # /mu_{q_t} parameter
            self.q_embed_diff = Embedding(
                self.num_skills, self.embedding_size, padding_idx=0)  # d_{c_t}
            self.qr_embed_diff = Embedding(
                2 * self.num_skills, self.embedding_size, padding_idx=0)  # f_{(c_t, r_t)} or h_{r_t}

        self.q_embed = Embedding(
            self.num_skills, self.embedding_size, padding_idx=0)  # c_{c_t}

        if self.separate_qr:
            self.qr_embed = Embedding(
                2 * self.num_skills, self.embedding_size, padding_idx=0)  # e_{(c_t, r_t)}
        else:
            self.r_embed = Embedding(
                2 + 1, self.embedding_size, padding_idx=0)  # e_{(c_t, r_t)}

        self.de = de_type.split('_')[0]
        self.token_num = int(de_type.split('_')[1])
        if self.de in ["sde", "lsde"]:
            diff_vec = torch.from_numpy(SinusoidalPositionalEmbeddings(
                2*(self.token_num+1), embedding_size)).to(device)
            self.diff_emb = Embedding.from_pretrained(diff_vec, freeze=True)
            rotary = "none"
        elif self.de in ["rde", "lrde"]:
            rotary = "qkv"
        else:
            rotary = "none"

        self.model = Architecture(
            n_question=self.num_skills,
            n_blocks=self.num_blocks,
            n_heads=self.num_attn_heads,
            dropout=self.dropout,
            d_model=self.embedding_size,
            d_feature=self.embedding_size / self.num_attn_heads,
            d_ff=self.d_ff,
            kq_same=self.kq_same,
            model_type=self.model_type,
            de=self.de,
            rotary=rotary,
            device_info=self.device_info
        )

        self.out = Sequential(
            Linear(2 * self.embedding_size, self.final_fc_dim),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.final_fc_dim, self.final_fc_dim // 2),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.final_fc_dim // 2, 1),
        )
        self.reset()
        self.loss_fn = nn.BCELoss(reduction="mean")
        if self.diff_as_loss_weight:
            self.loss_fn = nn.BCELoss(reduction="none")

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.num_questions + 1 and self.num_questions > 0:
                torch.nn.init.constant_(p, 0.0)

    def forward(self, batch):

        if self.training:
            # augmented q_i, augmented q_j and original q
            q_i, q_j, q = batch["skills"][0][:, :-1], batch["skills"][1][:, :-1], batch["skills"][2][:, :-1]
            
            # augmented r_i, augmented r_j and original r
            r_i, r_j, r, neg_r = batch["responses"][0][:, :-1], batch["responses"][1][:, :-1], batch["responses"][2][:, :-1], batch["responses"][3][:, :-1]
            masked_r_i = r_i * (r_i > -1).long()
            masked_r_j = r_j * (r_j > -1).long()
            masked_r = r * (r > -1).long()
            masked_neg_r = neg_r * (neg_r > -1).long()
            
            attention_mask_i, attention_mask_j, attention_mask = batch["attention_mask"][0][:, :-1], batch["attention_mask"][1][:, :-1], batch["attention_mask"][2][:, :-1]

            pid_data_i, pid_data_j, pid_data = batch["questions"][0][:, :-1], batch["questions"][1][:, :-1], batch["questions"][2][:, :-1]

            # augmented diff_i, augmented diff_j and original diff
            diff_i, diff_j, diff = batch["sdiff"][0][:, :-1], batch["sdiff"][1][:, :-1], batch["sdiff"][2][:, :-1]            
        
            if self.token_num < 1000:
                boundaries = torch.linspace(0, 1, steps=self.token_num+1)
                diff = torch.bucketize(diff, boundaries)
                diff_i = torch.bucketize(diff_i, boundaries)
                diff_j = torch.bucketize(diff_j, boundaries)

                diff_ox = torch.where(r==0 , (diff-(self.token_num+1)) * (r > -1).int(), diff * (r > -1).int())
                diff_ox_i = torch.where(r_i==0 , (diff_i-(self.token_num+1)) * (r_i > -1).int(), diff_i * (r_i > -1).int())
                diff_ox_j = torch.where(r_j==0 , (diff_j-(self.token_num+1)) * (r_j > -1).int(), diff_j * (r_j > -1).int())
                diff_neg = torch.where(neg_r==1 , (diff-(self.token_num+1)) * (neg_r > -1).int(), diff * (neg_r > -1).int())

            else:
                diff = diff * 100
                diff_i = diff_i * 100
                diff_j = diff_j * 100

                diff_ox = torch.where(r==0 , (diff-(100+1)) * (r > -1).int(), diff * (r > -1).int())
                diff_ox_i = torch.where(r_i==0 , (diff_i-(100+1)) * (r_i > -1).int(), diff_i * (r_i > -1).int())
                diff_ox_j = torch.where(r_j==0 , (diff_j-(100+1)) * (r_j > -1).int(), diff_j * (r_j > -1).int())
                diff_neg = torch.where(neg_r==1 , (diff-(100+1)) * (neg_r > -1).int(), diff * (neg_r > -1).int())    
            
            q_embed_data_i = self.q_embed(q_i)
            q_embed_data_j = self.q_embed(q_j)
            q_embed_data = self.q_embed(q)

            if self.separate_qr:
                qr_i = q_i + self.num_skills * masked_r_i
                qr_j = q_j + self.num_skills * masked_r_j
                qr = q + self.num_skills * masked_r

                qr_embed_data_i = self.qr_embed(qr_i)
                qr_embed_data_j = self.qr_embed(qr_j)
                qr_embed_data = self.qr_embed(qr)
            else:
                qr_i = masked_r_i
                qr_j = masked_r_j                
                qr = masked_r

                qr_embed_data_i = self.q_embed(qr_i)
                qr_embed_data_j = self.q_embed(qr_j)
                qr_embed_data = self.q_embed(qr)
            
            if self.num_questions > 0:
                q_embed_diff_data_i = self.q_embed_diff(q_i)  # d_{c_t}: variation vector
                q_embed_diff_data_j = self.q_embed_diff(q_j)
                q_embed_diff_data = self.q_embed_diff(q)

                pid_embed_data_i = self.difficult_param(pid_data_i)  # \mu_{q_t}
                pid_embed_data_j = self.difficult_param(pid_data_j)
                pid_embed_data = self.difficult_param(pid_data)

                q_embed_data_i = (
                    q_embed_data_i + pid_embed_data_i * q_embed_diff_data_i
                )
                q_embed_data_j = (
                    q_embed_data_j + pid_embed_data_j * q_embed_diff_data_j
                )
                q_embed_data = (
                    q_embed_data + pid_embed_data * q_embed_diff_data
                )

                qr_embed_diff_data_i = self.qr_embed_diff(
                    qr_i)
                qr_embed_diff_data_j = self.qr_embed_diff(
                    qr_j)
                qr_embed_diff_data = self.qr_embed_diff(
                    qr)
                
                if self.separate_qr:
                    qr_embed_data_i = qr_embed_data_i + pid_embed_data_i * qr_embed_diff_data_i
                    qr_embed_data_j = qr_embed_data_j + pid_embed_data_j * qr_embed_diff_data_j
                    qr_embed_data = qr_embed_data + pid_embed_data * qr_embed_diff_data

                else:
                    if self.de in ["sde", "lsde"]:
                        diffx = (self.token_num+1) + diff * (r > -1).long()
                        diffo = diff * (r > -1).int()
                        diffox = torch.where(r == 0 ,diffo, diffx)
                        demb = self.diff_emb(diffox).float()
                        qr_embed_data += demb

                        diff_x_i = (self.token_num+1) + diff_i * (r_i > -1).long()
                        diff_o_i = diff_i * (r_i > -1).int()
                        diff_ox_i = torch.where(r_i == 0 ,diff_o_i, diff_x_i)
                        demb_i = self.diff_emb(diff_ox_i).float()
                        qr_embed_data_i += demb_i

                        diff_x_j = (self.token_num+1) + diff_j * (r_j > -1).long()
                        diff_o_j = diff_j * (r_j > -1).int()
                        diff_ox_j = torch.where(r_j == 0 ,diff_o_j, diff_x_j)
                        demb_j = self.diff_emb(diff_ox_j).float()
                        qr_embed_data_j += demb_j

                    elif self.de in ["rde", "lrde"]:
                        demb = None
                        demb_i = None
                        demb_j = None
                    else:
                        demb = None
                        demb_i = None
                        demb_j = None
                        qr_embed_data_i = qr_embed_data_i + pid_embed_data_i * (
                            qr_embed_diff_data_i + q_embed_diff_data_i          
                        )
                        qr_embed_data_j = qr_embed_data_j + pid_embed_data_j * (
                            qr_embed_diff_data_j + q_embed_diff_data_j          
                        )
                        qr_embed_data = qr_embed_data + pid_embed_data * (
                            qr_embed_diff_data + q_embed_diff_data          
                        )             

                c_reg_loss = torch.mean(pid_embed_data ** 2.0) * self.reg_l
                c_reg_loss_i = torch.mean(pid_embed_data_i ** 2.0) * self.reg_l
                c_reg_loss_j = torch.mean(pid_embed_data_j ** 2.0) * self.reg_l

            else:
                c_reg_loss = 0
                c_reg_loss_i = 0
                c_reg_loss_j = 0

            pooled_ques_score_i = (self.q_embed(q_i) * attention_mask_i.unsqueeze(-1)).sum(
                1
            ) / attention_mask_i.sum(-1).unsqueeze(-1)

            pooled_ques_score_j = (self.q_embed(q_j) * attention_mask_j.unsqueeze(-1)).sum(
                1
            ) / attention_mask_j.sum(-1).unsqueeze(-1)

            pooled_ques_score = (self.q_embed(q) * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1).unsqueeze(-1)

            pooled_inter_score_i = (qr_embed_data_i * attention_mask_i.unsqueeze(-1)).sum(
                1
            ) / attention_mask_i.sum(-1).unsqueeze(-1)

            pooled_inter_score_j = (qr_embed_data_j * attention_mask_j.unsqueeze(-1)).sum(
                1
            ) / attention_mask_j.sum(-1).unsqueeze(-1)

            pooled_inter_score = (qr_embed_data * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1).unsqueeze(-1)


            d_output, attn = self.model(q_embed_data, qr_embed_data, demb, diff_ox)  # 211x512
            d_output_i, attn_i = self.model(q_embed_data_i, qr_embed_data_i, demb_i, diff_ox_i)
            d_output_j, attn_j = self.model(q_embed_data_j, qr_embed_data_j, demb_j, diff_ox_j)

            concat_q_i = torch.cat([d_output_i, q_embed_data_i],
                                dim=-1)
            
            concat_q_j = torch.cat([d_output_j, q_embed_data_j],
                                dim=-1)
            
            concat_q = torch.cat([d_output, q_embed_data],
                                dim=-1)
            
            output_i = torch.sigmoid(self.out(concat_q_i)).squeeze()
            output_j = torch.sigmoid(self.out(concat_q_j)).squeeze()
            output = torch.sigmoid(self.out(concat_q)).squeeze()
            
            out_dict = {
                "pred": output[:, 1:],
                "true": r[:, 1:].float(),
                "c_reg_loss": c_reg_loss,
            }

        else: 
            q = batch["skills"]
            r = batch["responses"]
            attention_mask = batch["attention_mask"]
            masked_r = r * (r > -1).long()
            pid_data = batch["questions"]
            diff = batch["sdiff"]

            if self.token_num < 1000:
                boundaries = torch.linspace(0, 1, steps=self.token_num+1)
                diff = torch.bucketize(diff, boundaries)
                diff_ox = torch.where(
                    r == 0, (diff-(self.token_num+1)) * (r > -1).int(), diff * (r > -1).int())
            else:
                diff = diff * 100
                diff_ox = torch.where(r == 0, (diff-(100+1))
                                    * (r > -1).int(), diff * (r > -1).int())

            # c_{c_t}: [batch_size, seq_len, embedding_size]
            q_embed_data = self.q_embed(q)

            if self.separate_qr:
                qr = q + self.num_skills * masked_r
                # f_{(c_t, r_t)}: [batch_size, seq_len, d_model]
                qr_embed_data = self.qr_embed(qr)
            else:
                qr = masked_r
                qr_embed_data = q_embed_data + self.r_embed(qr)

            if self.num_questions > 0:
                q_embed_diff_data = self.q_embed_diff(
                    q)  # d_{c_t}: variation vector
                pid_embed_data = self.difficult_param(pid_data)  # \mu_{q_t}
                q_embed_data = (
                    q_embed_data + pid_embed_data * q_embed_diff_data
                )  # x_t = c_{c_t} + \mu_{q_t} + d_{c_t}
                qr_embed_diff_data = self.qr_embed_diff(
                    qr)  # f_{(c_t, r_t)} or h_{r_t}

                if self.separate_qr:
                    qr_embed_data = qr_embed_data + pid_embed_data * qr_embed_diff_data
                else:
                    # y_t = e_{(c_t, r_t)} + \mu_{q_t} * f_{(c_t, r_t)}
                    # , where e_{(c_t, r_t)} = c_{c_t} + g_{r_t}
                    # f_{(c_t, r_t)} = f_{(c_t, r_t)} + d_{c_t}
                    # e_{(c_t, r_t)} + \mu_{q_t} * (h_{r_t} + d_{c_t})
                    if self.de in ["sde", "lsde"]:
                        diffx = (self.token_num+1) + diff * (r > -1).long()
                        diffo = diff * (r > -1).int()
                        diffox = torch.where(r == 0, diffo, diffx)
                        demb = self.diff_emb(diffox).float()
                        qr_embed_data += demb
                    elif self.de in ["rde", "lrde"]:
                        demb = None
                    else:
                        demb = None
                        qr_embed_data = qr_embed_data + pid_embed_data * (
                            qr_embed_diff_data + q_embed_diff_data
                        )

                c_reg_loss = torch.mean(pid_embed_data ** 2.0) * self.reg_l
            else:
                c_reg_loss = 0

            pooled_ques_score = (self.q_embed(q) * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1).unsqueeze(-1)
            pooled_inter_score = (qr_embed_data * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1).unsqueeze(-1)

            # [batch_size, seq_len, d_model]
            # pass to the decoder
            # output shape [batch_size, seq_len, d_model or d_model//2]
            # d_output is h_t

            d_output, attn = self.model(
                q_embed_data, qr_embed_data, demb, diff_ox)  # 211x512

            concat_q = torch.cat([d_output, q_embed_data],
                                dim=-1)  # concat([h_t, x_t])
            output = torch.sigmoid(self.out(concat_q)).squeeze()

            out_dict = {
                "pred": output[:, 1:],
                "true": r[:, 1:].float(),
                "c_reg_loss": c_reg_loss,
                "q_embed": pooled_ques_score,
                "qr_embed": pooled_inter_score,
            }


        return out_dict

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        c_reg_loss = out_dict["c_reg_loss"]
        mask = true > -1

        loss = self.loss_fn(pred[mask], true[mask])
        if self.diff_as_loss_weight:
            weight = F.softmax(
                1-feed_dict['sdiff'][:, 1:].flatten()[mask], dim=0)
            loss = torch.sum(loss * weight)

        return loss + c_reg_loss, len(pred[mask]), true[mask].sum().item()

    def alignment_and_uniformity(self, out_dict):
        return (
            out_dict["uniformity"],
            out_dict["uniformity"],
            out_dict["uniformity"],
            out_dict["uniformity"],
        )


class Architecture(Module):
    def __init__(
        self,
        n_question,
        n_blocks,
        d_model,
        d_feature,
        d_ff,
        n_heads,
        dropout,
        kq_same,
        model_type,
        de="none",
        rotary="none",
        device_info="cpu"
    ):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.de = de
        self.d_model = d_model
        self.model_type = model_type
        print("model_type", model_type)

        if model_type == "akt":
            self.blocks_1 = ModuleList(
                [
                    AKTTransformerLayer(
                        d_model=d_model,
                        d_feature=d_model // n_heads,
                        d_ff=d_ff,
                        dropout=dropout,
                        n_heads=n_heads,
                        kq_same=kq_same,
                        rotary=rotary,
                    )
                    for _ in range(n_blocks)
                ]
            )
            self.blocks_2 = ModuleList(
                [
                    AKTTransformerLayer(
                        d_model=d_model,
                        d_feature=d_model // n_heads,
                        d_ff=d_ff,
                        dropout=dropout,
                        n_heads=n_heads,
                        kq_same=kq_same,
                    )
                    for _ in range(n_blocks * 2)
                ]
            )

    def forward(self, q_embed_data, qa_embed_data, demb=None, diff=None):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        # knowledge encoder: encode (question, response)'s
        for i, block in enumerate(self.blocks_1):
            # knowledge encoder
            # y^{\hat}_{t-1} = f_{enc_2} (y_1, ..., y_{t-1})
            # y can see both current and past information
            """
            mask: 0 means that it can peek only past values.
            1 means that block can peek only current and past values
            """
            if i > 0 and self.de == "lsde":
                y += demb
            if i > 0 and self.de == "rde":
                diff = None
            y, _ = block(mask=1, query=y, key=y, values=y, diff=diff)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                # question encoder
                # x^{\hat}_{t} = f_{enc_1} (x_1, ..., x_t)
                # x can see both current and past information
                x, _ = block(mask=1, query=x, key=x, values=x, apply_pos=False)
                flag_first = False
            else:  # dont peek current response
                # knoweldge retriever
                # h_t = f_{kr} (x^{\hat}_1, ..., x^{\hat}_t, y^{\hat}_1, ..., y^{\hat}_{t-1})
                # h can see past only
                x, attn = block(mask=0, query=x, key=x,
                                values=y, apply_pos=True)
                flag_first = True
        return x, attn
