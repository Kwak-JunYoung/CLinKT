import torch

from torch.nn import Module, Embedding, Linear, LayerNorm, Dropout, BCELoss
from .modules import transformer_FFN, pos_encode, ut_mask, get_clones, MultiheadAttention
from .rpe import SinusoidalPositionalEmbeddings 
from .sakt import SAKT
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

from IPython import embed

class CLSAKT(Module):
    def __init__(
        self, 
        device, 
        num_skills,
        num_questions, 
        seq_len, 
        **kwargs
        ):
        super().__init__()
        self.device = device 

        embedding_size = kwargs["embedding_size"]
        num_attn_heads = kwargs["num_attn_heads"]
        dropout = kwargs["dropout"]
        de_type = kwargs["de_type"]

        self.num_questions = num_questions
        self.num_skills = num_skills
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_blocks = kwargs["num_blocks"]
        self.loss_fn = BCELoss(reduction="mean")
        self.cl_loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

        # num_questions, seq_len, embedding_size, num_attn_heads, dropout, emb_path="")
        self.interaction_emb = Embedding(num_skills * 2, embedding_size, padding_idx=0)
        self.exercise_emb = Embedding(num_skills, embedding_size, padding_idx=0)
        # self.P = Parameter(torch.Tensor(self.seq_len, self.embedding_size))
        self.position_emb = Embedding(seq_len + 1, embedding_size, padding_idx=0)

        self.de = de_type.split('_')[0]
        assert self.de not in ["lsde", "lrde"], "de_type error! should not in [lsde, lrde]"
        self.token_num = int(de_type.split('_')[1])
        
        if self.de in ["sde"]:
            diff_vec = torch.from_numpy(SinusoidalPositionalEmbeddings(2*(self.token_num+1), embedding_size)).to(device)
            self.diff_emb = Embedding.from_pretrained(diff_vec, freeze=True)
            rotary = "none"
        elif self.de in ["rde"]:
            rotary = "kv"
        else: 
            rotary = "none"

        self.blocks = get_clones(Blocks(device, embedding_size, num_attn_heads, dropout, rotary), self.num_blocks)

        self.dropout_layer = Dropout(dropout)
        self.pred = Linear(self.embedding_size, 1)

    def base_emb(self, q, r, qry, pos, diff):
        masked_responses = r * (r > -1).long()
        x = q + self.num_skills * masked_responses
        qshftemb, xemb = self.exercise_emb(qry), self.interaction_emb(x)
        posemb = self.position_emb(pos)
        xemb = xemb + posemb
        if self.de in ["sde"]:
            diffx = (self.token_num+1) + diff * (r > -1).long()
            diffo = diff * (r > -1).int()
            diffox = torch.where(r == 0 ,diffo, diffx)
            demb = self.diff_emb(diffox).float()
            xemb += demb
            return qshftemb, xemb, demb
        else:
            return qshftemb, xemb, None

    def forward(self, batch):
        # augmented q_i, augmented q_j and original q

        q_i, q_j, q = batch["skills"][0][:, :-1], batch["skills"][1][:, :-1], batch["skills"][2][:, :-1]
        
        # augmented r_i, augmented r_j and original r
        r_i, r_j, r, neg_r = batch["responses"][0][:, :-1], batch["responses"][1][:, :-1], batch["responses"][2][:, :-1], batch["responses"][:, :-1]

        # augmented qry_i, augmented qry_j and original qry
        qry_i, qry_j, qry = batch["skills"][0][:, 1:], batch["skills"][1][:, 1:], batch["skills"][2][:, 1:]

        # augmented pos_i, augmented pos_j and original pos
        pos_i, pos_j, pos = batch["position"][0][:, :-1], batch["position"][1][:, :-1], batch["position"][2][:, :-1]

        # augmented diff_i, augmented diff_j and original diff
        diff_i, diff_j, diff = batch["sdiff"][0][:, :-1], batch["sdiff"][1][:, :-1], batch["sdiff"][2][:, :-1]
        
        qshftemb_i, xemb_i, demb_i = self.base_emb(q_i, r_i, qry_i, pos_i, diff_i)
        qshftemb_j, xemb_j, demb_j = self.base_emb(q_j, r_j, qry_j, pos_j, diff_j)
        qshftemb, xemb, demb = self.base_emb(q, r, qry, pos, diff)

        if self.token_num < 1000:
            boundaries = torch.linspace(0, 1, steps=self.token_num+1)                
            diff = torch.bucketize(diff, boundaries)
            diff_ox = torch.where(r==0 , (diff-(self.token_num+1)) * (r > -1).int(), diff * (r > -1).int())  
        else:
            diff = diff * 100
            diff_ox = torch.where(r==0 , (diff-(100+1)) * (r > -1).int(), diff * (r > -1).int())
            
        qshftemb, xemb, demb = self.base_emb(q, r, qry, pos, diff)
        
        for i in range(self.num_blocks): #sakt's num_blocks = 1
            xemb_i = self.blocks[i](qshftemb_i, xemb_i, xemb_i, diff_ox)
            xemb_j = self.blocks[i](qshftemb_j, xemb_j, xemb_j, diff_ox)
            xemb = self.blocks[i](qshftemb, xemb, xemb, diff_ox)
            
        p_i = torch.sigmoid(self.pred(self.dropout_layer(xemb_i))).squeeze(-1)
        p_j = torch.sigmoid(self.pred(self.dropout_layer(xemb_j))).squeeze(-1)
        p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)

        out_dict = {
            "pred_i": p_i,
            "pred_j": p_j,
            "pred": p,
            "true_i": batch["responses"].float(),
            "true_j": batch["responses"].float(),
            "true": batch["responses"].float(),
        }
        return out_dict

    def loss(self, feed_dict, out_dict):
        pred_i = out_dict["pred_i"].flatten()
        pred_j = out_dict["pred_j"].flatten()
        pred = out_dict["pred"].flatten()

        true_i = out_dict["true_i"].flatten()
        true_j = out_dict["true_j"].flatten()
        true = out_dict["true"].flatten()

        mask_i = true_i > -1
        loss_i = self.loss_fn(pred_i[mask_i], true_i[mask_i])

        mask_j = true_j > -1
        loss_j = self.loss_fn(pred_j[mask_j], true_j[mask_j])

        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask])

        final_loss = loss + loss_i + loss_j

        return final_loss , len(pred[mask]), true[mask].sum().item()

class Blocks(Module):
    def __init__(self, device, embedding_size, num_attn_heads, dropout, rotary="none") -> None:
        super().__init__()
        self.device = device
        self.rotary  = rotary
        if self.rotary in ["kv"]:
            self.attn = MultiheadAttention(embedding_size, num_attn_heads, dropout=dropout, rotary=rotary)
        else:
            self.attn = MultiheadAttention(embedding_size, num_attn_heads, dropout=dropout)
        self.attn_dropout = Dropout(dropout)
        self.attn_layer_norm = LayerNorm(embedding_size)

        self.FFN = transformer_FFN(embedding_size, dropout)
        self.FFN_dropout = Dropout(dropout)
        self.FFN_layer_norm = LayerNorm(embedding_size)

    def forward(self, q=None, k=None, v=None, diff=None):
        causal_mask = ut_mask(self.device, seq_len = k.shape[1])
        attn_emb, _ = self.attn(q, k, v, diff=diff, mask=~causal_mask)
        attn_emb = self.attn_dropout(attn_emb)
        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb