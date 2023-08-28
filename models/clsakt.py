import torch

from torch.nn import Module, Embedding, Linear, LayerNorm, Dropout, BCELoss, ModuleList, Sequential, GELU
from .modules import transformer_FFN, pos_encode, ut_mask, get_clones, MultiheadAttention, CL4KTTransformerLayer
from .rpe import SinusoidalPositionalEmbeddings 

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

from IPython import embed

import torch.nn.functional as F
from torch.nn import CosineSimilarity
from .sakt import SAKT
from .contrastiveLoss import ContrastiveLoss

class CLSAKT(SAKT):
    def __init__(self, *args, **kwargs):
        super(CLSAKT, self).__init__(*args, **kwargs)


        self.args = kwargs
        self.hidden_size = self.args["hidden_size"]

        self.kq_same = self.args["kq_same"]
        self.final_fc_dim = self.args["final_fc_dim"]
        self.d_ff = self.args["d_ff"]

        self.reg_cl = self.args["reg_cl"]
        self.negative_prob = self.args["negative_prob"]
        self.hard_negative_weight = self.args["hard_negative_weight"]
        self.only_rp = self.args["only_rp"]
        self.choose_cl = self.args["choose_cl"]
        self.de = self.args["de_type"].split('_')[0]
        self.token_num = int(self.args["de_type"].split('_')[1])
        # self.diff_as_loss_weight = diff_as_loss_weight

        # if self.de in ["sde", "lsde"]:
        #     diff_vec = torch.from_numpy(SinusoidalPositionalEmbeddings(
        #         2*(self.token_num+1), self.hidden_size)).to(device)
        #     self.diff_emb = Embedding.from_pretrained(diff_vec, freeze=True)
        #     rotary = "none"
        # elif self.de in ["rde", "lrde"]:
        #     rotary = "qkv"
        # else:
        #     rotary = "none"

        self.question_encoder = ModuleList(
            [
                CL4KTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    n_heads=self.num_attn_heads,
                    dropout=self.dropout,
                    kq_same=self.kq_same,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.interaction_encoder = ModuleList(
            [
                CL4KTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    n_heads=self.num_attn_heads,
                    dropout=self.dropout,
                    kq_same=self.kq_same,
                    # rotary=rotary,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.knoweldge_retriever = ModuleList(
            [
                CL4KTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    n_heads=self.num_attn_heads,
                    dropout=self.dropout,
                    kq_same=self.kq_same,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.out = Sequential(
            Linear(2 * self.hidden_size, self.final_fc_dim),
            GELU(),
            Dropout(self.dropout),
            Linear(self.final_fc_dim, self.final_fc_dim // 2),
            GELU(),
            Dropout(self.dropout),
            Linear(self.final_fc_dim // 2, 1),
        )

        if self.diff_as_loss_weight:
            self.loss_fn = torch.nn.BCELoss(reduction="none")

        self.contrastive_loss = torch.nn.CrossEntropyLoss(reduction="mean")

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

    def forward(self, feed_dict):
        q = feed_dict["skills"][:, :-1]
        r = feed_dict["responses"][:, :-1]
        qry = feed_dict["skills"][:, 1:]
        pos = feed_dict["position"][:, :-1]
        diff = feed_dict["sdiff"][:, :-1]
        
        if self.token_num < 1000:
            boundaries = torch.linspace(0, 1, steps=self.token_num+1)                
            diff = torch.bucketize(diff, boundaries)
            diff_ox = torch.where(r==0 , (diff-(self.token_num+1)) * (r > -1).int(), diff * (r > -1).int())  
        else:
            diff = diff * 100
            diff_ox = torch.where(r==0 , (diff-(100+1)) * (r > -1).int(), diff * (r > -1).int())
            
        qshftemb, xemb, demb = self.base_emb(q, r, qry, pos, diff)
        
        for i in range(self.num_blocks): #sakt's num_blocks = 1
            xemb = self.blocks[i](qshftemb, xemb, xemb, diff_ox)
        
        # Compute positive and negative pairs
        positive_pairs = xemb[::2]
        negative_pairs = xemb[1::2]

        # Compute cosine similarity between positive and negative pairs
        cosine_sim = CosineSimilarity(dim=1)
        similarity = cosine_sim(positive_pairs, negative_pairs)

        # Labels for contrastive loss (1 for positive pairs, 0 for negative pairs)
        labels = torch.ones_like(similarity)

        # Compute contrastive loss
        contrastive_loss = self.contrastive_loss(similarity, labels)

        p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
        print("p", p.shape)
        out_dict = {
            "pred": p,
            "true": feed_dict["responses"][:, 1:].float(),
            "contrastive_loss": contrastive_loss,
        }
        return out_dict

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        mask = true > -1
        loss_before = self.loss_fn(pred[mask], true[mask])
        contrastive_loss = out_dict["contrastive_loss"]
        loss = contrastive_loss # + loss_before
        return loss , len(pred[mask]), true[mask].sum().item()

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