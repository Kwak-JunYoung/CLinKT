import torch

from torch.nn import Module, Embedding, Linear, LayerNorm, Dropout, BCELoss
from .modules import transformer_FFN, pos_encode, ut_mask, get_clones, MultiheadAttention
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
        self.contrastive_loss = ContrastiveLoss()

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