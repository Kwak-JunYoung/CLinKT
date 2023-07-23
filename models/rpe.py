# Rotary Positional Embeddings => RPE
import torch
import torch.nn as nn
from torch.nn import Module, Embedding, Linear, MultiheadAttention, LayerNorm, Dropout, CosineSimilarity
import torch.linalg as la
import numpy as np
from IPython import embed
from time import time 
from einops import rearrange, repeat
from torch.nn.init import xavier_uniform_, constant_
import math 

#https://nn.labml.ai/transformers/rope/index.html
#https://github.com/JunnYu/RoFormer_pytorch/blob/roformer_v2/src/roformer/modeling_roformer.py
#https://github.com/lucidrains/rotary-embedding-torch/blob/517ee2cfeb10602032ef9d282c19851e19dd8943/rotary_embedding_torch/rotary_embedding_torch.py#L57

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d, base = 10_000, device = None):
        """
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()

        self.base = base
        self.d = d
        self.freqs = None 
        self.device = device
        self._build_cache()

    def _build_cache(self):
        """
        x: [batch, head, seq_len, head_dim]
        Cache $\cos$ and $\sin$ values
        """
        # pos = torch.arange(seq_len).to(self.device)
        pos = torch.tensor([1]).to(self.device)

        # Return if cache is already built
        if self.freqs is not None and seq_len <= self.freqs.shape[0]:
            return
        # Get sequence length
        self.freqs = 1./  (self.base **(torch.arange(0, self.d, 2)[:(self.d//2)].float().to(self.device)/self.d))
        # self.freqs = self.freqs*10**4
        #pos @ self.freqs.T
        self.freqs = torch.einsum("..., f -> ... f", pos.type(self.freqs.dtype), self.freqs) # seq_len, dim//2 
        #seq_len, dim//2 -> seq_len, dim
        self.freqs = repeat(self.freqs, "... n -> ... (n r)", r=2)
        #unsqueeze
        # self.freqs = rearrange(self.freqs, "n d -> () () n d") # 1, 1, seq_len, dim 

    def rotate_half(self, x):
        x = rearrange(x, '... (d r) -> ... d r', r = 2)
        x1, x2 = x.unbind(dim = -1)
        x = torch.stack((-x2, x1), dim = -1)
        return rearrange(x, '... d r -> ... (d r)')

    def forward(self, t, diff, start_index = 0):
        b, head_num, s, head_dim = t.shape
        # t : [batch, head, seq_len, head_dim]
        # diff : [batch, seq_len]
        # self.freqs : [max_pos, head_dim]
        self.freqs = self.freqs.to(t) # device matching 
        diff_freqs = (diff*100).repeat(1,head_num*head_dim).view(b, head_num, s, head_dim)*self.freqs.squeeze() #[ batch, head, seq_len, head_dim ]
        rot_dim = self.freqs.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        # none, t, none
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * diff_freqs.cos()) + (self.rotate_half(t) * diff_freqs.sin())
        return torch.cat((t_left, t, t_right), dim = -1)

class MultiHeadAttention_Rotary(nn.Module):
    """
    ## Multi-head attention with rotary positional embeddings
    We override [multi-head attention from original transformer](../mha.html).
    """
    def __init__(self, d_model: int, heads: int, dropout_prob: float, max_p: int, bias=True, device=None):
        super().__init__()

        self.num_heads = heads
        self.head_dim = d_model // heads
        self.proj_bias = bias
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model, bias=bias)
        self.key = nn.Linear(d_model, d_model, bias=bias)
        self.value = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Rotary positional embedding layers
        self.rpe = RotaryPositionalEmbeddings(self.head_dim, max_p, device = device)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.query.weight)
        xavier_uniform_(self.key.weight)
        xavier_uniform_(self.value.weight)

        if self.proj_bias:
            constant_(self.query.bias, 0.)
            constant_(self.key.bias, 0.)
            constant_(self.value.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, diff, mask = None):
        batch_size = q.size(0)
        q = self.query(q).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.key(k).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.value(v).view(batch_size, -1, self.num_heads, self.head_dim)

        q = self.rpe(q.transpose(1, 2), diff) # [batch_size, head, len_q,  head_dim]
        # q = q.transpose(1, 2) 
        k = self.rpe(k.transpose(1, 2), diff) # [batch_size, head, len_k,  head_dim]
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = attn / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e32)
        attn = self.dropout(torch.softmax(attn, dim = -1)) # [batch_size, head, len_q,  len_k]

        v = v.transpose(1, 2) # [batch_size, head, len_v,  head_dim]
        # v = self.rpe(v.transpose(1, 2), diff) # [batch_size, head, len_k,  head_dim]
        output = torch.matmul(attn, v) # [batch_size, head, len_q,  head_dim]
        output = output.permute(0, 2, 1, 3).contiguous() #x = [batch size, query len, n heads, head dim]
        output = output.view(batch_size, -1, self.d_model) #x = [batch size, query len, hid dim]
        output = self.out_proj(output)

        return output