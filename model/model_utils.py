#@title Import packages
import torch
import torch.nn as nn
import torch.nn.functional as F 
import einops
from transformers.modeling_outputs import Seq2SeqLMOutput
from PIL import Image
import numpy as np


class Transformer(nn.Module):
    """Transformer Encoder 
    Args:
        embedding_dim: dimension of embedding
        n_heads: number of attention heads
        n_layers: number of attention layers
        feedforward_dim: hidden dimension of MLP layer
    Returns:
        Transformer embedding of input
    """
    def __init__(self, embedding_dim=256, n_heads=4, n_layers=3, feedforward_dim=1024, dropout=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.feedforward_dim = feedforward_dim
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=self.n_heads,
                dim_feedforward=self.feedforward_dim,
                activation=F.gelu,
                batch_first=True,
                dropout=dropout,
            ),
            num_layers=n_layers,
        )

    def forward(self, x):
        return self.transformer(x)
    


class MLPs(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLPs, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(dim_in, 640, 1),
            nn.ReLU(),
            nn.Conv1d(640, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, dim_out, 1)
            )
    def forward(self, x):
        return self.mlp(x)
    

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
            Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class FilterLayer(nn.Module):
    '''
        https://arxiv.org/abs/2202.13556
    '''
    def __init__(self, max_seq_length, hidden_size, hidden_dropout_prob):
        super(FilterLayer, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, max_seq_length//2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_tensor):
        input_tensor = input_tensor.transpose(2, 1).contiguous()
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # hidden_states = sequence_emb_fft + input_tensor
        return hidden_states.transpose(2, 1).contiguous()

#  Transformer Customization
future_mask = torch.triu(torch.zeros([1024, 1024]).fill_(float("-inf")), 1)
def scaled_dot_product_attention(q, k, v, key_padding_mask=None, causal=False, return_attn = False):
    d_head = q.size(-1)
    s = einops.einsum(q, k, "n tl dh, n sl dh -> n tl sl") / d_head ** 0.5
    if key_padding_mask is not None:
        s = s.masked_fill(
            key_padding_mask.unsqueeze(1).to(torch.bool),
            float("-inf"),
        )
    if causal:
        attn_mask = future_mask[: s.size(1), : s.size(2)].to(s)
        s += attn_mask.unsqueeze(0)
    a = F.softmax(s, dim=-1, dtype=torch.float32).type_as(s)
    if return_attn:
        return a
    return einops.einsum(a, v, "n tl sl, n sl dh -> n tl dh")

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
                
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.apply(self._init_weights)
        self._init_mimetic()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def _init_mimetic(self):
        self.q_proj /= self.d_head
        self.k_proj /= self.d_head
        self.v_proj *= (1/self.d_model)
        self.o_proj *= (-1/self.d_model)

    
    def forward(self, q, k, v, key_padding_mask=None, causal=False):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = einops.rearrange(q, "b tl (nh dh) -> (b nh) tl dh", nh=self.n_heads)
        k = einops.rearrange(k, "b sl (nh dh) -> (b nh) sl dh", nh=self.n_heads)
        v = einops.rearrange(v, "b sl (nh dh) -> (b nh) sl dh", nh=self.n_heads)
        if key_padding_mask is not None:
            key_padding_mask = einops.repeat(
                key_padding_mask, "b sl -> (b nh) sl", nh=self.n_heads)
        o = scaled_dot_product_attention(q, k, v, key_padding_mask, causal)
        o = einops.rearrange(o, "(b nh) tl dh -> b tl (nh dh)", nh=self.n_heads)
        return self.o_proj(o)

class HandmadeTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn, p_drop, is_decoder=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.self_attn = MultiheadAttention(d_model, n_heads)
        self.self_attn_drop = nn.Dropout(p_drop)
        self.self_attn_ln = nn.LayerNorm(d_model)
        if is_decoder:
            self.cross_attn = MultiheadAttention(d_model, n_heads)
            self.cross_attn_drop = nn.Dropout(p_drop)
            self.cross_attn_ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.ffn_drop = nn.Dropout(p_drop)
        self.ffn_ln = nn.LayerNorm(d_model)
    
    def forward(self, x, padding_mask=None, encoder_out=None, encoder_padding_mask=None):
        residual = x
        x = self.self_attn(x, x, x, padding_mask, causal=self.is_decoder)
        x = self.self_attn_drop(x)
        x = self.self_attn_ln(x + residual)

        if self.is_decoder:
            residual = x
            x = self.cross_attn(x, encoder_out, encoder_out, encoder_padding_mask)
            x = self.cross_attn_drop(x)
            x = self.cross_attn_ln(x + residual)

        residual = x
        x = self.fc2(F.relu(self.fc1(x)))
        x = self.ffn_drop(x)
        x = self.ffn_ln(x + residual)
        return x



from typing import  Optional, Tuple
Tensor = torch.Tensor

def simplified_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    only_attn: bool=True
    ) -> Tuple[Tensor, Optional[Tensor]]:

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    head_dim = embed_dim // num_heads

    # compute in-projection
    q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    # reshape q, k, v for multihead attention and make em batch first
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

    # (deep breath) calculate attention
    attn_map = scaled_dot_product_attention(q, k, v, return_attn=True)
    if only_attn:
        return attn_map
    embeddings = einops.einsum(attn_map, v, "n tl sl, n sl dh -> n tl dh")
    embeddings = einops.rearrange(embeddings, "(b n) tl dh -> b tl (n dh)", b = bsz)
    return embeddings, attn_map



def attn_diverse_loss(attn, th=2):
    sim_sum = 0
    counter = 1e-6
    for i in range(len(attn)):
        mask0 = attn[i].mean(dim=1).squeeze()
        n_tokens = mask0.shape[-1]
        threshold = th/n_tokens
        score0 = torch.mean(mask0, dim=1, keepdim=True)
        mask0 = (mask0 > threshold) * (mask0)
        score0 = (score0 > threshold) * (score0)
        sim = F.cosine_similarity(score0, mask0, dim=-1)
        sim = sim.mean()
        sim_sum += sim
        counter += 1
    sim_sum = sim_sum / counter
    # print(sim_sum)
    return sim_sum

