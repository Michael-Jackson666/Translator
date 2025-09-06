import torch

# A compacts self-attention class
import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.randn(d_in, d_out))
        self.W_key   = nn.Parameter(torch.randn(d_in, d_out))
        self.W_value = nn.Parameter(torch.randn(d_in, d_out))
    
    def forward(self, x):
        # x shape: (seq_len, d_in)
        queries = x @ self.W_query  # (seq_len, d_out)
        keys    = x @ self.W_key    # (seq_len, d_out)
        values  = x @ self.W_value  # (seq_len, d_out)
        attn_scores = queries @ keys.T # omega: (seq_len, seq_len)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
            )  # (seq_len, seq_len)
        context_vec = attn_weights @ values  # (seq_len, d_out)
        return context_vec
    

#  A self-attention class using Pytorch's Linear layers
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec