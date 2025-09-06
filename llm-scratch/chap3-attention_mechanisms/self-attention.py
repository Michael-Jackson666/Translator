import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# 计算注意力分数和权重
attn_scores = inputs @ inputs.T
attn_weights = torch.softmax(attn_scores, dim=1)
# print("Attention weights:\n", attn_weights)
# print("All row sums:", attn_weights.sum(dim=1))

all_context_vecs = attn_weights @ inputs
# print("All context vectors:\n", all_context_vecs)

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
keys = inputs @ W_key
value = inputs @ W_value

attn_scores_2 = query_2 @ keys.T

d_k = keys.shape[-1] # d_k = 2
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
# print(attn_weights_2)

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

# torch.manual_seed(123)
# sa_v1 = SelfAttention_v1(d_in, d_out)
# print(sa_v1(inputs))

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
    
# torch.manual_seed(123)
# sa_v2 = SelfAttention_v2(d_in, d_out)
# print(sa_v2(inputs))
    
# musked attention
queries = sa_v2.W_query(inputs)
keys  = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print("Attention weights before masking:\n", attn_weights)