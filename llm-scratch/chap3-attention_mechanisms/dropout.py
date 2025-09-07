import torch
import torch.nn as nn

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

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

d_in, d_out = inputs.shape[1], 2
torch.manual_seed(123)
sa_v2 = SelfAttention_v2(d_in, d_out)

queries = sa_v2.W_query(inputs)
keys  = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T

context_length = attn_scores.shape[0]
# 更加高效的掩码技术（用-inf）
mask = torch.triu(torch.ones(context_length, context_length), diagonal = 1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)

# dropout技术实现
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) 
example = torch.ones(6, 6) 
# print(dropout(example))

# print(dropout(attn_weights))

batch = torch.stack((inputs, inputs), dim=0) # 两个文本输入，复制了一遍
print("Batch shape:", batch.shape)

# A compact causal attention class
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, kqv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=kqv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=kqv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=kqv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), 
            diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x)    
        queries = self.W_query(x)  
        values  = self.W_value(x)  

        attn_scores = queries @ keys.transpose(1, 2) 
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec
    

# 测试
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, dropout=0.0)
context_vec = ca(batch)
print("Context vector shape:", context_vec.shape)
print("Context vector:", context_vec)