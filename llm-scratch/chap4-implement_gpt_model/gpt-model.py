import torch
import torch.nn as nn

# example batch input
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

# add an important calss in chapter 3
# An efficient multi-head attention class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads 

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec

# config
# GPT-2 Small
GPT_CONFIG_124M = {
 "vocab_size": 50257, # Vocabulary size
 "context_length": 1024, # Context length
 "emb_dim": 768, # Embedding dimension
 "n_heads": 12, # Number of attention heads
 "n_layers": 12, # Number of layers
 "drop_rate": 0.1, # Dropout rate
 "qkv_bias": False # Query-Key-Value bias
}

# other bigger models
# GPT-2 Medium
GPT2_MEDIUM_CONFIG = {
 "vocab_size": 50257,
 "context_length": 1024,
 "emb_dim": 1024,
 "n_heads": 16,
 "n_layers": 24,
 "drop_rate": 0.1,
 "qkv_bias": False
}

# GPT-2 Large
GPT2_LARGE_CONFIG = {
 "vocab_size": 50257,
 "context_length": 1024,
 "emb_dim": 1280,
 "n_heads": 20,
 "n_layers": 36,
 "drop_rate": 0.1,
 "qkv_bias": False
}

# GPT-2 XL
GPT2_XL_CONFIG = {
 "vocab_size": 50257,
 "context_length": 1024,
 "emb_dim": 1600,
 "n_heads": 25,
 "n_layers": 48,
 "drop_rate": 0.1,
 "qkv_bias": False
}

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
    
# A layer normalization class
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_norm + self.shift
    
# An implementation of the GELU activation function
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
# A feed forward neural network module
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
    
# A neural network to illustrate shortcut connections
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # Compute the output of the current layer
            layer_output = layer(x)
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x
    
# the transformer block component of GPT
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x

# new in this file:
# The GPT model architecture implementation
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
# example usage
torch.manual_seed(123)
model = GPTModel(GPT2_XL_CONFIG)

out = model(batch)
# print("Input batch:\n", batch)
# print("\nOutput shape:", out.shape)
# print(out)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# compute number of parameters excluding output head
total_params_gpt2 = (
 total_params - sum(p.numel()
 for p in model.out_head.parameters())
)
print(f"Number of trainable parameters "
 f"considering weight tying: {total_params_gpt2:,}"
)

# compute number of parameters in feed forward modules
ff_params = sum(
    p.numel() for block in model.trf_blocks
    for p in block.ff.parameters()
)
print(f"Number of parameters in feed forward modules: {ff_params:,}")

# compute number of parameters in attention modules
attn_params = sum(
    p.numel() for block in model.trf_blocks
    for p in block.att.parameters()
)
print(f"Number of parameters in attention modules: {attn_params:,}")

# compute the memory requirements of the model
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 ** 2)
print(f"Approximate memory requirement for the model: {total_size_mb:.2f} MB")

# other bigger models
# GPT-2 Medium
GPT2_MEDIUM_CONFIG = {
 "vocab_size": 50257,
 "context_length": 1024,
 "emb_dim": 1024,
 "n_heads": 16,
 "n_layers": 24,
 "drop_rate": 0.1,
 "qkv_bias": False
}

# GPT-2 Large
GPT2_LARGE_CONFIG = {
 "vocab_size": 50257,
 "context_length": 1024,
 "emb_dim": 1280,
 "n_heads": 20,
 "n_layers": 36,
 "drop_rate": 0.1,
 "qkv_bias": False
}

# GPT-2 XL
GPT2_XL_CONFIG = {
 "vocab_size": 50257,
 "context_length": 1024,
 "emb_dim": 1600,
 "n_heads": 25,
 "n_layers": 48,
 "drop_rate": 0.1,
 "qkv_bias": False
}

# 只是上面的模型是不够的，展示目前最大的开源模型
# Meta Llama 3 70B (Instruct version)
# 性能极强的开源模型，应用广泛
LLAMA_3_70B_CONFIG = {
    "vocab_size": 128256,           # 词汇表大小显著增加，能更高效地处理多语言和代码
    "context_length": 8192,         # 上下文长度，是 GPT-2 的 8 倍 (社区已扩展至更长)
    "emb_dim": 8192,                # 嵌入维度 (隐藏层大小)，非常宽的网络
    "n_layers": 80,                 # 模型层数，非常深
    "n_heads": 64,                  # 主注意力头的数量
    "n_kv_heads": 8,                # 键/值头的数量 (使用了分组查询注意力 GQA 来加速推理)
    "ffn_dim_multiplier": 1.3,      # FFN (前馈网络) 中间层的维度乘数，不直接给出而是通过计算
    "rope_theta": 500000.0,         # RoPE 旋转位置编码的 base alpha
    "norm_eps": 1e-5,               # RMSNorm 中的 epsilon 值
    "total_params": "约 700 亿 (70B)" # 总参数量
}

# Mistral AI - Mixtral 8x22B
# 顶级的混合专家（MoE）开源模型，推理效率高
MIXTRAL_8x22B_CONFIG = {
    "vocab_size": 32768,
    "context_length": 65536,      # 64K 超长上下文
    "emb_dim": 6144,
    "n_layers": 56,
    "n_heads": 48,
    "n_kv_heads": 8,              # 同样使用了 GQA
    "ffn_dim": 16384,             # 直接给出了 FFN 中间层大小
    "moe_num_experts": 8,         # 共有 8 个专家网络
    "moe_top_k": 2,               # 每个 token 会选择最相关的 2 个专家来处理
    "total_params": "约 1410 亿 (141B)", # 总参数量 (很多是稀疏的)
    "active_params": "约 390 亿 (39B)"  # 每次前向传播时实际激活的参数量
}

# xAI Grok-1 (Base Model)
# 目前参数量最大的开源模型
GROK_1_CONFIG = {
    "vocab_size": 128000,
    "context_length": 8192,
    "emb_dim": 6144,
    "n_layers": 64,
    "n_heads": 48,
    "n_kv_heads": 8,              # 同样使用了 GQA
    "ffn_dim_multiplier": 8/3,    # FFN 中间层的维度乘数
    "total_params": "约 3140 亿 (314B)" # 总参数量
}