#!/usr/bin/env python3
"""
批量翻译脚本
使用训练好的Transformer模型进行批量英语到法语翻译
"""

import torch
from torch import nn
import math
import os
import collections

# === 词汇表类 ===
class Vocab:
    """词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 统计词频
        counter = collections.Counter()
        for token_list in tokens:
            counter.update(token_list)
        
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        # 构建词汇表：保留词元 + 高频词元
        self.idx_to_token = list(reserved_tokens)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property
    def unk(self):  # 未知词元的索引为0
        return 0
    
    @property 
    def token_freqs(self):
        return self._token_freqs

# === 工具函数 ===
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# === 注意力机制 ===
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        values = self.W_v(values)

        queries = transpose_qkv(queries, self.num_heads)
        keys = transpose_qkv(keys, self.num_heads)
        values = transpose_qkv(values, self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

# === 位置编码 ===
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

# === Transformer组件 ===
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class TransformerEncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block_"+str(i),
                TransformerEncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

class TransformerDecoderBlock(nn.Module):
    """解码器中第i个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(TransformerDecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block_"+str(i),
                TransformerDecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

# === 批量翻译器类 ===
class BatchTranslator:
    """批量翻译器"""
    
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.net, self.src_vocab, self.tgt_vocab = self._load_model()
    
    def _truncate_pad(self, line, num_steps, padding_token):
        if len(line) > num_steps:
            return line[:num_steps]
        return line + [padding_token] * (num_steps - len(line))
    
    def _load_model(self):
        """加载训练好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        print(f"正在加载模型: {self.model_path}")
        # 加载模型文件，兼容不同版本的PyTorch
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 检查是否是完整的checkpoint格式
        if 'model_state_dict' in checkpoint:
            src_vocab = checkpoint['src_vocab']
            tgt_vocab = checkpoint['tgt_vocab']
            
            # 使用训练时的参数重建模型
            num_hiddens, num_layers, dropout = 32, 2, 0.1
            ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
            key_size, query_size, value_size = 32, 32, 32
            norm_shape = [32]
            
            encoder = TransformerEncoder(
                len(src_vocab), key_size, query_size, value_size, num_hiddens,
                norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                num_layers, dropout)
            decoder = TransformerDecoder(
                len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
                norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                num_layers, dropout)
            net = EncoderDecoder(encoder, decoder)
            net.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError("不支持的模型格式")
        
        net.to(self.device)
        net.eval()
        print(f"模型已加载到 {self.device}")
        return net, src_vocab, tgt_vocab
    
    def translate_single(self, sentence):
        """翻译单个句子"""
        # 确保句子以标点符号结尾
        if not sentence.strip().endswith(('.', '!', '?')):
            sentence = sentence.strip() + ' .'
        
        # 预处理: 转换为小写并分词
        src_tokens = self.src_vocab[sentence.lower().split(' ')] + [self.src_vocab['<eos>']]
        enc_valid_len = torch.tensor([len(src_tokens)], device=self.device)
        
        # 截断或填充到固定长度
        num_steps = 15
        src_tokens = self._truncate_pad(src_tokens, num_steps, self.src_vocab['<pad>'])
        enc_X = torch.unsqueeze(
            torch.tensor(src_tokens, dtype=torch.long, device=self.device), dim=0)
        
        # 编码
        enc_outputs = self.net.encoder(enc_X, enc_valid_len)
        dec_state = self.net.decoder.init_state(enc_outputs, enc_valid_len)
        
        # 解码
        dec_X = torch.unsqueeze(torch.tensor(
            [self.tgt_vocab['<bos>']], dtype=torch.long, device=self.device), dim=0)
        output_seq = []
        
        for _ in range(num_steps):
            Y, dec_state = self.net.decoder(dec_X, dec_state)
            dec_X = Y.argmax(dim=2)
            pred = dec_X.squeeze(dim=0).type(torch.int32).item()
            if pred == self.tgt_vocab['<eos>']:
                break
            output_seq.append(pred)
        
        return ' '.join(self.tgt_vocab.to_tokens(output_seq))
    
    def translate_batch(self, sentences, show_progress=True):
        """批量翻译多个句子"""
        results = []
        total = len(sentences)
        
        print(f"开始批量翻译 {total} 个句子...")
        
        for i, sentence in enumerate(sentences):
            try:
                if show_progress:
                    print(f"进度: {i+1}/{total} - 正在翻译: {sentence[:30]}{'...' if len(sentence) > 30 else ''}")
                
                translation = self.translate_single(sentence)
                results.append(translation)
                
            except Exception as e:
                print(f"翻译句子 '{sentence}' 时出错: {e}")
                results.append(f"[翻译失败: {str(e)}]")
        
        print(f"批量翻译完成！成功翻译 {len([r for r in results if not r.startswith('[翻译失败')])} 个句子")
        return results

# === 便利函数 ===
def translate_sentences(model_path, sentences):
    """
    批量翻译句子的便利函数
    
    Args:
        model_path: 模型文件路径
        sentences: 要翻译的英语句子列表
    
    Returns:
        翻译后的法语句子列表
    """
    translator = BatchTranslator(model_path)
    return translator.translate_batch(sentences)

def translate_from_file(model_path, input_file, output_file=None):
    """
    从文件读取句子进行批量翻译
    
    Args:
        model_path: 模型文件路径
        input_file: 输入文件路径（每行一个英语句子）
        output_file: 输出文件路径（可选，默认为输入文件名_translated.txt）
    """
    # 读取输入文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误: 输入文件 {input_file} 不存在！")
        return
    
    # 设置输出文件名
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_translated.txt"
    
    # 批量翻译
    translator = BatchTranslator(model_path)
    translations = translator.translate_batch(sentences)
    
    # 保存结果
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for eng, fra in zip(sentences, translations):
                f.write(f"EN: {eng}\n")
                f.write(f"FR: {fra}\n")
                f.write("-" * 50 + "\n")
        
        print(f"翻译结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"保存文件时出错: {e}")

def main():
    """主函数：演示批量翻译功能"""
    model_path = 'transformer_fra_eng.pth'
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在！")
        print("请先运行 transformer-d2l.py 进行训练。")
        return
    
    # 示例英语句子
    test_sentences = [
        'hello .',
        'how are you ?',
        'i am fine .',
        'good morning .',
        'see you later .',
        'i love you .',
        'thank you .',
        'what is your name ?',
        'i am learning french .',
        'this is a beautiful day .',
        'where are you from ?',
        'i like to read books .',
        'the weather is nice today .',
        'can you help me ?',
        'i want to go home .'
    ]
    
    print("=" * 60)
    print("🌍 批量翻译示例")
    print("使用训练好的Transformer模型进行英语到法语翻译")
    print("=" * 60)
    
    try:
        # 创建批量翻译器
        translator = BatchTranslator(model_path)
        
        # 进行批量翻译
        print(f"\n开始翻译 {len(test_sentences)} 个句子...")
        results = translator.translate_batch(test_sentences, show_progress=True)
        
        # 显示结果
        print(f"\n{'='*60}")
        print("📋 翻译结果对比:")
        print(f"{'='*60}")
        
        for i, (eng, fra) in enumerate(zip(test_sentences, results), 1):
            print(f"{i:2d}. 英语: {eng}")
            print(f"    法语: {fra}")
            print("-" * 50)
        
        # 保存到文件
        output_file = "batch_translation_results.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("批量翻译结果\n")
            f.write("=" * 50 + "\n\n")
            for i, (eng, fra) in enumerate(zip(test_sentences, results), 1):
                f.write(f"{i:2d}. 英语: {eng}\n")
                f.write(f"    法语: {fra}\n")
                f.write("-" * 30 + "\n")
        
        print(f"✅ 翻译结果已保存到: {output_file}")
        
        # 演示从文件翻译的功能
        print(f"\n{'='*60}")
        print("📄 文件翻译演示:")
        print(f"{'='*60}")
        
        # 创建示例输入文件
        input_file = "example_input.txt"
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write("hello world .\n")
            f.write("i am happy .\n")
            f.write("good luck .\n")
            f.write("have a nice day .\n")
        
        print(f"创建示例输入文件: {input_file}")
        translate_from_file(model_path, input_file)
        
    except Exception as e:
        print(f"❌ 批量翻译时出错: {e}")
        print("请确保模型文件正确且完整")

# === 命令行接口 ===
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # 命令行模式
        if sys.argv[1] == "--file" and len(sys.argv) >= 3:
            # 从文件翻译
            input_file = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else None
            model_path = sys.argv[4] if len(sys.argv) > 4 else 'transformer_fra_eng.pth'
            
            print(f"从文件 {input_file} 进行批量翻译...")
            translate_from_file(model_path, input_file, output_file)
            
        elif sys.argv[1] == "--sentences":
            # 从命令行参数翻译
            sentences = sys.argv[2:]
            model_path = 'transformer_fra_eng.pth'
            
            if not sentences:
                print("错误: 请提供要翻译的句子")
                print("用法: python batch_translate.py --sentences \"句子1\" \"句子2\" ...")
                sys.exit(1)
            
            print(f"翻译 {len(sentences)} 个句子...")
            results = translate_sentences(model_path, sentences)
            
            for eng, fra in zip(sentences, results):
                print(f"英语: {eng}")
                print(f"法语: {fra}")
                print("-" * 30)
                
        else:
            print("用法:")
            print("  演示模式: python batch_translate.py")
            print("  文件模式: python batch_translate.py --file input.txt [output.txt] [model.pth]")
            print("  句子模式: python batch_translate.py --sentences \"句子1\" \"句子2\" ...")
    else:
        # 演示模式
        main()
