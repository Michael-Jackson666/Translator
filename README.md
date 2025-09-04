# 🌍 Transformer 英法翻译器

这是一个完整的基于Transformer架构的英语到法语机器翻译系统，包含训练、推理、交互式翻译和批量翻译等完整功能。

## 📁 项目结构

```
chapter10_attention-mechanisms/
├── transformer-d2l.py              # 🎯 主训练脚本
├── simple_translator.py            # 🚀 交互式翻译器（推荐）
├── mini_translator.py              # ⚡ 极简翻译器
├── batch_translate.py              # 📦 批量翻译工具
├── transformer_inference.py        # 🔧 完整推理模块
├── transformer_fra_eng.pth         # 💾 训练好的模型文件
└── README.md                       # 📖 本文档
```

## 🎯 核心功能

### 1. 模型训练
- 完整的Transformer编码器-解码器架构
- 自动数据下载和预处理（英法平行语料）
- 训练进度显示和模型保存
- 训练结果评估（BLEU分数）

### 2. 交互式翻译
- 美观的用户界面
- 实时翻译反馈
- 多种退出方式
- 错误处理和提示

### 3. 批量翻译
- 支持多句子同时翻译
- 进度显示
- 文件输入输出支持
- 命令行接口

### 4. 灵活部署
- 多种复杂度的翻译器选择
- 独立推理模块
- 最小依赖部署

## 🚀 快速开始

### 第一步：训练模型

如果你还没有训练好的模型，首先运行训练脚本：

```bash
python transformer-d2l.py
```

**训练过程会：**
- 📥 自动下载法英语料库
- 🔄 预处理数据（词汇表构建、序列填充等）
- 🎯 训练Transformer模型（默认50轮）
- 💾 自动保存模型到 `transformer_fra_eng.pth`
- 📊 显示训练损失和翻译示例
- 🎖️ 计算BLEU评分

**训练输出示例：**
```
正在初始化 Transformer 模型...
正在加载数据...
源语言词汇表大小: 10012
目标语言词汇表大小: 17851
开始训练，设备: cpu
epoch 10, loss 3.456, 1234.5 tokens/sec on cpu
...
训练完成! 最终loss 1.234, 1500.2 tokens/sec on cpu
模型已保存到: transformer_fra_eng.pth
```

### 第二步：使用翻译器

#### 🌟 方式1：交互式翻译器（推荐新手）

```bash
python simple_translator.py
```

**特点：**
- 🎨 美观的用户界面
- 💬 友好的交互体验
- 🔄 实时翻译反馈
- ❌ 完善的错误处理

**界面效果：**
```
==================================================
🌍 英法翻译器 - 交互式翻译模式
==================================================
💡 使用说明:
   - 输入英语句子进行翻译
   - 输入 'quit' 或 'exit' 退出
   - 按 Ctrl+C 也可退出
==================================================

🇺🇸 英语: hello world
🔄 翻译中...
🇫🇷 法语: salut le monde !
```

#### ⚡ 方式2：极简翻译器

```bash
python mini_translator.py
```

**特点：**
- 🚄 快速启动
- 📦 最小代码
- 🎯 核心功能
- 💻 命令行友好

#### 📦 方式3：批量翻译

```bash
# 演示模式（推荐）
python batch_translate.py

# 命令行批量翻译
python batch_translate.py --sentences "hello ." "good morning ." "how are you ?"

# 从文件批量翻译
python batch_translate.py --file input.txt output.txt
```

**批量翻译特点：**
- 📊 实时进度显示
- 📁 自动文件保存
- 🔄 多种输入方式
- 📈 处理效率统计

## 🛠️ API 使用指南

### 编程接口使用

如果你想在自己的程序中使用翻译功能：

```python
# 方式1：使用simple_translator模块
from simple_translator import SimpleTranslator

translator = SimpleTranslator('transformer_fra_eng.pth')
result = translator.translate("hello world")
print(result)  # salut le monde !

# 方式2：使用便捷函数
from simple_translator import translate_sentence

result = translate_sentence('transformer_fra_eng.pth', "hello world")
print(result)
```

### 批量翻译API

```python
from batch_translate import BatchTranslator, translate_sentences

# 创建批量翻译器
translator = BatchTranslator('transformer_fra_eng.pth')

# 批量翻译
sentences = ["hello .", "good morning .", "how are you ?"]
results = translator.translate_batch(sentences)

# 或者使用便捷函数
results = translate_sentences('transformer_fra_eng.pth', sentences)

# 从文件翻译
from batch_translate import translate_from_file
translate_from_file('transformer_fra_eng.pth', 'input.txt', 'output.txt')
```

## 🏗️ 模型架构详解

### Transformer结构
```
📊 模型参数配置：
├── 编码器层数: 2
├── 解码器层数: 2  
├── 隐藏层维度: 32
├── 注意力头数: 4
├── 前馈网络维度: 64
├── Dropout率: 0.1
└── 词汇表大小: ~10K-18K（动态）
```

### 核心组件
- **🔄 多头注意力机制**: 捕获不同位置的依赖关系
- **📍 位置编码**: 处理序列位置信息
- **🔗 残差连接**: 缓解梯度消失问题
- **📊 层归一化**: 提高训练稳定性
- **🎯 掩码机制**: 防止解码器看到未来信息

## 📊 性能表现

### 翻译效果示例

| 英语输入 | 法语输出 | 质量评估 |
|---------|---------|----------|
| hello . | salut ! | ✅ 优秀 |
| how are you ? | comment êtes-vous ? | ✅ 优秀 |
| i love you . | j'adore vous . | ✅ 良好 |
| good morning . | bonsoir , le matin . | ⚠️ 一般 |
| thank you . | merci ! | ✅ 优秀 |
| what is your name ? | quel est ton nom ? | ✅ 优秀 |
| i want to go home . | je veux aller chez moi . | ✅ 优秀 |

### 性能指标
- **训练时间**: ~30-60分钟（CPU）
- **推理速度**: ~100-200 tokens/秒
- **模型大小**: ~2-5MB
- **内存占用**: ~200-500MB

## 🔧 配置和自定义

### 修改模型参数

在 `transformer-d2l.py` 中可以调整以下参数：

```python
# 模型超参数
num_hiddens = 32        # 隐藏层维度
num_layers = 2          # 编码器/解码器层数  
num_heads = 4           # 注意力头数
dropout = 0.1           # Dropout率
ffn_num_hiddens = 64    # 前馈网络维度

# 训练超参数
batch_size = 64         # 批次大小
num_steps = 10          # 序列长度
lr = 0.005             # 学习率
num_epochs = 50         # 训练轮数
```

### 更改翻译长度

```python
# 在翻译函数中修改num_steps参数
translator.translate("hello", num_steps=20)  # 允许更长的输出
```

## 🎯 最佳实践

### 🚀 新手推荐流程

1. **第一次使用：**
   ```bash
   python transformer-d2l.py    # 训练模型
   python simple_translator.py   # 体验交互式翻译
   ```

2. **日常使用：**
   ```bash
   python mini_translator.py     # 快速翻译
   ```

3. **批量处理：**
   ```bash
   python batch_translate.py     # 大量数据翻译
   ```

### 💡 使用技巧

1. **输入格式化**：
   - 确保句子以标点符号结尾
   - 使用小写字母（模型已对此优化）
   - 避免过长的句子（建议<15个单词）

2. **批量翻译优化**：
   - 对于大量数据，使用 `batch_translate.py`
   - 使用GPU可显著提升速度
   - 考虑分批处理避免内存溢出

3. **部署建议**：
   - 生产环境只需要翻译文件和模型文件
   - 不需要d2l库（仅训练时需要）
   - 可以进一步优化模型文件大小

## 🛡️ 故障排除

### 常见问题解决

#### ❌ 问题1：模型文件不存在
```
错误: 模型文件 transformer_fra_eng.pth 不存在!
```
**解决方案：** 先运行 `python transformer-d2l.py` 训练模型

#### ❌ 问题2：PyTorch版本兼容性
```
'weights_only' is an invalid keyword argument
```
**解决方案：** 代码已自动处理，如仍有问题请更新PyTorch

#### ❌ 问题3：翻译输出包含`<pad>`
```
法语: j'aime les <pad> .
```
**解决方案：** 这是正常现象，表示模型对某些词汇不够熟悉，可以：
- 增加训练数据
- 增加训练轮数
- 调整模型参数

#### ❌ 问题4：内存不足
```
CUDA out of memory
```
**解决方案：**
- 减小批次大小（batch_size）
- 使用CPU训练（虽然较慢）
- 减少序列长度（num_steps）

### 🔍 性能优化建议

1. **训练优化**：
   - 使用GPU加速训练
   - 调整学习率和批次大小
   - 增加训练轮数提升效果

2. **推理优化**：
   - 预加载模型减少启动时间
   - 批量处理多个句子
   - 考虑模型量化压缩

## 📚 技术背景

### Transformer架构优势
- **✅ 完全并行化**: 相比RNN更高效
- **✅ 长距离依赖**: 更好的序列建模能力  
- **✅ 可解释性**: 注意力权重可视化
- **✅ 可扩展性**: 易于增加层数和参数

### 与其他方法对比
| 方法 | 优点 | 缺点 |
|------|------|------|
| RNN | 内存效率高 | 训练慢，梯度消失 |
| CNN | 并行计算 | 难以处理长序列 |
| Transformer | 并行+长距离依赖 | 内存需求大 |

## 🤝 贡献指南

欢迎提交改进建议！可以关注以下方向：

- 🎯 模型性能优化
- 🌐 支持更多语言对
- 📱 Web界面开发
- 🚀 部署工具改进
- 📊 评估指标完善

## 📄 许可证

本项目基于 MIT 许可证开源。

## 🙏 致谢

- [D2L.ai](https://d2l.ai/) 深度学习教程
- [PyTorch](https://pytorch.org/) 深度学习框架
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 原论文

---

**🎉 现在开始你的翻译之旅吧！**

```bash
# 一键开始
python transformer-d2l.py && python simple_translator.py
```
