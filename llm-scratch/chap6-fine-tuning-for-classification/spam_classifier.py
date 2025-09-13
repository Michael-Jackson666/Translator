import tiktoken 
import torch
import os
import json
import tensorflow as tf
from gpt_download import load_gpt2_params_from_tf_ckpt
from previous_chapters import GPTModel, load_weights_into_gpt

tokenizer = tiktoken.get_encoding("gpt2")


# 模型配置
BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
}

CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])


# 直接从已下载的模型加载权重
def load_from_existing_files(model_size, models_dir="gpt2"):
    """直接从已下载的模型文件加载权重"""
    model_dir = os.path.join(models_dir, model_size)
    
    # 加载设置文件
    settings_path = os.path.join(model_dir, "hparams.json")
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)
    
    # 获取checkpoint路径并加载参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    print(f"📂 从已下载文件加载模型: {tf_ckpt_path}")
    
    # 使用gpt_download.py中的函数
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    
    return settings, params

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = load_from_existing_files(model_size=model_size, models_dir="gpt2")

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

# 添加分类头
torch.manual_seed(123)
num_classes = 2  # spam or not spam
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载已训练的模型
print("加载预训练的垃圾邮件分类器...")
try:
    checkpoint = torch.load("spam_classifier_full_finetune.pth", map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 成功加载全参数微调模型 (准确率: {checkpoint.get('test_accuracy', 'N/A'):.2%})")
    else:
        model.load_state_dict(checkpoint)
        print("✅ 成功加载全参数微调模型")
except FileNotFoundError:
    try:
        checkpoint = torch.load("spam_classifier_partial_finetune.pth", map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 成功加载部分微调模型 (准确率: {checkpoint.get('test_accuracy', 'N/A'):.2%})")
        else:
            model.load_state_dict(checkpoint)
            print("✅ 成功加载部分微调模型")
    except FileNotFoundError:
        print("⚠️ 未找到预训练模型，使用基础GPT-2模型")

# 垃圾邮件分类函数
def classify_review(text, model, tokenizer, device, max_length=120, pad_token_id=50256):
    model.eval()

    # 准备输入
    input_ids = tokenizer.encode(text)
    
    # 截断序列（确保不超过最大长度）
    input_ids = input_ids[:max_length]
    
    # 填充序列
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    # 模型推理
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"

# 测试样例
text_1 = "You are a winner you have been specially selected to receive $1000 cash or a $2000 award."
print("🚫 垃圾邮件:", classify_review(text_1, model, tokenizer, device))

text_2 = "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"
print("✅ 正常邮件:", classify_review(text_2, model, tokenizer, device))