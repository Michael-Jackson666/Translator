"""
简洁版垃圾邮件分类器 - 只包含核心分类功能
用法: python simple_spam_classifier.py "your text here"
"""

import tiktoken
import torch
import sys
import os
from previous_chapters import GPTModel

def load_spam_classifier(model_path=None):
    """加载垃圾邮件分类器"""
    # 自动查找模型
    if model_path is None:
        possible_paths = [
            "spam_classifier_full_finetune.pth",
            "spam_classifier_partial_finetune.pth", 
            "review_classifier.pth"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError("未找到模型文件")
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    
    # 配置
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        config = {
            "vocab_size": 50257, "context_length": 1024, "drop_rate": 0.0,
            "qkv_bias": True, "emb_dim": 768, "n_layers": 12, "n_heads": 12
        }
    
    # 创建模型
    model = GPTModel(config)
    model.out_head = torch.nn.Linear(config["emb_dim"], 2)
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, device

def classify_spam(text, model=None, device=None):
    """
    分类文本是否为垃圾邮件
    
    Args:
        text: 要分类的文本
        model: 模型（可选，自动加载）
        device: 设备（可选，自动选择）
    
    Returns:
        "spam" 或 "not spam"
    """
    # 如果没有提供模型，则加载
    if model is None:
        model, device = load_spam_classifier()
    
    # 编码文本
    tokenizer = tiktoken.get_encoding("gpt2")
    input_ids = tokenizer.encode(text)
    
    # 处理长度
    max_length = 120
    input_ids = input_ids[:max_length]
    input_ids += [50256] * (max_length - len(input_ids))  # padding
    
    # 转换为tensor
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    
    # 预测
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
        predicted_label = torch.argmax(logits, dim=-1).item()
    
    return "spam" if predicted_label == 1 else "not spam"

def main():
    """命令行接口"""
    if len(sys.argv) < 2:
        print("用法: python simple_spam_classifier.py '你的文本'")
        print("\n示例:")
        print("python simple_spam_classifier.py 'You won $1000! Call now!'")
        print("python simple_spam_classifier.py 'Hi, how are you today?'")
        return
    
    text = " ".join(sys.argv[1:])
    
    try:
        result = classify_spam(text)
        icon = "🚫" if result == "spam" else "✅"
        print(f"{icon} {result}: {text}")
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()
