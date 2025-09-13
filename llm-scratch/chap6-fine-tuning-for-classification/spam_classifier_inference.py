"""
GPT-2 垃圾邮件分类器 - 推理/预测专用版本
只包含加载模型和分类功能，不包含训练代码
"""

import tiktoken
import torch
import json
import os
from previous_chapters import GPTModel

class SpamClassifier:
    def __init__(self, model_path=None, device=None):
        """
        初始化垃圾邮件分类器
        
        Args:
            model_path: 模型文件路径，默认使用当前目录下的模型
            device: 设备，默认自动选择
        """
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 如果没有指定模型路径，尝试找到可用的模型
        if model_path is None:
            model_path = self._find_model()
        
        self.model, self.config, self.max_length = self._load_model(model_path)
        print(f"✅ 垃圾邮件分类器已加载，使用设备: {self.device}")
    
    def _find_model(self):
        """自动查找可用的模型文件"""
        possible_paths = [
            "spam_classifier_full_finetune.pth",
            "spam_classifier_partial_finetune.pth", 
            "review_classifier.pth"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"🔍 找到模型文件: {path}")
                return path
        
        raise FileNotFoundError(
            f"未找到模型文件。请确保以下文件之一存在: {possible_paths}"
        )
    
    def _load_model(self, model_path):
        """加载训练好的模型"""
        print(f"📂 加载模型: {model_path}")
        
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 获取配置
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
        else:
            # 默认配置（兼容旧版本）
            config = {
                "vocab_size": 50257,
                "context_length": 1024,
                "drop_rate": 0.0,
                "qkv_bias": True,
                "emb_dim": 768,
                "n_layers": 12,
                "n_heads": 12
            }
        
        # 创建模型
        model = GPTModel(config)
        
        # 添加分类头
        model.out_head = torch.nn.Linear(
            in_features=config["emb_dim"],
            out_features=2  # spam or not spam
        )
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        # 设置最大长度
        max_length = 120  # 默认值，基于训练数据的典型长度
        
        # 显示模型信息
        if 'model_config' in checkpoint:
            print(f"📊 模型信息:")
            if 'test_accuracy' in checkpoint:
                print(f"   测试准确率: {checkpoint['test_accuracy']*100:.2f}%")
            if 'fine_tune_all' in checkpoint:
                strategy = "全参数微调" if checkpoint['fine_tune_all'] else "部分微调"
                print(f"   微调策略: {strategy}")
        
        return model, config, max_length
    
    def classify_text(self, text, return_confidence=False):
        """
        分类单个文本
        
        Args:
            text: 要分类的文本
            return_confidence: 是否返回置信度
            
        Returns:
            如果return_confidence=False: "spam" 或 "not spam"
            如果return_confidence=True: (分类结果, 置信度)
        """
        # 编码文本
        input_ids = self.tokenizer.encode(text)
        
        # 截断序列
        supported_context_length = self.model.pos_emb.weight.shape[0]
        max_len = min(self.max_length, supported_context_length)
        input_ids = input_ids[:max_len]
        
        # 填充序列
        pad_token_id = 50256
        input_ids += [pad_token_id] * (max_len - len(input_ids))
        input_tensor = torch.tensor(input_ids, device=self.device).unsqueeze(0)
        
        # 模型推理
        with torch.no_grad():
            logits = self.model(input_tensor)[:, -1, :]  # 最后一个token的logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_label = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_label].item()
        
        result = "spam" if predicted_label == 1 else "not spam"
        
        if return_confidence:
            return result, confidence
        else:
            return result
    
    def classify_batch(self, texts, return_confidence=False):
        """
        批量分类文本
        
        Args:
            texts: 文本列表
            return_confidence: 是否返回置信度
            
        Returns:
            分类结果列表
        """
        results = []
        for text in texts:
            result = self.classify_text(text, return_confidence)
            results.append(result)
        return results
    
    def classify_with_details(self, text):
        """
        详细分类结果，包含所有信息
        
        Args:
            text: 要分类的文本
            
        Returns:
            字典包含分类结果、置信度和概率
        """
        # 编码文本
        input_ids = self.tokenizer.encode(text)
        
        # 截断序列
        supported_context_length = self.model.pos_emb.weight.shape[0]
        max_len = min(self.max_length, supported_context_length)
        input_ids = input_ids[:max_len]
        
        # 填充序列
        pad_token_id = 50256
        input_ids += [pad_token_id] * (max_len - len(input_ids))
        input_tensor = torch.tensor(input_ids, device=self.device).unsqueeze(0)
        
        # 模型推理
        with torch.no_grad():
            logits = self.model(input_tensor)[:, -1, :]
            probabilities = torch.softmax(logits, dim=-1)
            predicted_label = torch.argmax(logits, dim=-1).item()
        
        result = "spam" if predicted_label == 1 else "not spam"
        confidence = probabilities[0][predicted_label].item()
        
        return {
            "text": text,
            "prediction": result,
            "confidence": confidence,
            "probabilities": {
                "not_spam": probabilities[0][0].item(),
                "spam": probabilities[0][1].item()
            },
            "text_length": len(text),
            "token_count": len(input_ids)
        }

def main():
    """示例用法"""
    print("🎯 GPT-2 垃圾邮件分类器 - 推理版本")
    print("="*50)
    
    # 初始化分类器
    try:
        classifier = SpamClassifier()
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print("💡 请先运行训练脚本生成模型文件")
        return
    
    # 测试文本
    test_texts = [
        "You are a winner you have been specially selected to receive $1000 cash or a $2000 award.",
        "Hey, just wanted to check if we're still on for dinner tonight? Let me know!",
        "URGENT! You have won £2000! Call now to claim your prize!",
        "Hi mom, can you pick me up from school at 3pm today?",
        "Free phone! Text WIN to 12345 to get your free smartphone now!",
        "The meeting has been rescheduled to tomorrow at 2pm. Please confirm your attendance."
    ]
    
    print("🔍 单个文本分类示例:")
    print("-" * 50)
    
    for i, text in enumerate(test_texts[:3], 1):
        result, confidence = classifier.classify_text(text, return_confidence=True)
        print(f"{i}. 文本: {text[:50]}...")
        print(f"   结果: {result} (置信度: {confidence:.3f})")
        print()
    
    print("📊 详细分类示例:")
    print("-" * 50)
    
    sample_text = test_texts[0]
    details = classifier.classify_with_details(sample_text)
    
    print(f"文本: {details['text'][:60]}...")
    print(f"预测: {details['prediction']}")
    print(f"置信度: {details['confidence']:.3f}")
    print(f"概率分布:")
    print(f"  正常邮件: {details['probabilities']['not_spam']:.3f}")
    print(f"  垃圾邮件: {details['probabilities']['spam']:.3f}")
    print(f"文本长度: {details['text_length']} 字符")
    print(f"Token数量: {details['token_count']}")
    
    print("\n🚀 批量分类示例:")
    print("-" * 50)
    
    batch_results = classifier.classify_batch(test_texts)
    for text, result in zip(test_texts, batch_results):
        status = "🚫" if result == "spam" else "✅"
        print(f"{status} {result}: {text[:60]}...")

if __name__ == "__main__":
    main()
