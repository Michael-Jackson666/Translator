"""
GPT-2 垃圾邮件分类器 - 全参数微调版本
支持完整的模型微调和高级保存功能
"""

import tiktoken 
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import tensorflow as tf
import os
import time
import pickle
import matplotlib.pyplot as plt

# 初始化tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        return max(len(encoded_text) for encoded_text in self.encoded_texts)

def load_from_existing_files(model_size, models_dir="gpt2"):
    """直接从已下载的模型文件加载权重"""
    from gpt_download import load_gpt2_params_from_tf_ckpt
    
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

def setup_model_and_data():
    """设置模型和数据"""
    print("🔧 初始化模型和数据...")
    
    # 模型配置
    CHOOSE_MODEL = "gpt2-small (124M)"
    
    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    
    # 创建数据集
    train_dataset = SpamDataset("train.csv", tokenizer, max_length=None)
    val_dataset = SpamDataset("validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
    test_dataset = SpamDataset("test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
    
    # 检查序列长度
    assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
        f"`max_length={BASE_CONFIG['context_length']}`"
    )
    
    # 创建数据加载器
    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=0,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=0,
        drop_last=False,
    )
    
    print(f"📊 数据集信息:")
    print(f"   训练集: {len(train_dataset)} 样本")
    print(f"   验证集: {len(val_dataset)} 样本") 
    print(f"   测试集: {len(test_dataset)} 样本")
    print(f"   最大序列长度: {train_dataset.max_length}")
    
    return BASE_CONFIG, CHOOSE_MODEL, train_loader, val_loader, test_loader

def create_model(BASE_CONFIG, CHOOSE_MODEL, fine_tune_all=True):
    """创建并配置模型"""
    from previous_chapters import GPTModel, load_weights_into_gpt
    
    print("🏗️ 创建模型...")
    
    # 加载预训练权重
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = load_from_existing_files(model_size=model_size, models_dir="gpt2")

    # 创建模型
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    # 配置微调策略
    if fine_tune_all:
        print("🔧 使用全参数微调策略：微调所有层的参数")
        for param in model.parameters():
            param.requires_grad = True
    else:
        print("🔧 使用部分微调策略：只微调最后一个transformer block和输出层")
        for param in model.parameters():
            param.requires_grad = False

    # 添加分类头
    torch.manual_seed(123)
    num_classes = 2  # spam or not spam
    model.out_head = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"],
        out_features=num_classes
    )

    # 确保分类头可训练
    for param in model.out_head.parameters():
        param.requires_grad = True
        
    # 如果是部分微调，确保最后一层transformer可训练
    if not fine_tune_all:
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 模型参数统计:")
    print(f"   总参数数量: {total_params:,}")
    print(f"   可训练参数数量: {trainable_params:,}")
    print(f"   可训练参数比例: {100 * trainable_params / total_params:.2f}%")
    
    return model, fine_tune_all

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """计算分类准确率"""
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
        
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
    """计算单个batch的损失"""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """计算数据加载器的平均损失"""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
        
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """评估模型"""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def train_classifier_advanced(model, train_loader, val_loader, optimizer, device, 
                             num_epochs, eval_freq, eval_iter):
    """高级训练函数"""
    print("🚀 开始训练...")
    
    # 初始化跟踪列表
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # 主训练循环
    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式
        epoch_start_time = time.time()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # 重置梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # 计算梯度
            optimizer.step() # 更新权重
            examples_seen += input_batch.shape[0]
            global_step += 1

            # 定期评估
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 计算每个epoch后的准确率
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s): "
              f"Train Acc: {train_accuracy*100:.2f}% | Val Acc: {val_accuracy*100:.2f}%")
        
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

def plot_training_results(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    """绘制训练结果"""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制训练和验证曲线
    ax1.plot(epochs_seen, train_values, label=f"Training {label}", linewidth=2)
    ax1.plot(epochs_seen, val_values, linestyle="--", label=f"Validation {label}", linewidth=2)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 创建第二个x轴显示样本数
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    plt.title(f"Training Progress - {label.capitalize()}")
    fig.tight_layout()
    plt.savefig(f"{label}-plot-advanced.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def save_model_advanced(model, config, metrics, fine_tune_all, num_epochs, 
                       training_history, execution_time):
    """高级模型保存功能"""
    print("\n💾 保存微调后的模型...")
    
    # 保存完整的模型状态
    model_save_path = f"spam_classifier_{'full' if fine_tune_all else 'partial'}_finetune.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': config,
        'train_accuracy': metrics['train_accuracy'],
        'val_accuracy': metrics['val_accuracy'], 
        'test_accuracy': metrics['test_accuracy'],
        'fine_tune_all': fine_tune_all,
        'num_epochs': num_epochs,
        'tokenizer_encoding': 'gpt2',
        'training_time_minutes': execution_time,
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }, model_save_path)

    print(f"✅ 模型已保存到: {model_save_path}")

    # 保存训练历史
    history_save_path = f"training_history_{'full' if fine_tune_all else 'partial'}.pkl"
    
    with open(history_save_path, 'wb') as f:
        pickle.dump(training_history, f)

    print(f"✅ 训练历史已保存到: {history_save_path}")
    
    return model_save_path, history_save_path

def load_fine_tuned_model(model_path, device='cpu'):
    """
    加载微调后的垃圾邮件分类模型
    
    Args:
        model_path: 模型文件路径
        device: 设备 ('cpu' 或 'cuda')
    
    Returns:
        model: 加载的模型
        config: 模型配置
        metrics: 性能指标
    """
    from previous_chapters import GPTModel
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 重建模型
    config = checkpoint['model_config']
    model = GPTModel(config)
    
    # 添加分类头
    model.out_head = torch.nn.Linear(
        in_features=config["emb_dim"],
        out_features=2  # spam or not spam
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 性能指标
    metrics = {
        'train_accuracy': checkpoint['train_accuracy'],
        'val_accuracy': checkpoint['val_accuracy'],
        'test_accuracy': checkpoint['test_accuracy'],
        'fine_tune_strategy': 'full' if checkpoint['fine_tune_all'] else 'partial',
        'training_time': checkpoint.get('training_time_minutes', 'N/A'),
        'total_params': checkpoint.get('total_params', 'N/A'),
        'trainable_params': checkpoint.get('trainable_params', 'N/A')
    }
    
    return model, config, metrics

def main():
    """主函数"""
    print("🎯 GPT-2 垃圾邮件分类器 - 全参数微调版本")
    print("="*60)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用设备: {device}")
    
    # 设置模型和数据
    BASE_CONFIG, CHOOSE_MODEL, train_loader, val_loader, test_loader = setup_model_and_data()
    
    # 创建模型 - 设置为全参数微调
    model, fine_tune_all = create_model(BASE_CONFIG, CHOOSE_MODEL, fine_tune_all=True)
    model.to(device)
    
    # 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    
    # 训练参数
    num_epochs = 3  # 减少epoch数因为全参数微调较慢
    eval_freq = 50
    eval_iter = 5
    
    # 开始训练
    start_time = time.time()
    torch.manual_seed(123)

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_advanced(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"\n✅ 训练完成! 用时: {execution_time_minutes:.2f} 分钟")

    # 绘制训练结果
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    
    plot_training_results(epochs_tensor, examples_seen_tensor, train_losses, val_losses, "loss")
    
    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_training_results(epochs_tensor, examples_seen_tensor, train_accs, val_accs, "accuracy")

    # 计算最终准确率
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device) 
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"\n📊 最终性能指标:")
    print(f"   训练准确率: {train_accuracy*100:.2f}%")
    print(f"   验证准确率: {val_accuracy*100:.2f}%")
    print(f"   测试准确率: {test_accuracy*100:.2f}%")

    # 保存模型和结果
    metrics = {
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy
    }
    
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'examples_seen': examples_seen,
        'execution_time_minutes': execution_time_minutes
    }
    
    model_path, history_path = save_model_advanced(
        model, BASE_CONFIG, metrics, fine_tune_all, 
        num_epochs, training_history, execution_time_minutes
    )
    
    print(f"\n📖 模型加载示例:")
    print(f"model, config, metrics = load_fine_tuned_model('{model_path}')")
    print(f"print(f'Test accuracy: {{metrics[\"test_accuracy\"]*100:.2f}}%')")
    
    print("\n🎉 微调完成!")

if __name__ == "__main__":
    main()
