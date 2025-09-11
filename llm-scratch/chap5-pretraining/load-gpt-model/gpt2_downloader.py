#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-2 模型下载器
调用 gpt_download.py 中的函数来下载不同大小的 GPT-2 模型
"""

import os
import sys
import time
from gpt_download import download_and_load_gpt2


class GPT2Downloader:
    """GPT-2 模型下载器类"""
    
    def __init__(self):
        self.models_dir = "gpt2"
        self.supported_models = {
            "124M": {
                "name": "GPT-2 Small",
                "params": "117M parameters",
                "layers": 12,
                "hidden_size": 768,
                "heads": 12,
                "size_estimate": "~500MB"
            },
            "355M": {
                "name": "GPT-2 Medium", 
                "params": "354M parameters",
                "layers": 24,
                "hidden_size": 1024,
                "heads": 16,
                "size_estimate": "~1.4GB"
            },
            "774M": {
                "name": "GPT-2 Large",
                "params": "774M parameters", 
                "layers": 36,
                "hidden_size": 1280,
                "heads": 20,
                "size_estimate": "~3.1GB"
            },
            "1558M": {
                "name": "GPT-2 XL",
                "params": "1.5B parameters",
                "layers": 48, 
                "hidden_size": 1600,
                "heads": 25,
                "size_estimate": "~6.2GB"
            }
        }
    
    def show_welcome(self):
        """显示欢迎信息"""
        print("=" * 60)
        print("🤖 GPT-2 模型下载器")
        print("=" * 60)
        print("📥 支持下载所有 OpenAI GPT-2 模型大小")
        print("🔄 自动断点续传，支持批量下载")
        print("=" * 60)
    
    def show_available_models(self):
        """显示可用的模型"""
        print("\n📋 可用的 GPT-2 模型:")
        print("-" * 70)
        print(f"{'大小':<8} {'名称':<15} {'参数':<18} {'层数':<6} {'大小估计':<12}")
        print("-" * 70)
        
        for model_id, info in self.supported_models.items():
            print(f"{model_id:<8} {info['name']:<15} {info['params']:<18} "
                  f"{info['layers']:<6} {info['size_estimate']:<12}")
        
        print("-" * 70)
        total_size = sum([float(info['size_estimate'].replace('~', '').replace('GB', '').replace('MB', '')) 
                         for info in self.supported_models.values()])
        print(f"💾 所有模型总大小约: ~11.0GB")
    
    def check_model_exists(self, model_size):
        """检查模型是否已经存在"""
        model_path = os.path.join(self.models_dir, model_size)
        if not os.path.exists(model_path):
            return False
        
        # 检查关键文件是否存在
        required_files = [
            "model.ckpt.data-00000-of-00001",
            "model.ckpt.index", 
            "model.ckpt.meta",
            "hparams.json",
            "encoder.json",
            "vocab.bpe"
        ]
        
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                return False
        
        return True
    
    def get_model_disk_size(self, model_size):
        """获取模型在磁盘上的实际大小（MB）"""
        model_path = os.path.join(self.models_dir, model_size)
        if not os.path.exists(model_path):
            return 0
        
        total_size = 0
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        
        return total_size / (1024 * 1024)  # 转换为MB
    
    def download_single_model(self, model_size, force_redownload=False):
        """下载单个模型"""
        if model_size not in self.supported_models:
            print(f"❌ 不支持的模型大小: {model_size}")
            print(f"✅ 支持的大小: {', '.join(self.supported_models.keys())}")
            return False
        
        model_info = self.supported_models[model_size]
        
        # 检查是否已存在
        if not force_redownload and self.check_model_exists(model_size):
            actual_size = self.get_model_disk_size(model_size)
            print(f"✅ {model_info['name']} ({model_size}) 已存在")
            print(f"📁 磁盘大小: {actual_size:.1f} MB")
            
            while True:
                choice = input("是否重新下载? (y/n): ").lower().strip()
                if choice in ['n', 'no']:
                    return True
                elif choice in ['y', 'yes']:
                    break
                else:
                    print("请输入 y 或 n")
        
        print(f"\n🚀 开始下载 {model_info['name']} ({model_size})")
        print(f"📊 {model_info['params']} | {model_info['layers']} 层 | 预计大小: {model_info['size_estimate']}")
        print("-" * 50)
        
        try:
            start_time = time.time()
            
            # 调用 gpt_download.py 中的函数
            settings, params = download_and_load_gpt2(
                model_size=model_size,
                models_dir=self.models_dir
            )
            
            end_time = time.time()
            download_time = end_time - start_time
            
            if settings is not None and params is not None:
                actual_size = self.get_model_disk_size(model_size)
                
                print(f"\n✅ {model_info['name']} 下载完成!")
                print(f"⏱️  下载时间: {download_time:.1f} 秒")
                print(f"💾 实际大小: {actual_size:.1f} MB")
                
                # 验证模型配置
                print(f"\n📋 模型配置验证:")
                expected = model_info
                print(f"   层数: {settings.get('n_layer', 'N/A')} (预期: {expected['layers']})")
                print(f"   隐藏层: {settings.get('n_embd', 'N/A')} (预期: {expected['hidden_size']})")
                print(f"   注意力头: {settings.get('n_head', 'N/A')} (预期: {expected['heads']})")
                print(f"   词汇表: {settings.get('n_vocab', 'N/A')}")
                print(f"   上下文: {settings.get('n_ctx', 'N/A')}")
                
                if 'wte' in params:
                    print(f"   Token嵌入: {params['wte'].shape}")
                
                return True
            else:
                print(f"❌ {model_info['name']} 下载失败")
                return False
                
        except KeyboardInterrupt:
            print(f"\n⚠️ 用户取消下载")
            return False
        except Exception as e:
            print(f"❌ 下载错误: {str(e)}")
            return False
    
    def download_multiple_models(self, model_list):
        """批量下载多个模型"""
        print(f"\n📦 批量下载 {len(model_list)} 个模型")
        
        # 验证所有模型
        invalid_models = [m for m in model_list if m not in self.supported_models]
        if invalid_models:
            print(f"❌ 无效的模型: {invalid_models}")
            return
        
        # 计算总估计大小
        total_estimate = 0
        for model in model_list:
            size_str = self.supported_models[model]['size_estimate']
            if 'GB' in size_str:
                total_estimate += float(size_str.replace('~', '').replace('GB', '')) * 1024
            else:
                total_estimate += float(size_str.replace('~', '').replace('MB', ''))
        
        print(f"💾 预计总大小: ~{total_estimate/1024:.1f} GB")
        
        # 确认下载
        while True:
            choice = input("确认开始批量下载? (y/n): ").lower().strip()
            if choice in ['n', 'no']:
                print("❌ 取消下载")
                return
            elif choice in ['y', 'yes']:
                break
            else:
                print("请输入 y 或 n")
        
        # 开始下载
        successful = []
        failed = []
        
        for i, model_size in enumerate(model_list, 1):
            print(f"\n{'='*20} 进度 {i}/{len(model_list)} {'='*20}")
            
            if self.download_single_model(model_size):
                successful.append(model_size)
            else:
                failed.append(model_size)
        
        # 显示结果
        print(f"\n{'='*50}")
        print("📊 批量下载完成!")
        print(f"✅ 成功: {len(successful)} 个 {successful}")
        if failed:
            print(f"❌ 失败: {len(failed)} 个 {failed}")
        print(f"{'='*50}")
    
    def interactive_mode(self):
        """交互式模式"""
        while True:
            print(f"\n{'='*50}")
            print("📋 请选择操作:")
            print("1. 下载单个模型")
            print("2. 批量下载模型") 
            print("3. 查看已下载模型")
            print("4. 下载所有模型")
            print("0. 退出")
            print("-" * 50)
            
            try:
                choice = input("请输入选择 (0-4): ").strip()
                
                if choice == '0':
                    print("👋 再见!")
                    break
                elif choice == '1':
                    self._interactive_single_download()
                elif choice == '2':
                    self._interactive_batch_download()
                elif choice == '3':
                    self._show_downloaded_models()
                elif choice == '4':
                    self._download_all_models()
                else:
                    print("❌ 无效选择，请重试")
                    
            except KeyboardInterrupt:
                print(f"\n👋 再见!")
                break
    
    def _interactive_single_download(self):
        """交互式单个下载"""
        self.show_available_models()
        
        while True:
            model_size = input(f"\n请输入要下载的模型大小 (或 'back' 返回): ").strip()
            
            if model_size.lower() == 'back':
                return
            elif model_size in self.supported_models:
                self.download_single_model(model_size)
                break
            else:
                print(f"❌ 无效选择，支持: {', '.join(self.supported_models.keys())}")
    
    def _interactive_batch_download(self):
        """交互式批量下载"""
        self.show_available_models()
        
        print(f"\n📥 批量下载模式")
        print("💡 输入多个模型大小，用空格或逗号分隔")
        print("💡 例如: 124M 355M 或 124M,355M,774M")
        
        while True:
            input_str = input(f"\n请输入模型列表 (或 'back' 返回): ").strip()
            
            if input_str.lower() == 'back':
                return
            
            # 解析输入
            if ',' in input_str:
                model_list = [m.strip() for m in input_str.split(',') if m.strip()]
            else:
                model_list = input_str.split()
            
            if not model_list:
                print("❌ 请输入至少一个模型大小")
                continue
            
            # 验证输入
            invalid = [m for m in model_list if m not in self.supported_models]
            if invalid:
                print(f"❌ 无效的模型: {invalid}")
                print(f"✅ 支持的模型: {', '.join(self.supported_models.keys())}")
                continue
            
            # 去重
            unique_models = list(dict.fromkeys(model_list))
            if len(unique_models) != len(model_list):
                print(f"⚠️ 已去除重复模型")
            
            self.download_multiple_models(unique_models)
            break
    
    def _show_downloaded_models(self):
        """显示已下载的模型"""
        print(f"\n📁 已下载的模型:")
        print("-" * 60)
        
        downloaded_count = 0
        total_size = 0
        
        for model_size, info in self.supported_models.items():
            if self.check_model_exists(model_size):
                size_mb = self.get_model_disk_size(model_size)
                total_size += size_mb
                downloaded_count += 1
                print(f"✅ {info['name']:<15} ({model_size}) - {size_mb:.1f} MB")
            else:
                print(f"❌ {info['name']:<15} ({model_size}) - 未下载")
        
        print("-" * 60)
        if downloaded_count > 0:
            print(f"📊 已下载: {downloaded_count}/{len(self.supported_models)} 个模型")
            print(f"💾 总大小: {total_size:.1f} MB ({total_size/1024:.1f} GB)")
        else:
            print("📭 没有已下载的模型")
    
    def _download_all_models(self):
        """下载所有模型"""
        print(f"\n📥 下载所有 GPT-2 模型")
        print(f"⚠️ 这将下载 {len(self.supported_models)} 个模型，总计约 11GB")
        print("⚠️ 需要较长时间和充足的磁盘空间!")
        
        while True:
            choice = input("确认下载所有模型? (yes/no): ").lower().strip()
            if choice in ['no', 'n']:
                print("❌ 取消下载")
                return
            elif choice == 'yes':
                break
            else:
                print("请输入 'yes' 或 'no'")
        
        all_models = list(self.supported_models.keys())
        self.download_multiple_models(all_models)


def main():
    """主函数"""
    downloader = GPT2Downloader()
    downloader.show_welcome()
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 命令行模式
        models_to_download = []
        for arg in sys.argv[1:]:
            if ',' in arg:
                models_to_download.extend([m.strip() for m in arg.split(',') if m.strip()])
            else:
                models_to_download.append(arg.strip())
        
        if 'all' in models_to_download:
            models_to_download = list(downloader.supported_models.keys())
        
        # 去重并验证
        models_to_download = list(dict.fromkeys(models_to_download))
        invalid_models = [m for m in models_to_download if m not in downloader.supported_models]
        
        if invalid_models:
            print(f"❌ 无效的模型: {invalid_models}")
            downloader.show_available_models()
            return
        
        if len(models_to_download) == 1:
            downloader.download_single_model(models_to_download[0])
        else:
            downloader.download_multiple_models(models_to_download)
    else:
        # 交互式模式
        downloader.show_available_models()
        downloader.interactive_mode()


if __name__ == "__main__":
    main()
