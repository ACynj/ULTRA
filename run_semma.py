#!/usr/bin/env python3
"""
SEMMA 运行脚本
一键运行SEMMA实验，包含完整的文本语义生成流程
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="运行SEMMA实验")
    parser.add_argument("--dataset", type=str, default="FB15k237", help="数据集名称")
    parser.add_argument("--epochs", type=int, default=0, help="训练轮数")
    parser.add_argument("--gpus", type=str, default="[0]", help="GPU设备")
    parser.add_argument("--ckpt", type=str, default="./ckpts/ultra_4g.pth", help="检查点路径")
    parser.add_argument("--config", type=str, default="config/transductive/semma_example.yaml", help="配置文件")
    parser.add_argument("--test-only", action="store_true", help="仅运行测试")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SEMMA 实验运行器")
    print("=" * 80)
    
    if args.test_only:
        print("🧪 运行测试模式...")
        cmd = [sys.executable, "test_semma.py"]
    else:
        print("🚀 运行SEMMA实验...")
        print(f"   数据集: {args.dataset}")
        print(f"   训练轮数: {args.epochs}")
        print(f"   GPU: {args.gpus}")
        print(f"   检查点: {args.ckpt}")
        print(f"   配置: {args.config}")
        
        cmd = [
            sys.executable, "script/run.py",
            "-c", args.config,
            "--dataset", args.dataset,
            "--epochs", str(args.epochs),
            "--gpus", args.gpus,
            "--ckpt", args.ckpt
        ]
    
    print(f"\n执行命令: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        # 运行命令
        result = subprocess.run(cmd, check=True, cwd=os.getcwd())
        
        print("\n" + "=" * 80)
        print("✅ 运行成功!")
        print("=" * 80)
        
        if not args.test_only:
            print("\n📊 实验结果:")
            print("   - 检查日志文件获取详细结果")
            print("   - 关系语义已缓存，下次运行更快")
            print("   - 文本关系图已构建完成")
        
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 80)
        print("❌ 运行失败!")
        print("=" * 80)
        print(f"错误代码: {e.returncode}")
        print("\n🔧 故障排除建议:")
        print("   1. 检查配置文件路径")
        print("   2. 确认数据集已下载")
        print("   3. 验证GPU可用性")
        print("   4. 检查API密钥和网络连接")
        print("   5. 运行测试模式: python run_semma.py --test-only")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断运行")
        
    except Exception as e:
        print(f"\n❌ 意外错误: {e}")


if __name__ == "__main__":
    main()
