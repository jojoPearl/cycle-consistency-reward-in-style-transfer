import torch
import os
import sys
import numpy as np

# 你的权重文件路径
WEIGHT_PATH = "/home/bjia-25/workspace/papers/gen/SMS/feedforward_results/feedforward_final.pth"

def check_tensor_health(name, tensor):
    """诊断单个张量的健康状况"""
    tensor = tensor.float() # 转为float32计算统计量
    
    # 1. 检查 NaN (Not a Number)
    if torch.isnan(tensor).any():
        return "❌ 包含 NaN (损坏)"
    
    # 2. 检查 Inf (无穷大)
    if torch.isinf(tensor).any():
        return "❌ 包含 Inf (数值溢出)"
    
    # 3. 检查数值范围
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    mean_val = tensor.mean().item()
    std_val = tensor.std().item()
    
    # 阈值判断
    if abs(max_val) > 100 or abs(min_val) > 100:
        return f"⚠️ 数值过大 (Min:{min_val:.1f}, Max:{max_val:.1f})"
    
    if std_val == 0:
        return "⚠️ 标准差为0 (死神经元/纯色)"
        
    return f"✅ 正常 (Mean:{mean_val:.3f}, Std:{std_val:.3f})"

def main():
    print(f"🔍 正在进行深度数值诊断: {WEIGHT_PATH}")
    
    if not os.path.exists(WEIGHT_PATH):
        print("文件不存在")
        return

    try:
        checkpoint = torch.load(WEIGHT_PATH, map_location="cpu")
        
        # 提取 state_dict
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint: state_dict = checkpoint['model']
            else: state_dict = checkpoint
            
        print(f"📊 总层数: {len(state_dict)}")
        print("-" * 60)
        print(f"{'层名称':<40} | {'健康状况'}")
        print("-" * 60)
        
        nan_count = 0
        explosion_count = 0
        
        for key, val in state_dict.items():
            # 只检查权重(weight)和偏置(bias)，跳过统计量(running_mean/var)
            if 'num_batches_tracked' in key: continue
            
            status = check_tensor_health(key, val)
            print(f"{key:<40} | {status}")
            
            if "NaN" in status: nan_count += 1
            if "数值过大" in status: explosion_count += 1

        print("-" * 60)
        print("诊断总结:")
        if nan_count > 0:
            print(f"🔴 严重故障: 发现 {nan_count} 层包含 NaN。模型已彻底损坏。")
            print("   -> 原因: 学习率过高 (LR=1e-4) 或 SDS 梯度计算出现除以零。")
        elif explosion_count > 0:
            print(f"🟠 潜在风险: 发现 {explosion_count} 层数值爆炸 (>100)。")
            print("   -> 原因: 梯度累积导致权重发散，未做 Gradient Clipping。")
        else:
            print("🟢 数值正常: 所有权重都在合理范围内。")
            print("   -> 如果图片还是乱的，可能是输入预处理 (Normalize) 没对齐。")

    except Exception as e:
        print(f"读取失败: {e}")

if __name__ == "__main__":
    main()