import torch
import os
import sys

# 1. 导入你当前的网路定义
try:
    from generator import ResnetGenerator
except ImportError:
    print("❌ 错误: 找不到 generator.py，请确保它在当前目录下。")
    sys.exit(1)

# ================= 配置 =================
# 你的权重文件路径
WEIGHT_PATH = "/home/bjia-25/workspace/papers/gen/SMS/feedforward_results/feedforward_final.pth"
# =======================================

def main():
    print(f"🔍 正在检查权重对齐情况...")
    print(f"📁 权重文件: {WEIGHT_PATH}")

    if not os.path.exists(WEIGHT_PATH):
        print("❌ 文件不存在！")
        return

    # --- 1. 实例化当前代码定义的模型 ---
    print("\n[1] 实例化当前代码模型 (Current Model)...")
    try:
        model = ResnetGenerator()
        current_keys = set(model.state_dict().keys())
        print(f"✅ 当前模型实例化成功，包含 {len(current_keys)} 个参数层。")
    except Exception as e:
        print(f"❌ 当前模型定义有误，无法实例化: {e}")
        return

    # --- 2. 加载权重文件 ---
    print("\n[2] 加载权重文件 (Checkpoint)...")
    try:
        checkpoint = torch.load(WEIGHT_PATH, map_location="cpu")
        
        # 兼容性处理：提取真正的 state_dict
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                loaded_state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                loaded_state_dict = checkpoint['model']
            else:
                # 假设整个 dict 就是权重
                loaded_state_dict = checkpoint
        else:
            print("❌ 权重文件格式无法识别。")
            return

        # 去除 'module.' 前缀 (DDP 训练遗留)
        loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
        loaded_keys = set(loaded_state_dict.keys())
        print(f"✅ 权重文件加载成功，包含 {len(loaded_keys)} 个参数层。")

    except Exception as e:
        print(f"❌ 权重文件损坏或无法加载: {e}")
        return

    # --- 3. 对比分析 ---
    print("\n[3] 开始对比分析...")
    
    # A. 检查缺失的键 (Missing Keys) - 模型有，但文件里没有
    missing_keys = current_keys - loaded_keys
    # B. 检查多余的键 (Unexpected Keys) - 文件里有，但模型不需要
    unexpected_keys = loaded_keys - current_keys
    # C. 检查形状不匹配 (Shape Mismatch)
    shape_mismatch = []
    
    common_keys = current_keys.intersection(loaded_keys)
    for k in common_keys:
        model_shape = model.state_dict()[k].shape
        file_shape = loaded_state_dict[k].shape
        if model_shape != file_shape:
            shape_mismatch.append(f"{k}: 模型{model_shape} vs 文件{file_shape}")

    # --- 4. 输出诊断报告 ---
    print("-" * 40)
    print("📊 诊断报告:")
    
    if len(missing_keys) == 0 and len(unexpected_keys) == 0 and len(shape_mismatch) == 0:
        print("🟢 完美匹配！(Perfect Match)")
        print("   -> 现在的代码架构与训练时的权重完全一致。")
        print("   -> 如果推理结果还是乱码，那就是权重数值本身坏了(NaN/爆炸)，而不是架构问题。")
    else:
        print("🔴 发现不匹配！(Mismatch Detected)")
        
        if len(missing_keys) > 0:
            print(f"\n⚠️  缺失的层 (模型需要但文件没有): {len(missing_keys)} 个")
            for k in list(missing_keys)[:5]: print(f"   - {k}")
            if len(missing_keys) > 5: print("   ...")

        if len(unexpected_keys) > 0:
            print(f"\n⚠️  多余的层 (文件有但模型不需要): {len(unexpected_keys)} 个")
            for k in list(unexpected_keys)[:5]: print(f"   - {k}")
            if len(unexpected_keys) > 5: print("   ...")
            
        if len(shape_mismatch) > 0:
            print(f"\n⛔ 形状不匹配 (最严重的问题): {len(shape_mismatch)} 个")
            for msg in shape_mismatch: print(f"   - {msg}")

    print("-" * 40)

if __name__ == "__main__":
    main()