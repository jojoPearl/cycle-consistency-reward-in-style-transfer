import torch
import os
import sys

# 你的权重路径
WEIGHT_PATH = "/home/bjia-25/workspace/papers/gen/SMS/cycle/feedforward_results/feedforward_final.pth"

def check_weights():
    if not os.path.exists(WEIGHT_PATH):
        print(f"❌ 找不到文件: {WEIGHT_PATH}")
        return

    print(f"正在检查: {WEIGHT_PATH}")
    try:
        # 加载到 CPU
        state_dict = torch.load(WEIGHT_PATH, map_location="cpu")
        
        # 处理可能的嵌套
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        elif 'model' in state_dict: state_dict = state_dict['model']
        
        has_nan = False
        max_val = 0.0
        min_val = 0.0
        
        for key, val in state_dict.items():
            # 检查 NaN
            if torch.isnan(val).any():
                print(f"⚠️ 发现 NaN (无效数值) 在层: {key}")
                has_nan = True
            
            # 检查数值是否过大
            current_max = val.max().item()
            current_min = val.min().item()
            if abs(current_max) > max_val: max_val = abs(current_max)
            if abs(current_min) > min_val: min_val = abs(current_min)

        print("-" * 30)
        if has_nan:
            print("🔴 结论: 模型权重已损坏 (包含 NaN)。训练发散了。")
            print("   -> 建议: 降低学习率，加入梯度裁剪，重新训练。")
        elif max_val > 100:
            print(f"🟠 警告: 权重数值异常大 (Max: {max_val:.2f})。")
            print("   -> 这可能导致输出过饱和。建议检查归一化层。")
        else:
            print("🟢 结论: 权重数值范围看起来正常。")
            print("   -> 如果输出依然是噪声，说明是输入预处理或架构不匹配的问题。")

    except Exception as e:
        print(f"❌ 无法加载权重: {e}")

if __name__ == "__main__":
    check_weights()