"""
第一步：验证生成器基础能力
目标：只用L1 Loss训练生成器复制输入图像
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
import os
from PIL import Image
import matplotlib.pyplot as plt

# ========== 1. 导入你的生成器 ==========
# 确保路径正确
import sys
sys.path.append('.')
from generator import ResnetGenerator  # 或你的生成器类

# ========== 2. 创建简单测试数据集 ==========
class SimpleTestDataset(Dataset):
    """只用少量图像测试"""
    def __init__(self, image_dir, num_samples=10):
        self.image_paths = []
        for f in os.listdir(image_dir)[:num_samples]:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                self.image_paths.append(os.path.join(image_dir, f))
        
        self.transform = transforms.Compose([
            transforms.Resize(256),  # 先用小分辨率测试
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        return img, self.image_paths[idx]

# ========== 3. 基础测试函数 ==========
def test_generator_capability():
    """测试生成器能否学会恒等映射"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 3.1 初始化生成器
    net_G = ResnetGenerator().to(device)
    print(f"生成器参数量: {sum(p.numel() for p in net_G.parameters()):,}")
    
    # 3.2 创建测试数据
    test_dataset = SimpleTestDataset(
        image_dir="./data/content",  # 你的内容图像目录
        num_samples=5  # 只用5张图测试
    )
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 3.3 固定一张测试图像用于监控
    test_img, test_path = test_dataset[0]
    test_img = test_img.unsqueeze(0).to(device)
    print(f"测试图像: {test_path}")
    print(f"输入范围: [{test_img.min():.3f}, {test_img.max():.3f}]")
    
    # 3.4 检查生成器初始输出
    with torch.no_grad():
        initial_output = net_G(test_img)
        print(f"初始输出范围: [{initial_output.min():.3f}, {initial_output.max():.3f}]")
        
        # 保存初始状态
        save_comparison(
            test_img, 
            initial_output, 
            "step1_initial.png",
            title="初始化状态"
        )
    
    # 3.5 简单训练：只用L1 Loss
    optimizer = torch.optim.Adam(net_G.parameters(), lr=1e-4)  # 小学习率
    criterion = nn.L1Loss()
    
    print("\n开始基础训练（恒等映射）...")
    losses = []
    
    # 只用少量迭代测试
    num_iterations = 1000  # 约10-20分钟
    for iteration in range(num_iterations):
        # 随机取一张图
        img_batch, _ = next(iter(dataloader))
        img_batch = img_batch.to(device)
        
        # 前向传播
        output = net_G(img_batch)
        
        # 计算L1 Loss（输入=目标）
        loss = criterion(output, img_batch)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止爆炸）
        torch.nn.utils.clip_grad_norm_(net_G.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        
        # 每100次迭代输出进度
        if (iteration + 1) % 100 == 0:
            print(f"迭代 [{iteration+1}/{num_iterations}] - L1 Loss: {loss.item():.4f}")
            
            # 监控固定测试图像
            with torch.no_grad():
                test_output = net_G(test_img)
                test_loss = criterion(test_output, test_img)
                print(f"  测试图像Loss: {test_loss.item():.4f}")
                
                # 保存可视化
                if (iteration + 1) % 500 == 0:
                    save_comparison(
                        test_img, 
                        test_output, 
                        f"step1_iteration_{iteration+1}.png",
                        title=f"迭代 {iteration+1}"
                    )
    
    # 3.6 最终评估
    print("\n" + "="*50)
    print("最终评估:")
    
    with torch.no_grad():
        final_output = net_G(test_img)
        final_loss = criterion(final_output, test_img)
        
        # 计算PSNR（评估重建质量）
        mse = F.mse_loss(final_output, test_img)
        psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  # 因为范围是[-1,1]
        
        print(f"最终L1 Loss: {final_loss.item():.4f}")
        print(f"最终PSNR: {psnr.item():.2f} dB")
        
        # PSNR > 20dB 表示可以接受的重建
        if psnr > 20:
            print("✅ 生成器学会了恒等映射！")
            print("   可以进行下一步训练。")
        else:
            print("❌ 生成器未能学会恒等映射。")
            print("   需要调整生成器架构或学习率。")
        
        # 保存最终对比
        save_comparison(
            test_img, 
            final_output, 
            "step1_final.png",
            title=f"最终结果 (PSNR: {psnr.item():.2f}dB)"
        )
    
    # 3.7 保存损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('迭代')
    plt.ylabel('L1 Loss')
    plt.title('生成器基础训练损失曲线')
    plt.grid(True)
    plt.savefig('step1_loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3.8 保存训练好的生成器（用于下一步）
    torch.save(net_G.state_dict(), 'step1_trained_generator.pth')
    print("\n生成器已保存: step1_trained_generator.pth")

def save_comparison(input_img, output_img, filename, title=""):
    """保存输入和输出的对比图像"""
    # 反归一化到[0,1]
    input_norm = (input_img[0].cpu() + 1) / 2
    output_norm = (output_img[0].cpu() + 1) / 2
    
    # 拼接
    comparison = torch.cat([input_norm, output_norm], dim=2)
    
    # 保存
    save_image(comparison, filename)
    print(f"  已保存: {filename}")

# ========== 4. 额外诊断功能 ==========
def debug_generator_architecture():
    """检查生成器架构问题"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_G = ResnetGenerator().to(device)
    
    print("\n" + "="*50)
    print("生成器架构诊断:")
    print("="*50)
    
    # 4.1 检查每层输出范围
    test_input = torch.randn(1, 3, 256, 256).to(device)
    
    print("前向传播检查...")
    
    # 如果生成器有多个模块，可以逐层检查
    def check_layer_outputs(model, input_tensor, layer_names=None):
        outputs = []
        
        # 临时注册钩子
        hooks = []
        def hook_fn(module, input, output):
            outputs.append((module.__class__.__name__, output))
        
        # 注册钩子
        for name, module in model.named_modules():
            if not name:  # 跳过根模块
                continue
            if layer_names is None or name in layer_names:
                hooks.append(module.register_forward_hook(hook_fn))
        
        # 前向传播
        with torch.no_grad():
            _ = model(input_tensor)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 打印每层输出
        for i, (layer_name, output) in enumerate(outputs):
            if i % 3 == 0:  # 每3层打印一次
                print(f"  {layer_name}: [{output.min():.3f}, {output.max():.3f}]")
    
    check_layer_outputs(net_G, test_input)
    
    # 4.2 检查梯度流动
    print("\n梯度流动检查...")
    test_input.requires_grad = False
    output = net_G(test_input)
    target = torch.randn_like(output)
    loss = F.mse_loss(output, target)
    
    loss.backward()
    
    grad_norms = []
    for name, param in net_G.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if grad_norm < 1e-10:
                print(f"  ⚠️ {name}: 梯度消失 ({grad_norm:.2e})")
            elif grad_norm > 100:
                print(f"  ⚠️ {name}: 梯度爆炸 ({grad_norm:.2e})")
    
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
    print(f"  平均梯度范数: {avg_grad_norm:.2e}")
    
    return avg_grad_norm

# ========== 5. 主执行函数 ==========
def main():
    print("="*60)
    print("第一步：验证生成器基础能力")
    print("目标：只用L1 Loss训练生成器复制输入图像")
    print("="*60)
    
    # 5.1 先进行架构诊断
    print("\n[阶段1] 架构诊断...")
    grad_norm = debug_generator_architecture()
    
    # 如果梯度异常，先调整学习率
    if grad_norm > 100:
        print("检测到梯度爆炸风险，建议：")
        print("  1. 检查生成器初始化")
        print("  2. 使用更小的学习率（如1e-5）")
        print("  3. 添加BatchNorm层")
        choice = input("是否继续测试？(y/n): ")
        if choice.lower() != 'y':
            return
    
    # 5.2 运行基础训练测试
    print("\n[阶段2] 基础训练测试...")
    try:
        test_generator_capability()
    except Exception as e:
        print(f"训练过程中出错: {e}")
        print("\n尝试更简单的测试...")
        quick_fix_test()
    
    print("\n" + "="*60)
    print("第一步完成！")
    print("检查保存的图像文件:")
    print("  - step1_initial.png: 初始状态")
    print("  - step1_final.png: 最终结果")
    print("  - step1_loss_curve.png: 损失曲线")
    print("="*60)

def quick_fix_test():
    """如果主测试失败，运行更简单的测试"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建更简单的生成器
    class SimpleGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 3, 3, padding=1),
            )
        
        def forward(self, x):
            return torch.tanh(self.net(x))
    
    print("使用简化生成器测试...")
    net_G = SimpleGenerator().to(device)
    
    # 用随机数据测试
    test_input = torch.randn(1, 3, 256, 256).to(device)
    
    optimizer = torch.optim.Adam(net_G.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    
    for i in range(500):
        output = net_G(test_input)
        loss = criterion(output, test_input)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f"简化测试 [{i+1}/500] - Loss: {loss.item():.4f}")
    
    print("简化测试完成。如果这个能成功，说明你的ResnetGenerator可能需要调整。")

if __name__ == "__main__":
    main()