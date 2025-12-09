"""
：
：L1 Loss
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
import sys
sys.path.append('.')
from generator import ResnetGenerator  
class SimpleTestDataset(Dataset):
    def __init__(self, image_dir, num_samples=10):
        self.image_paths = []
        for f in os.listdir(image_dir)[:num_samples]:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                self.image_paths.append(os.path.join(image_dir, f))
        self.transform = transforms.Compose([
            transforms.Resize(256),  
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
def test_generator_capability():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    net_G = ResnetGenerator().to(device)
    print(f"生成器参数量: {sum(p.numel() for p in net_G.parameters()):,}")
    test_dataset = SimpleTestDataset(
        image_dir="./data/content",  
        num_samples=5  
    )
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_img, test_path = test_dataset[0]
    test_img = test_img.unsqueeze(0).to(device)
    print(f"测试图像: {test_path}")
    print(f"输入范围: [{test_img.min():.3f}, {test_img.max():.3f}]")
    with torch.no_grad():
        initial_output = net_G(test_img)
        print(f"初始输出范围: [{initial_output.min():.3f}, {initial_output.max():.3f}]")
        save_comparison(
            test_img, 
            initial_output, 
            "step1_initial.png",
            title="初始化状态"
        )
    optimizer = torch.optim.Adam(net_G.parameters(), lr=1e-4)  
    criterion = nn.L1Loss()
    print("\n开始基础训练（恒等映射）...")
    losses = []
    num_iterations = 1000  
    for iteration in range(num_iterations):
        img_batch, _ = next(iter(dataloader))
        img_batch = img_batch.to(device)
        output = net_G(img_batch)
        loss = criterion(output, img_batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_G.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
        if (iteration + 1) % 100 == 0:
            print(f"迭代 [{iteration+1}/{num_iterations}] - L1 Loss: {loss.item():.4f}")
            with torch.no_grad():
                test_output = net_G(test_img)
                test_loss = criterion(test_output, test_img)
                print(f"  测试图像Loss: {test_loss.item():.4f}")
                if (iteration + 1) % 500 == 0:
                    save_comparison(
                        test_img, 
                        test_output, 
                        f"step1_iteration_{iteration+1}.png",
                        title=f"迭代 {iteration+1}"
                    )
    print("\n" + "="*50)
    print("最终评估:")
    with torch.no_grad():
        final_output = net_G(test_img)
        final_loss = criterion(final_output, test_img)
        mse = F.mse_loss(final_output, test_img)
        psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  
        print(f"最终L1 Loss: {final_loss.item():.4f}")
        print(f"最终PSNR: {psnr.item():.2f} dB")
        if psnr > 20:
            print("✅ 生成器学会了恒等映射！")
            print("   可以进行下一步训练。")
        else:
            print("❌ 生成器未能学会恒等映射。")
            print("   需要调整生成器架构或学习率。")
        save_comparison(
            test_img, 
            final_output, 
            "step1_final.png",
            title=f"最终结果 (PSNR: {psnr.item():.2f}dB)"
        )
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('迭代')
    plt.ylabel('L1 Loss')
    plt.title('生成器基础训练损失曲线')
    plt.grid(True)
    plt.savefig('step1_loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    torch.save(net_G.state_dict(), 'step1_trained_generator.pth')
    print("\n生成器已保存: step1_trained_generator.pth")
def save_comparison(input_img, output_img, filename, title=""):
    input_norm = (input_img[0].cpu() + 1) / 2
    output_norm = (output_img[0].cpu() + 1) / 2
    comparison = torch.cat([input_norm, output_norm], dim=2)
    save_image(comparison, filename)
    print(f"  已保存: {filename}")
def debug_generator_architecture():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_G = ResnetGenerator().to(device)
    print("\n" + "="*50)
    print("生成器架构诊断:")
    print("="*50)
    test_input = torch.randn(1, 3, 256, 256).to(device)
    print("前向传播检查...")
    def check_layer_outputs(model, input_tensor, layer_names=None):
        outputs = []
        hooks = []
        def hook_fn(module, input, output):
            outputs.append((module.__class__.__name__, output))
        for name, module in model.named_modules():
            if not name:  
                continue
            if layer_names is None or name in layer_names:
                hooks.append(module.register_forward_hook(hook_fn))
        with torch.no_grad():
            _ = model(input_tensor)
        for hook in hooks:
            hook.remove()
        for i, (layer_name, output) in enumerate(outputs):
            if i % 3 == 0:  
                print(f"  {layer_name}: [{output.min():.3f}, {output.max():.3f}]")
    check_layer_outputs(net_G, test_input)
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
def main():
    print("="*60)
    print("第一步：验证生成器基础能力")
    print("目标：只用L1 Loss训练生成器复制输入图像")
    print("="*60)
    print("\n[阶段1] 架构诊断...")
    grad_norm = debug_generator_architecture()
    if grad_norm > 100:
        print("检测到梯度爆炸风险，建议：")
        print("  1. 检查生成器初始化")
        print("  2. 使用更小的学习率（如1e-5）")
        print("  3. 添加BatchNorm层")
        choice = input("是否继续测试？(y/n): ")
        if choice.lower() != 'y':
            return
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
    """，"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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