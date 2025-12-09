import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import glob
import sys
try:
    from generator import ResnetGenerator
except ImportError:
    print("❌ 错误: 找不到 generator.py，请确保它在当前目录下。")
    sys.exit(1)
MODEL_PATH = "/home/bjia-25/workspace/papers/gen/SMS/robust_results/robust_final.pth"
INPUT_DIR = "/home/bjia-25/workspace/papers/gen/SMS/data/content/class" 
OUTPUT_DIR = "./cycle/final_results"
DEVICE = torch.device('cpu') 
# =======================================
def main():
    print(f"🚀 开始推理... 设备: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("📦 加载模型中...")
    net = ResnetGenerator().to(DEVICE)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if 'state_dict' in checkpoint: checkpoint = checkpoint['state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        net.load_state_dict(new_state_dict)
        net.eval()
        print("✅ 权重加载成功！")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        sys.exit(1)
    preprocess = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(), # [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # -> [-1, 1]
    ])
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    print(f"📂 找到 {len(image_paths)} 张图片")
    for i, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        try:
            img = Image.open(img_path).convert("RGB")
            input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output_tensor = net(input_tensor)
            output_tensor = (output_tensor + 1) / 2.0
            output_tensor = torch.clamp(output_tensor, 0, 1)
            save_path = os.path.join(OUTPUT_DIR, f"stylized_{img_name}")
            save_image(output_tensor, save_path)
            print(f"[{i+1}/{len(image_paths)}] ✅ 已保存: {save_path}")
        except Exception as e:
            print(f"❌ 处理 {img_name} 失败: {e}")
    print("\n🎉 全部完成！请查看文件夹:", OUTPUT_DIR)
if __name__ == "__main__":
    main()