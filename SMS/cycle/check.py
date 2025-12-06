import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import os

# 导入你的组件
from generator import ResnetGenerator
from diffusers import StableDiffusionPipeline, DDIMScheduler

# ================= 配置 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./cycle/sanity_check"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 既然只有一张图，我们手动加载它，不依赖 DataLoader
IMG_PATH = "/home/bjia-25/workspace/papers/gen/SMS/data/beach.jpg" # ⚠️ 请替换为你手里随便一张存在的图路径
STYLE_PROMPT = "oil painting style"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "./lora_ckpt/fechin.safetensors" # ⚠️ 替换为你的 LoRA 路径

# ================= 准备工作 =================
print("1. 初始化模型...")
net_G = ResnetGenerator().to(DEVICE)
optimizer = torch.optim.AdamW(net_G.parameters(), lr=1e-4) # 用标准学习率

# 加载原图
raw_img = Image.open(IMG_PATH).convert("RGB")
preprocess = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
input_tensor = preprocess(raw_img).unsqueeze(0).to(DEVICE) # [1, 3, 512, 512]

# 加载 SD 用于 SDS (暂时不加 Reward Model，减少变量)
print("2. 加载 SD Teacher...")
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
if os.path.exists(LORA_PATH):
    pipe.load_lora_weights(LORA_PATH)
pipe.vae.requires_grad_(False)
pipe.unet.requires_grad_(False)
scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

# 预计算 Text Embeddings
with torch.no_grad():
    text_input = pipe.tokenizer([STYLE_PROMPT, ""], padding="max_length", truncation=True, return_tensors="pt")
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(DEVICE))[0]

# ================= 测试阶段 1: 纯重建 (Reconstruction) =================
print("\n=== 测试阶段 1: 能否学会复制原图? (验证 Generator 完好) ===")
# 目标：生成器应该能迅速学会输出和原图一模一样的图
# 如果这一步都全是噪点，说明 Generator 架构坏了

for step in range(50):
    optimizer.zero_grad()
    fake_img = net_G(input_tensor)
    
    # 简单的像素损失
    loss = F.l1_loss(fake_img, input_tensor)
    
    loss.backward()
    optimizer.step()
    
    if step % 10 == 0:
        print(f"Step {step} | L1 Loss: {loss.item():.4f}")

# 保存阶段1结果
with torch.no_grad():
    save_image((net_G(input_tensor) + 1)/2, f"{OUTPUT_DIR}/check_1_reconstruction.png")
print("✅ 阶段 1 完成。请检查 check_1_reconstruction.png。如果是原图，说明生成器没问题。")


# ================= 测试阶段 2: SDS 风格化 (Style) =================
print("\n=== 测试阶段 2: 加入 SDS Loss (验证梯度方向) ===")
# 目标：图片应该开始变形、变色，向风格靠拢。
# 如果图片变成紫色噪点，说明 SDS 梯度计算错误。

# 降低一点 LR 防止震荡
for g in optimizer.param_groups: g['lr'] = 1e-5

for step in range(50):
    optimizer.zero_grad()
    fake_img = net_G(input_tensor)
    
    # --- SDS 计算 (手动展开，方便调试) ---
    # 1. Encode
    latents = pipe.vae.encode(fake_img.to(dtype=torch.float16)).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor # 关键！必须乘
    
    # 2. Add Noise
    noise = torch.randn_like(latents)
    t = torch.randint(20, 980, (1,), device=DEVICE).long()
    noisy_latents = scheduler.add_noise(latents, noise, t)
    
    # 3. Predict
    latent_input = torch.cat([noisy_latents] * 2)
    with torch.no_grad():
        noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
    
    pred_text, pred_uncond = noise_pred.chunk(2)[1], noise_pred.chunk(2)[0]
    noise_pred = pred_uncond + 7.5 * (pred_text - pred_uncond)
    
    # 4. Gradient (关键点)
    # w(t) 省略，设为 1
    # 理论公式：Grad = (noise_pred - noise)
    # 我们希望 latents 移动，使得预测的 noise 接近真实的 noise
    grad = (noise_pred - noise) 
    
    # *尝试翻转符号*：如果你之前的图是噪点，试试把下面这行取消注释
    # grad = -grad 
    
    # Backprop SDS
    # 我们要最小化 Loss，或者说我们要让 latents 沿着 -grad 方向走
    # 在 PyTorch backward(gradient) 中，gradient 代表 dL/dx
    # 如果我们认为 dL/dx = (noise_pred - noise)，那么 x 会向 -(noise_pred - noise) 更新
    # 这就是去噪方向，是正确的。
    
    latents.backward(gradient=grad, retain_graph=True)
    
    # 梯度裁剪 (防炸)
    torch.nn.utils.clip_grad_norm_(net_G.parameters(), 1.0)
    
    optimizer.step()
    
    if step % 10 == 0:
        print(f"Step {step} | SDS Update Done")

# 保存阶段2结果
with torch.no_grad():
    save_image((net_G(input_tensor) + 1)/2, f"{OUTPUT_DIR}/check_2_sds_style.png")
print("✅ 阶段 2 完成。请检查 check_2_sds_style.png。")
print("   - 如果是风格画：成功！")
print("   - 如果是紫色/彩色噪点：SDS 梯度爆炸或符号错误。")