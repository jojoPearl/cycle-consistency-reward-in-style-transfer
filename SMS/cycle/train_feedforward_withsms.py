import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image
import sys

# --- 1. 导入必要的外部库 ---
from generator import ResnetGenerator 
from train_reward_model import CycleRewardModel 
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPProcessor, CLIPModel 

# --- 2. 定义 SDS 核心逻辑 (与调试脚本保持一致) ---

def load_sms_teacher(model_id, lora_path, device):
    """加载 SMS 的老师模型 (SD + Style LoRA)"""
    print(f"Loading SMS Teacher: {model_id} + {lora_path}")
    
    # 强制加载 SD 1.5 Base
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    ).to(device)
    
    # 加载 LoRA 权重
    if lora_path and os.path.exists(lora_path):
        pipe.load_lora_weights(lora_path)
    else:
        print(f"⚠️ Warning: LoRA path {lora_path} not found! Using base model only.")
        
    # 冻结所有参数
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    
    return pipe

def calculate_sds_loss(
    pipe, fake_img, text_embeddings, scheduler, device, guidance_scale=7.5
):
    """
    计算 SMS (SDS) 损失。
    """
    # 1. 编码生成的图片 -> Latent Space
    # 必须乘 scaling factor (调试脚本验证过这是关键)
    latents = pipe.vae.encode(fake_img.to(dtype=torch.float16)).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor

    # 2. 随机采样时间步 t
    # 避免极端的 timestep
    bsz = latents.shape[0]
    t = torch.randint(20, 980, (bsz,), device=device).long()

    # 3. 加噪
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, t)

    # 4. 预测噪声 (Teacher Prediction)
    latent_model_input = torch.cat([noisy_latents] * 2)
    t_input = torch.cat([t] * 2)
    
    with torch.no_grad():
        noise_pred = pipe.unet(
            latent_model_input,
            t_input,
            encoder_hidden_states=text_embeddings,
        ).sample

    # CFG 引导
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # 5. 计算 SDS 梯度
    # 我们验证过，grad = (noise_pred - noise) 配合 latents.backward(gradient=grad) 是有效的
    grad = (noise_pred - noise)

    # 6. 反向传播这个梯度
    # 我们不返回标量 Loss，而是直接在这里反向传播给 latents
    # 这样 fake_img (以及 net_G) 就会收到梯度
    latents.backward(gradient=grad, retain_graph=True)
    
    # 返回一个 Loss 数值仅供打印查看 (不参与 backward)
    loss_value = F.mse_loss(noise_pred, noise).item()
    return loss_value

# --- 3. MAIN TRAINING LOOP ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", type=str, default="./data/content")
    parser.add_argument("--output_dir", type=str, default="./cycle/sms_feedforward_results")
    parser.add_argument("--reward_model_path", type=str, default="./reward_model_checkpoint/epoch_1.pth")
    parser.add_argument("--lora_path", type=str, default="lora_ckpt/fechin.safetensors")
    parser.add_argument("--style_prompt", type=str, default="oil painting style")
    parser.add_argument("--sd_path", type=str, default="runwayml/stable-diffusion-v1-5")
    
    # ⚠️ 关键修改：降低 Batch Size 和 Learning Rate 以求稳
    parser.add_argument("--batch_size", type=int, default=1) 
    parser.add_argument("--lr", type=float, default=1e-5) # 调试脚本用的是 1e-5，我们保持一致
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=100)
    
    # 权重配置
    parser.add_argument("--lambda_sms", type=float, default=1.0)
    parser.add_argument("--lambda_cycle", type=float, default=5.0) # 提高内容权重
    parser.add_argument("--lambda_tv", type=float, default=1e-4) # 增加 TV Loss 防止噪点
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(42)

    # ================= 模型加载 =================
    # A. 教师模型 (SD + LoRA)
    pipe = load_sms_teacher(args.sd_path, args.lora_path, device)
    scheduler = DDIMScheduler.from_pretrained(args.sd_path, subfolder="scheduler")
    
    # 预计算 Text Embeddings
    with torch.no_grad():
        text_inputs = pipe.tokenizer([args.style_prompt, ""], padding="max_length", truncation=True, return_tensors="pt")
        text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
        full_text_embeddings = text_embeddings # [2, 77, 768]

    # B. 裁判模型 (CycleReward)
    reward_model = CycleRewardModel().to(device)
    if os.path.exists(args.reward_model_path):
        reward_model.load_state_dict(torch.load(args.reward_model_path, map_location=device))
        print("✅ Reward Model loaded.")
    else:
        print("⚠️ Warning: Reward Model path incorrect, content loss might be random.")
    reward_model.eval()
    
    # C. 学生模型 (Generator)
    net_G = ResnetGenerator().to(device)
    optimizer_G = torch.optim.AdamW(net_G.parameters(), lr=args.lr)

    # ================= 数据准备 =================
    transform = transforms.Compose([
        transforms.Resize(512), 
        transforms.CenterCrop(512), 
        transforms.ToTensor(),
        # 归一化到 [-1, 1] 匹配 Generator 的 Tanh 输出
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    
    # 检查数据目录
    if not os.path.exists(args.content_dir):
        print(f"❌ Error: Content dir {args.content_dir} not found.")
        return

    dataset = ImageFolder(args.content_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    print(f"🚀 Starting Training on {len(dataset)} images for {args.epochs} epochs...")
    
    # ================= 训练循环 =================
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for i, (real_img, _) in enumerate(pbar):
            real_img = real_img.to(device) # [-1, 1]
            
            optimizer_G.zero_grad()

            # 1. 前向生成
            fake_img = net_G(real_img) # [-1, 1]
            
            # 2. SMS Style Loss (SDS)
            # 这里直接调用函数，它会自己在内部执行 latents.backward()
            loss_sms_value = calculate_sds_loss(
                pipe, fake_img, full_text_embeddings, scheduler, device, guidance_scale=7.5
            )

            # 3. Cycle Reward Content Loss
            # 准备输入: [-1, 1] -> [0, 1] -> Resize 224
            real_img_norm = (real_img + 1) / 2
            fake_img_norm = (fake_img + 1) / 2
            
            real_small = F.interpolate(real_img_norm, (224, 224), mode='bilinear')
            fake_small = F.interpolate(fake_img_norm, (224, 224), mode='bilinear')
            
            # Reward Model 希望看到 (Content, Style)，输出越高越好
            reward_score = reward_model(real_small, fake_small)
            
            # 我们要最大化 reward，所以 loss = -reward
            loss_cycle = -torch.mean(reward_score)
            
            # 4. TV Loss (降噪)
            loss_tv = (torch.mean(torch.abs(fake_img[:, :, :, :-1] - fake_img[:, :, :, 1:])) + 
                       torch.mean(torch.abs(fake_img[:, :, :-1, :] - fake_img[:, :, 1:, :])))

            # 5. 组合剩下的 Loss 并 backward
            # 注意：SDS 的梯度已经在第 2 步反向传播了，所以这里只需要传剩下的
            loss_others = (args.lambda_cycle * loss_cycle) + (args.lambda_tv * loss_tv)
            loss_others.backward()
            
            # ⚠️ 关键修复：梯度裁剪 (Gradient Clipping)
            # 这能防止 SDS 偶尔产生的大梯度炸毁模型
            torch.nn.utils.clip_grad_norm_(net_G.parameters(), max_norm=1.0)
            
            optimizer_G.step()
            
            pbar.set_description(f"Ep {epoch} | SMS: {loss_sms_value:.3f} | Cyc: {loss_cycle.item():.3f}")
            
            # --- 保存中间结果 ---
            global_step = epoch * len(dataloader) + i
            if global_step % args.save_interval == 0:
                with torch.no_grad():
                    save_path = os.path.join(args.output_dir, f"step_{global_step}.png")
                    # 拼接原图和生成图
                    vis = torch.cat([real_img, fake_img], dim=3)
                    vis = (vis + 1) / 2 # [-1, 1] -> [0, 1]
                    vis = torch.clamp(vis, 0, 1)
                    save_image(vis, save_path)

    # 保存最终模型
    torch.save(net_G.state_dict(), os.path.join(args.output_dir, "feedforward_final.pth"))
    print("Done! Model saved.")

if __name__ == "__main__":
    main()