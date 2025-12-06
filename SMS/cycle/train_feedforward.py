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
import math

# 导入组件
from generator import ResnetGenerator 
from train_reward_model import CycleRewardModel 
from diffusers import StableDiffusionPipeline, DDIMScheduler

def load_sms_teacher(model_id, lora_path, device):
    print(f"Loading Teacher: {model_id}")
    # 强制 FP16 节省显存
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    
    if lora_path:
        # 注意：diffusers 新版本加载 LoRA 的方式可能不同，但在 load_lora_weights 后模型依然是 pipe
        pipe.load_lora_weights(lora_path)
    
    # ❌ 错误写法: pipe.requires_grad_(False)
    
    # ✅ 正确写法: 分别冻结内部组件
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    
    return pipe

def robust_sds_loss(pipe, fake_img, text_embeddings, scheduler, device):
    # 1. 安全编码 (Clamp 防止越界)
    fake_img = torch.clamp(fake_img, -1.0, 1.0)
    
    # 2. VAE Encode (输入转 FP16)
    latents = pipe.vae.encode(fake_img.to(dtype=torch.float16)).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor

    # 3. 加噪
    noise = torch.randn_like(latents)
    # t = torch.randint(20, 980, (latents.shape[0],), device=device).long()
    t = torch.randint(50, 600, (latents.shape[0],), device=device).long()
    noisy_latents = scheduler.add_noise(latents, noise, t)

    # 4. 预测
    latent_input = torch.cat([noisy_latents] * 2)
    with torch.no_grad():
        noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=text_embeddings).sample

    # 5. CFG (降低 Guidance Scale 求稳)
    guidance_scale = 2.5 
    pred_uncond, pred_text = noise_pred.chunk(2)
    noise_pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)

    # 6. 计算梯度并手动 Backward
    grad = (noise_pred - noise)
    
    # [关键保护]：如果梯度有 NaN，置为 0
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 缩放梯度，防止冲击过大
    grad = grad * 0.1 
    
    latents.backward(gradient=grad, retain_graph=True)
    return F.mse_loss(noise_pred, noise).item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", type=str, default="./data/content")
    parser.add_argument("--reward_model_path", type=str, default="./reward_model_checkpoint/epoch_1.pth")
    parser.add_argument("--lora_path", type=str, default="lora_ckpt/fechin.safetensors")
    parser.add_argument("--output_dir", type=str, default="./robust_results")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5) # 降低学习率
    parser.add_argument("--epochs", type=int, default=25) # 单图测试多跑几轮
    
    args = parser.parse_args()
    device = "cuda"
    os.makedirs(args.output_dir, exist_ok=True)

    # 模型加载
    pipe = load_sms_teacher("runwayml/stable-diffusion-v1-5", args.lora_path, device)
    scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    
    # 预计算 Prompt
    with torch.no_grad():
        text_inputs = pipe.tokenizer(["oil painting style", ""], padding="max_length", truncation=True, return_tensors="pt")
        text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

    # Reward Model
    reward_model = CycleRewardModel().to(device)
    try:
        reward_model.load_state_dict(torch.load(args.reward_model_path, map_location=device))
    except:
        print("⚠️ Reward Model 加载失败，将使用随机权重（仅用于测试流程）")
    reward_model.eval()

    # Generator
    net_G = ResnetGenerator().to(device)
    optimizer = torch.optim.AdamW(net_G.parameters(), lr=args.lr)

    # 数据
    transform = transforms.Compose([
        transforms.Resize(512), transforms.CenterCrop(512), transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    dataset = ImageFolder(args.content_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print("🚀 开始稳健训练...")
    
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader)
        for i, (real_img, _) in enumerate(pbar):
            real_img = real_img.to(device)
            optimizer.zero_grad()
            
            # --- 1. Forward ---
            fake_img = net_G(real_img)
            
            # --- 2. Warmup Strategy (关键) ---
            # 前 5 个 Epoch，主要练 L1 Loss，让网络先学会“画画”
            # 之后再逐渐加大风格 Loss
            if epoch < 20: 
                w_style = 0.0
                w_content = 0.0
                w_l1 = 100.0  # 强力拉回原图
            else:
                w_style = 1.0
                w_content = 5.0
                w_l1 = 1.0   

            # --- 3. Losses ---
            loss_log = {}
            
            # A. L1 Pixel Loss (保底)
            loss_l1 = F.l1_loss(fake_img, real_img)
            (loss_l1 * w_l1).backward(retain_graph=True)
            loss_log['L1'] = loss_l1.item()

            # B. Style Loss (SDS)
            if w_style > 0:
                sds_val = robust_sds_loss(pipe, fake_img, text_embeddings, scheduler, device)
                loss_log['SDS'] = sds_val

            # C. Reward Loss
            if w_content > 0:
                # Resize to 224 for CLIP
                real_224 = F.interpolate((real_img+1)/2, (224,224))
                fake_224 = F.interpolate((fake_img+1)/2, (224,224))
                reward = reward_model(real_224, fake_224)
                loss_reward = -torch.mean(reward)
                (loss_reward * w_content).backward()
                loss_log['Rwd'] = loss_reward.item()

            # --- 4. 安全更新 (Safety Update) ---
            # 检查是否有 NaN 梯度
            has_nan_grad = False
            for param in net_G.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        param.grad = None # 清空坏梯度
            
            if has_nan_grad:
                print("⚠️ 警告：检测到 NaN 梯度！本次迭代跳过更新。")
            else:
                # 强力梯度裁剪
                torch.nn.utils.clip_grad_norm_(net_G.parameters(), max_norm=0.5)
                optimizer.step()

            # 打印进度
            desc = f"Ep {epoch} | " + " | ".join([f"{k}: {v:.3f}" for k,v in loss_log.items()])
            pbar.set_description(desc)

            # 保存
            if i == 0 and epoch % 10 == 0:
                with torch.no_grad():
                    vis = torch.cat([real_img, fake_img], dim=3)
                    vis = torch.clamp((vis + 1) / 2, 0, 1)
                    save_image(vis, os.path.join(args.output_dir, f"epoch_{epoch}.png"))

    torch.save(net_G.state_dict(), os.path.join(args.output_dir, "robust_final.pth"))
    print("训练结束。")

if __name__ == "__main__":
    main()