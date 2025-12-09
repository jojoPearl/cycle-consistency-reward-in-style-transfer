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
from generator import ResnetGenerator 
from train_reward_model import CycleRewardModel 
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPProcessor, CLIPModel 
def load_sms_teacher(model_id, lora_path, device):
    """ SMS  (SD + Style LoRA)"""
    print(f"Loading SMS Teacher: {model_id} + {lora_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    ).to(device)
    if lora_path and os.path.exists(lora_path):
        pipe.load_lora_weights(lora_path)
    else:
        print(f"⚠️ Warning: LoRA path {lora_path} not found! Using base model only.")
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    return pipe
def calculate_sds_loss(
    pipe, fake_img, text_embeddings, scheduler, device, guidance_scale=7.5
):
    """
     SMS (SDS) 。
    """
    latents = pipe.vae.encode(fake_img.to(dtype=torch.float16)).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor
    bsz = latents.shape[0]
    t = torch.randint(20, 980, (bsz,), device=device).long()
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, t)
    latent_model_input = torch.cat([noisy_latents] * 2)
    t_input = torch.cat([t] * 2)
    with torch.no_grad():
        noise_pred = pipe.unet(
            latent_model_input,
            t_input,
            encoder_hidden_states=text_embeddings,
        ).sample
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    grad = (noise_pred - noise)
    latents.backward(gradient=grad, retain_graph=True)
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
    parser.add_argument("--batch_size", type=int, default=1) 
    parser.add_argument("--lr", type=float, default=1e-5) 
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--lambda_sms", type=float, default=1.0)
    parser.add_argument("--lambda_cycle", type=float, default=5.0) 
    parser.add_argument("--lambda_tv", type=float, default=1e-4) 
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(42)
    pipe = load_sms_teacher(args.sd_path, args.lora_path, device)
    scheduler = DDIMScheduler.from_pretrained(args.sd_path, subfolder="scheduler")
    with torch.no_grad():
        text_inputs = pipe.tokenizer([args.style_prompt, ""], padding="max_length", truncation=True, return_tensors="pt")
        text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
        full_text_embeddings = text_embeddings # [2, 77, 768]
    reward_model = CycleRewardModel().to(device)
    if os.path.exists(args.reward_model_path):
        reward_model.load_state_dict(torch.load(args.reward_model_path, map_location=device))
        print("✅ Reward Model loaded.")
    else:
        print("⚠️ Warning: Reward Model path incorrect, content loss might be random.")
    reward_model.eval()
    net_G = ResnetGenerator().to(device)
    optimizer_G = torch.optim.AdamW(net_G.parameters(), lr=args.lr)
    transform = transforms.Compose([
        transforms.Resize(512), 
        transforms.CenterCrop(512), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    if not os.path.exists(args.content_dir):
        print(f"❌ Error: Content dir {args.content_dir} not found.")
        return
    dataset = ImageFolder(args.content_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"🚀 Starting Training on {len(dataset)} images for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for i, (real_img, _) in enumerate(pbar):
            real_img = real_img.to(device) # [-1, 1]
            optimizer_G.zero_grad()
            fake_img = net_G(real_img) # [-1, 1]
            # 2. SMS Style Loss (SDS)
            loss_sms_value = calculate_sds_loss(
                pipe, fake_img, full_text_embeddings, scheduler, device, guidance_scale=7.5
            )
            # 3. Cycle Reward Content Loss
            real_img_norm = (real_img + 1) / 2
            fake_img_norm = (fake_img + 1) / 2
            real_small = F.interpolate(real_img_norm, (224, 224), mode='bilinear')
            fake_small = F.interpolate(fake_img_norm, (224, 224), mode='bilinear')
            reward_score = reward_model(real_small, fake_small)
            loss_cycle = -torch.mean(reward_score)
            loss_tv = (torch.mean(torch.abs(fake_img[:, :, :, :-1] - fake_img[:, :, :, 1:])) + 
                       torch.mean(torch.abs(fake_img[:, :, :-1, :] - fake_img[:, :, 1:, :])))
            loss_others = (args.lambda_cycle * loss_cycle) + (args.lambda_tv * loss_tv)
            loss_others.backward()
            torch.nn.utils.clip_grad_norm_(net_G.parameters(), max_norm=1.0)
            optimizer_G.step()
            pbar.set_description(f"Ep {epoch} | SMS: {loss_sms_value:.3f} | Cyc: {loss_cycle.item():.3f}")
            global_step = epoch * len(dataloader) + i
            if global_step % args.save_interval == 0:
                with torch.no_grad():
                    save_path = os.path.join(args.output_dir, f"step_{global_step}.png")
                    vis = torch.cat([real_img, fake_img], dim=3)
                    vis = (vis + 1) / 2 # [-1, 1] -> [0, 1]
                    vis = torch.clamp(vis, 0, 1)
                    save_image(vis, save_path)
    torch.save(net_G.state_dict(), os.path.join(args.output_dir, "feedforward_final.pth"))
    print("Done! Model saved.")
if __name__ == "__main__":
    main()