import os
import copy
import json
import torch
import logging
import argparse
from tqdm import tqdm
from PIL import Image
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 打印一下路径确认（调试用，没问题后可删除）
print(f"Added {parent_dir} to sys.path")
# 确保引入了你的核心代码
from sms import SMS, SMSConfig
from stylize import prepare_latent, train
from utils import tensor_to_pil, resize_image

CANDIDATE_STRATEGIES = {
    # 1. 基准最佳 (Anchor): 论文推荐参数
    "01_best_anchor": {
        "lambda_dct": 5e-4, "num_iters": 500, "guidance_scale": 4.5, "seed": 42
    },
    
    # 2. 内容崩坏 (Content Collapse): 0正则化，高引导 -> 结构丢失
    "02_content_collapse": {
        "lambda_dct": 0.0, "num_iters": 200, "guidance_scale": 7.5, "seed": 42
    },
    
    # 3. 欠拟合 (Under Stylized): 步数少，引导低 -> 像原图
    "03_under_stylized": {
        "lambda_dct": 5e-4, "num_iters": 50, "guidance_scale": 2.5, "seed": 42
    },

    # 4. 高伪影 (High Artifacts): 极高引导 -> 画面烧焦/噪点
    "04_high_artifacts": {
        "lambda_dct": 5e-4, "num_iters": 200, "guidance_scale": 9.0, "seed": 42
    },

    # 5. 备选最佳 (Alt Seed): 同最佳参数，不同种子 -> 捕捉随机性差异
    "05_best_alt_seed": {
        "lambda_dct": 5e-4, "num_iters": 200, "guidance_scale": 4.5, "seed": 1234
    },
}

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description='Generate 5 Candidates for CycleReward')
    parser.add_argument('--input_dir', type=str, required=True, help='Input folder with content images')
    parser.add_argument('--output_dir', type=str, default="./dataset_cycle_5", help='Output folder')
    parser.add_argument('--style_prompt', type=str, default="oil painting, dramatic lighting, thick brushstrokes", help='Target style prompt')
    parser.add_argument('--sd_path', type=str, default="Lykon/dreamshaper-8")
    parser.add_argument('--lora_path', type=str, default="lora_ckpt/fechin.safetensors", help='Path to .safetensors LoRA')
    parser.add_argument('--device', type=str, default="cuda:0")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    # 1. 初始化模型 (只加载一次)
    logging.info(f"Loading SMS Model on {args.device}...")
    sms_config = SMSConfig(
        sd_pretrained_model_or_path=args.sd_path,
        src_prompt="", 
        tgt_prompt="", # 动态设置
        guidance_scale=4.5,
        device=args.device,
        lora_rank=8,
        lora_alpha=32,
        method="sms",
        lora_path=args.lora_path,
    )
    sms_module = SMS(sms_config)
    
    # *** 关键：备份初始权重，用于每次生成前重置 ***
    initial_state_dict = copy.deepcopy(sms_module.unet.state_dict())

    # 2. 扫描图片
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    img_files = sorted([f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in valid_exts])
    
    if not img_files:
        logging.error("No images found in input directory!")
        return

    metadata_log = []

    # 3. 循环处理
    for img_file in tqdm(img_files, desc="Processing Content Images"):
        img_path = os.path.join(args.input_dir, img_file)
        img_name = os.path.splitext(img_file)[0]
        
        # 为每张图创建一个子文件夹
        current_save_dir = os.path.join(args.output_dir, img_name)
        os.makedirs(current_save_dir, exist_ok=True)

        try:
            # 准备 Latent
            src_x0, src_img_pil = prepare_latent(sms_module, img_path, device=args.device)
            # 保存原图用于参照
            src_img_pil.save(os.path.join(current_save_dir, "00_original.png"))

            # 更新 Prompt
            sms_module.config.src_prompt = "a photo" 
            sms_module.config.tgt_prompt = f"{args.style_prompt}, a photo"

            # 生成 5 张候选图
            for strat_name, params in CANDIDATE_STRATEGIES.items():
                save_filename = f"{strat_name}.png"
                final_path = os.path.join(current_save_dir, save_filename)

                # 跳过已存在的 (方便断点续传)
                if os.path.exists(final_path):
                    continue

                # A. 重置模型 (Reset)
                sms_module.unet.load_state_dict(initial_state_dict)
                
                # B. 设置种子
                set_seed(params['seed'])
                
                # C. 设置参数
                sms_module.config.guidance_scale = params['guidance_scale']

                # D. 训练/生成
                # logging.info(f"  -> Generating {strat_name}...")
                tgt_x0, tgt_img = train(
                    sms_module=sms_module,
                    src_x0=src_x0,
                    output_dir=current_save_dir,
                    img_name=f"temp_{strat_name}",
                    method="sms",
                    lr=1e-1,
                    num_iters=params['num_iters'],
                    optimizer='AdamW',
                    lambda_dct=params['lambda_dct']
                )

                # E. 保存结果
                tgt_img.save(final_path)

                # F. 记录元数据
                metadata_log.append({
                    "content_id": img_name,
                    "candidate_id": strat_name,
                    "image_path": final_path,
                    "params": params
                })
                
                # 清理
                del tgt_x0, tgt_img
                torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Error processing {img_name}: {e}")
            continue

    # 4. 保存索引文件
    with open(os.path.join(args.output_dir, "dataset_index.json"), "w") as f:
        json.dump(metadata_log, f, indent=4)
    
    logging.info(f"Done! Generated dataset at {args.output_dir}")

if __name__ == "__main__":
    main()