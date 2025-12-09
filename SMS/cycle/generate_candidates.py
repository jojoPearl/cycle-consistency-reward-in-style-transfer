import os
import copy
import json
import torch
import logging
import argparse
import gc
from tqdm import tqdm
from PIL import Image
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

print(f"Added {parent_dir} to sys.path")
from sms import SMS, SMSConfig
from stylize import prepare_latent, train

CANDIDATE_STRATEGIES = {
    "01_best_anchor": {
        "lambda_dct": 5e-4, "num_iters": 500, "guidance_scale": 4.5, "seed": 42
    },
    
    "02_content_collapse": {
        "lambda_dct": 0.0, "num_iters": 700, "guidance_scale": 7.5, "seed": 42
    },
    
    "03_under_stylized": {
        "lambda_dct": 5e-4, "num_iters": 50, "guidance_scale": 2.5, "seed": 42
    },

    "04_high_artifacts": {
        "lambda_dct": 5e-4, "num_iters": 300, "guidance_scale": 9.0, "seed": 42
    },

    "05_best_alt_seed": {
        "lambda_dct": 5e-4, "num_iters": 300, "guidance_scale": 4.5, "seed": 1234
    }
}

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cleanup_cuda():
    gc.collect()
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description='Generate 5 Candidates for CycleReward')
    parser.add_argument('--input_dir', type=str, default="./data/content", help='Input folder with content images')
    parser.add_argument('--output_dir', type=str, default="./cycle/output_candidates", help='Output folder')
    parser.add_argument('--style_prompt', type=str, default="oil painting, thick brushstrokes, chiaroscuro lighting, textured canvas", help='Target style prompt')
    parser.add_argument('--sd_path', type=str, default="Lykon/dreamshaper-8")
    parser.add_argument('--lora_path', type=str, default="lora_ckpt/fechin.safetensors", help='Path to .safetensors LoRA')
    parser.add_argument('--device', type=str, default="cuda:0")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    SRC_PROMPT = "a photo"
    TGT_PROMPT = f"{args.style_prompt}, a photo"

    logging.info(f"Loading SMS Model on {args.device}...")
    sms_config = SMSConfig(
        sd_pretrained_model_or_path=args.sd_path,
        src_prompt=SRC_PROMPT, 
        tgt_prompt=TGT_PROMPT,
        guidance_scale=4.5,
        device=args.device,
        lora_rank=8,
        lora_alpha=32,
        method="sms",
        lora_path=args.lora_path,
    )
    sms_module = SMS(sms_config)
    
    logging.info("Backing up initial weights to CPU...")
    initial_state_dict = {k: v.cpu() for k, v in sms_module.unet.state_dict().items()}
    cleanup_cuda()

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    img_files = sorted([f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in valid_exts])
    
    if not img_files:
        logging.error("No images found in input directory!")
        return

    metadata_log = []

    for img_file in tqdm(img_files, desc="Processing Content Images"):
        img_path = os.path.join(args.input_dir, img_file)
        img_name = os.path.splitext(img_file)[0]
        
        current_save_dir = os.path.join(args.output_dir, img_name)
        os.makedirs(current_save_dir, exist_ok=True)

        try:
            with torch.no_grad():
                src_x0, src_img_pil = prepare_latent(sms_module, img_path, device=args.device)
            
            src_img_pil.save(os.path.join(current_save_dir, "00_original.png"))

            sms_module.config.src_prompt = "a photo" 
            sms_module.config.tgt_prompt = f"{args.style_prompt}, a photo"

            for strat_name, params in CANDIDATE_STRATEGIES.items():
                save_filename = f"{strat_name}.png"
                final_path = os.path.join(current_save_dir, save_filename)

                if os.path.exists(final_path):
                    logging.info(f"Skipping {strat_name}, already exists.")
                    continue

                sms_module.unet.load_state_dict({k: v.to(args.device) for k, v in initial_state_dict.items()})
                
                set_seed(params['seed'])
                
                sms_module.config.guidance_scale = params['guidance_scale']

        
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

                tgt_img.save(final_path)

                metadata_log.append({
                    "content_id": img_name,
                    "candidate_id": strat_name,
                    "image_path": final_path,
                    "params": params
                })
                
                del tgt_x0, tgt_img
                cleanup_cuda()
            
            del src_x0
            del src_img_pil
            cleanup_cuda()

        except Exception as e:
            logging.error(f"Error processing {img_name}: {e}")
            if 'src_x0' in locals(): del src_x0
            cleanup_cuda()
            continue

    with open(os.path.join(args.output_dir, "dataset_index.json"), "w") as f:
        json.dump(metadata_log, f, indent=4)
    
    logging.info(f"Done! Generated dataset at {args.output_dir}")

if __name__ == "__main__":
    main()