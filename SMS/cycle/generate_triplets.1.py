import os
import json
import torch
import logging
import argparse
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
from dreamsim import dreamsim
# Added accelerate for CPU offload (saves VRAM)
from accelerate import Accelerator 
from accelerate.utils import is_ml_platform_available

# Define mapping for style-specific destylization prompts
DESTYLIZE_MAPPING = {
    "oil": "remove oil painting texture, smooth out brushstrokes, turn into a realistic photo",
    "impressionism": "remove impressionism style, sharpen details, turn into a realistic photo",
    "ukiyo-e": "remove outlines, add realistic shading, turn into a realistic photo",
    "cyberpunk": "remove neon lighting, use natural lighting, turn into a realistic photo"
}

class DreamSimWrapper:
    def __init__(self, device):
        self.model, self.preprocess_fn = dreamsim(pretrained=True)
        self.model = self.model.eval()
        self.device = device
        
        # Robust device handling
        if "cuda" in str(device) and not is_ml_platform_available("huggingface"):
             self.model = self.model.to(device)

    def preprocess(self, pil_image):
        return self.preprocess_fn(pil_image).to(self.device)
    
    def __call__(self, img_tensor_a, img_tensor_b):
        return self.model(img_tensor_a, img_tensor_b)
    
def load_models(device):
    logging.info("Loading InstructPix2Pix...")
    # Load with float16 to save memory
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", 
        torch_dtype=torch.float16
    )
    # Enable CPU offload to prevent OOM errors on large batches
    pipe.enable_model_cpu_offload() 
    pipe.set_progress_bar_config(disable=True)

    logging.info("Loading DreamSim Metric...")
    metric = DreamSimWrapper(device)
    
    return pipe, metric

def process_single_group(pipe, metric, img_dir, rec_save_dir, device, destylize_prompt):
    if not os.path.exists(img_dir):
        return []
        
    files = os.listdir(img_dir)
    
    # 1. Identify Original Image
    orig_name = "00_original.png"
    if orig_name not in files:
        return []
    
    candidates = [f for f in files if f != orig_name and f.endswith('.png')]
    if len(candidates) < 2:
        return []

    # Ensure reconstruction save path exists
    os.makedirs(rec_save_dir, exist_ok=True)

    # Load Original Image
    orig_path = os.path.join(img_dir, orig_name)
    try:
        orig_img = Image.open(orig_path).convert("RGB")
    except Exception as e:
        logging.warning(f"Failed to open original image {orig_path}: {e}")
        return []
    
    orig_tensor = metric.preprocess(orig_img)
    scores = []
    
    # 2. Process Candidates (Backward Cycle)
    for cand_name in candidates:
        cand_path = os.path.join(img_dir, cand_name)
        try:
            cand_img = Image.open(cand_path).convert("RGB")
        except Exception:
            continue
        
        # A. Destylize (Inference)
        # Using the specific 'destylize_prompt' passed to the function
        with torch.no_grad():
            rec_img = pipe(
                prompt=destylize_prompt,
                image=cand_img,
                num_inference_steps=20, # Reduced steps for speed
                image_guidance_scale=1.5,
                guidance_scale=7.5
            ).images[0]
        
        # Save Reconstructed Image
        rec_img_name = f"rec_{cand_name}"
        rec_img_path = os.path.join(rec_save_dir, rec_img_name)
        rec_img.save(rec_img_path)
        # logging.debug(f"Saved reconstructed image to {rec_img_path}")

        # B. Calculate Distance (Metric)
        rec_tensor = metric.preprocess(rec_img)
        with torch.no_grad():
            distance = metric(orig_tensor, rec_tensor).item()
        
        scores.append({
            "cand_path": cand_path,
            "score": distance, # DreamSim Distance (0~1, lower is better)
            "name": cand_name
        })

    # 3. Generate Triplets
    triplets = []
    scores.sort(key=lambda x: x['score']) 

    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            winner = scores[i]
            loser = scores[j]
            # Threshold to ensure significant difference
            if (loser['score'] - winner['score']) > 0.05:
                triplets.append({
                    "content_path": orig_path,
                    "win_path": winner['cand_path'],
                    "lose_path": loser['cand_path'],
                    "win_score": winner['score'],
                    "lose_score": loser['score']
                })
    return triplets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="./output_candidates", help="Folder containing candidate images")
    parser.add_argument('--output_dir',type=str, default="./cycle_output/reconstructed_images", help="Base directory to save ALL reconstructed images.")
    parser.add_argument('--output_json', type=str, default="./cycle_output/cycle_pref_data.json")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--style_prompt', type=str, default="oil", help="The style prompt used for generation, used to select the best destylize prompt.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    
    # Dynamic Prompt Selection
    destylize_prompt = "turn into a realistic photo, high quality, 4k, photorealistic" # Default Fallback

    if args.style_prompt:
        lower_prompt = args.style_prompt.lower()
        if "oil" in lower_prompt:
            destylize_prompt = DESTYLIZE_MAPPING["oil"]
        elif "impressionism" in lower_prompt:
            destylize_prompt = DESTYLIZE_MAPPING["impressionism"]
        elif "ukiyo-e" in lower_prompt:
            destylize_prompt = DESTYLIZE_MAPPING["ukiyo-e"]
        elif "cyberpunk" in lower_prompt:
            destylize_prompt = DESTYLIZE_MAPPING["cyberpunk"]
            
    logging.info(f"Using Destylization Prompt: '{destylize_prompt}'")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)        

    pipe, metric = load_models(args.device)
    
    all_triplets = []
    if not os.path.exists(args.input_dir):
        logging.error(f"Input directory {args.input_dir} does not exist.")
        return
    
    subfolders = [f.path for f in os.scandir(args.input_dir) if f.is_dir()]
    
    logging.info(f"Found {len(subfolders)} image groups. Starting Backward Loop...")
    
    for folder in tqdm(subfolders):
        try:
            # Create unique subfolder for this group's reconstruction
            group_name = os.path.basename(folder)
            rec_group_dir = os.path.join(args.output_dir, group_name)
            
            # Pass the selected destylize_prompt to the processing function
            folder_triplets = process_single_group(pipe, metric, folder, rec_group_dir, args.device, destylize_prompt)
            all_triplets.extend(folder_triplets)
        except Exception as e:
            logging.error(f"Error processing {folder}: {e}")
            continue
            
    logging.info(f"Generated {len(all_triplets)} triplets.")
    
    with open(args.output_json, 'w') as f:
        json.dump(all_triplets, f, indent=4)
    
    logging.info(f"Saved dataset to {args.output_json}")

if __name__ == "__main__":
    main()