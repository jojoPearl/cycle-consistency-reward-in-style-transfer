import os
import json
import torch
import logging
import argparse
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
from dreamsim import dreamsim


DESTYLIZE_PROMPT = "turn into a realistic photo, high quality, 4k, photorealistic"

class DreamSimWrapper:
    def __init__(self, device):
        self.model, self.preprocess_fn = dreamsim(pretrained=True)
        self.model = self.model.to(device).eval()
        self.device = device

    def preprocess(self, pil_image):
        """封装预处理：PIL -> Tensor -> Device"""
        # dreamsim 的预处理返回 tensor，我们需要手动移动到 device
        return self.preprocess_fn(pil_image).to(self.device)

    def __call__(self, img_tensor_a, img_tensor_b):
        """封装推理：计算距离"""
        return self.model(img_tensor_a, img_tensor_b)

def load_models(device):
    logging.info("Loading InstructPix2Pix...")
    # 加载去风格化模型
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", 
        torch_dtype=torch.float16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    logging.info("Loading DreamSim Metric...")
    # 使用包装类加载 DreamSim，解决 Tuple 报错问题
    metric = DreamSimWrapper(device)
    
    return pipe, metric

def process_single_group(pipe, metric, img_dir, device):
    """处理单个内容图文件夹，返回三元组"""
    if not os.path.exists(img_dir):
        return []
        
    files = os.listdir(img_dir)
    
    # 1. 找到原图
    orig_name = "00_original.png"
    if orig_name not in files:
        return []
    
    candidates = [f for f in files if f != orig_name and f.endswith('.png')]
    # 至少要有两个候选图才能构建对比
    if len(candidates) < 2:
        return []

    # 加载原图
    orig_path = os.path.join(img_dir, orig_name)
    try:
        orig_img = Image.open(orig_path).convert("RGB")
    except Exception as e:
        logging.warning(f"Failed to open original image {orig_path}: {e}")
        return []
    
    # 使用 Wrapper 预处理原图
    orig_tensor = metric.preprocess(orig_img)

    scores = []
    
    # 2. 对每个候选图做回环 (Backward)
    for cand_name in candidates:
        cand_path = os.path.join(img_dir, cand_name)
        try:
            cand_img = Image.open(cand_path).convert("RGB")
        except Exception:
            continue
        
        # A. 去风格化 (Inference)
        # image_guidance_scale=1.5 兼顾原图结构和去风格化指令
        with torch.no_grad():
            rec_img = pipe(
                prompt=DESTYLIZE_PROMPT,
                image=cand_img,
                num_inference_steps=20, # 20步足够，追求速度
                image_guidance_scale=1.5,
                guidance_scale=7.5
            ).images[0]
        
        # B. 计算分数 (Metric)
        # 预处理重建图
        rec_tensor = metric.preprocess(rec_img)
        
        # 计算距离 (越小越好)
        with torch.no_grad():
            distance = metric(orig_tensor, rec_tensor).item()
        
        scores.append({
            "cand_path": cand_path,
            "score": distance, # DreamSim Distance (0~1)
            "name": cand_name
        })

    # 3. 构建三元组 (Triplets)
    triplets = []
    
    scores.sort(key=lambda x: x['score']) 

    # 简单的两两组合策略
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            winner = scores[i]
            loser = scores[j]
            
    
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
    # 默认路径修改为你生成脚本的输出路径
    parser.add_argument('--input_dir', type=str, default="./output_candidates", help="Folder containing candidate images")
    parser.add_argument('--output_json', type=str, default="./cycle_pref_data.json")
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    
    pipe, metric = load_models(args.device)
    
    all_triplets = []
    
    # 遍历 input_dir 下的所有子文件夹
    if not os.path.exists(args.input_dir):
        logging.error(f"Input directory {args.input_dir} does not exist.")
        return

    subfolders = [f.path for f in os.scandir(args.input_dir) if f.is_dir()]
    
    logging.info(f"Found {len(subfolders)} image groups. Starting Backward Loop...")

    for folder in tqdm(subfolders):
        try:
            folder_triplets = process_single_group(pipe, metric, folder, args.device)
            all_triplets.extend(folder_triplets)
        except Exception as e:
            logging.error(f"Error processing {folder}: {e}")
            continue
            
    logging.info(f"Generated {len(all_triplets)} triplets.")
    
    # 保存结果
    with open(args.output_json, 'w') as f:
        json.dump(all_triplets, f, indent=4)
    
    logging.info(f"Saved dataset to {args.output_json}")

if __name__ == "__main__":
    main()