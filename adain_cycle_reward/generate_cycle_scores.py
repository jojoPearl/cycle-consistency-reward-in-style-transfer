import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import json
from tqdm import tqdm
from dreamsim import dreamsim
import hashlib

# Assuming your AdaINModel is defined in adain_model.py
from adain_model import AdaINModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Please modify this to your actual path
base_path = "/home/bjia-25/workspace/papers/gen/adain_cycle_reward" + "/"
vgg_path = base_path + "models/vgg_normalised.pth"
decoder_path = base_path + "models/decoder.pth"

class PairDataset(Dataset):
    def __init__(self, content_paths, style_paths, alphas, transform=None):
        self.content_paths = content_paths
        self.style_paths = style_paths
        self.alphas = alphas
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)), # Suggested fixed size to avoid tensor dimension mismatch
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.content_paths) * len(self.style_paths) * len(self.alphas)

    def __getitem__(self, idx):
        n_c = len(self.content_paths)
        n_s = len(self.style_paths)
        n_a = len(self.alphas)

        i = idx // (n_s * n_a)
        j = (idx % (n_s * n_a)) // n_a
        k = idx % n_a

        content_path = self.content_paths[i]
        style_path = self.style_paths[j]
        alpha = self.alphas[k]

        content = Image.open(content_path).convert("RGB")
        style = Image.open(style_path).convert("RGB")

        if self.transform:
            content = self.transform(content)
            style = self.transform(style)

        return content, style, alpha, content_path, style_path

def load_image_list(root, exts=(".jpg", ".png", ".jpeg")):
    paths = []
    if os.path.isfile(root):
        return [root]
        
    for fname in sorted(os.listdir(root)):
        if fname.lower().endswith(exts):
            paths.append(os.path.join(root, fname))
    return paths

def hash_file_name(content_path, style_path, alpha_val):
    content_name = os.path.basename(content_path).split('.')[0]
    style_name = os.path.basename(style_path).split('.')[0]
    hash_id = hashlib.md5(f"{content_name}_{style_name}_{alpha_val}".encode()).hexdigest()[:8]
    return f"{hash_id}.jpg"

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", type=str, default="data/content")
    parser.add_argument("--style_dir", type=str, default="data/style", 
                        help="Directory containing style images if --style_file is not used.")
    parser.add_argument("--style_file", type=str, default=None,
                        help="Specific style image file path to use. Overrides --style_dir if set.")
    parser.add_argument("--stylized_dir", type=str, default="data/stylized")
    parser.add_argument("--output_json", type=str, default="data/cycle_scores.json")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9, 1.0])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=2000)
    args = parser.parse_args()

    os.makedirs(args.stylized_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    # Load model
    print("Loading AdaIN model...")
    model = AdaINModel(vgg_path=vgg_path, decoder_path=decoder_path).to(device).eval()
    
    print("Loading DreamSim metric...")
    dreamsim_model, _ = dreamsim(pretrained=True, device=device)

    # Load data
    content_paths = load_image_list(args.content_dir)
    # For testing flow, you can slice here, e.g., content_paths[:50]
    
    if args.style_file:
        if not os.path.isfile(args.style_file):
             raise FileNotFoundError(f"Specified style file not found: {args.style_file}")
        style_paths = [args.style_file]
        print(f"Using only specified style file: {args.style_file}")
    else:
        style_paths = load_image_list(args.style_dir)
        # For testing flow, you can slice here, e.g., style_paths[:10]
        print(f"Using {len(style_paths)} style images from: {args.style_dir}")

    if not content_paths or not style_paths:
        print("Error: Content or style image list is empty. Aborting.")
        return
    
    dataset = PairDataset(content_paths, style_paths, args.alphas)
    
    if len(dataset) > args.max_samples:
        print(f"Warning: dataset size {len(dataset)} > max_samples {args.max_samples}. Truncating.")
        from torch.utils.data import Subset
        indices = torch.randperm(len(dataset))[:args.max_samples].tolist()
        dataset = Subset(dataset, indices)

    # batch_size must be 1 because save_image and subsequent logic handle single images
    dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True)

    results = []

    for batch in tqdm(dataloader, desc="Generating cycle scores"):
        content, style, alpha, content_path, style_path = batch
        content = content.to(device)
        style = style.to(device)
        alpha_val = alpha.item()

        # ==========================================
        # 1. Forward AdaIN (Generate Stylized Image)
        # ==========================================
        orig = content
        
        # [Modification] Do not receive stats to prevent misuse in backward process
        # Original code: stylized, stats = model(orig, style, alpha=alpha_val)
        stylized, _ = model(orig, style, alpha=alpha_val)

        # ==========================================
        # 2. Inverse cycle (Backward Reconstruction)
        # ==========================================
        
        # --------- [Commented Out] Original Incorrect Logic (Cheating) ---------
        # This approach uses the mean/std (stats) of the original image to assist reconstruction, which is cheating.
        # stylized_feat = model.encode(stylized)
        # rec_feat = model.inverse_adain(stylized_feat, stats)
        # rec = model.decode(rec_feat)
        # ------------------------------------------------
        
        # +++++++++ [New Logic] Correct Cycle Consistency +++++++++
        # Logic: Treat the generated Stylized image as Input and try to transfer it back to the Content's style.
        # Here, the Content image itself is treated as the "Target Style", with alpha set to 1.0 to force alignment.
        rec, _ = model(stylized, content, alpha=1.0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++

        # ==========================================
        # 3. Compute Metrics & Save
        # ==========================================
        
        # Suggestion: If DreamSim library version has preprocess, it's better to call it here.
        # Currently assuming content/rec are already in [0,1] range and have appropriate dimensions.
        dist = dreamsim_model(orig, rec).item()
        
        # Simple conversion from distance to reward (smaller distance, higher reward)
        reward = 1.0 - dist 

        # Save stylized image
        # Unpack path tuple
        c_path = content_path[0]
        s_path = style_path[0]
        
        filename = hash_file_name(c_path, s_path, alpha_val)
        stylized_save_path = os.path.join(args.stylized_dir, filename)
        
        save_image(stylized.squeeze(0).cpu(), stylized_save_path)

        results.append({
            "content_path": c_path,
            "style_path": s_path,
            "alpha": alpha_val,
            "stylized_path": stylized_save_path,
            "reward": reward,
            "dreamsim_distance": dist
        })

    # Save
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} samples to {args.output_json}")

if __name__ == "__main__":
    main()