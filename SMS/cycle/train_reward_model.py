import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import argparse

class CycleRewardModel(nn.Module):
    def __init__(self, base_model="openai/clip-vit-large-patch14"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(base_model)
        self.config = self.clip.config
        
        # Freeze all parameters first
        for param in self.clip.parameters():
            param.requires_grad = False
            
        # Unfreeze the last layer of the vision encoder for fine-tuning
        for param in self.clip.vision_model.encoder.layers[-1:].parameters():
            param.requires_grad = True
            
        hidden_size = self.config.projection_dim
        
        # MLP Head: Predicts consistency score from (Content, Style) pairs
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1) 
        )
        
    def forward(self, content_pixel_values, style_pixel_values):
        # Extract features
        content_out = self.clip.get_image_features(pixel_values=content_pixel_values)
        style_out = self.clip.get_image_features(pixel_values=style_pixel_values)
        
        # Normalize features
        content_out = content_out / content_out.norm(p=2, dim=-1, keepdim=True)
        style_out = style_out / style_out.norm(p=2, dim=-1, keepdim=True)
        
        # Concatenate and predict score
        combined = torch.cat([content_out, style_out], dim=1)
        score = self.head(combined)
        return score
    
class PreferenceDataset(Dataset):
    def __init__(self, json_file, processor):
        print(f"Loading dataset from {json_file}...")
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        print(f"Dataset loaded: {len(self.data)} triplets.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            content_img = Image.open(item['content_path']).convert("RGB")
            win_img = Image.open(item['win_path']).convert("RGB")
            lose_img = Image.open(item['lose_path']).convert("RGB")
        except Exception as e:
            print(f"Error loading image in triplet {idx}: {e}")
            # Fallback to black images to prevent training crash
            content_img = Image.new('RGB', (224, 224))
            win_img = Image.new('RGB', (224, 224))
            lose_img = Image.new('RGB', (224, 224))

        inputs = self.processor(
            images=[content_img, win_img, lose_img], 
            return_tensors="pt", 
            padding=True
        )
        # return shape: [3, 3, 224, 224] (batch of 3 images)
        return inputs['pixel_values']
    

def train(args):
    print(f"Using preference data from: {args.json_path}")
    print(f"Saving checkpoints to: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    DEVICE = args.device
    print(f"Running on device: {DEVICE}")

    print("Initializing Model and Processor...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CycleRewardModel().to(DEVICE)
    
    dataset = PreferenceDataset(args.json_path, processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Bradley-Terry Loss (Log Sigmoid)
    # Loss = -log(sigmoid(r_win - r_lose))
    def ranking_loss(r_win, r_lose):
        return -torch.mean(torch.nn.functional.logsigmoid(r_win - r_lose))
    
    print(f"Start Training with {len(dataset)} pairs...")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for pixel_values in pbar:
            # pixel_values shape: [batch, 3, channels, h, w]
            pixel_values = pixel_values.to(DEVICE)
            
            # Split into Content, Winner, Loser
            content_imgs = pixel_values[:, 0]
            win_imgs = pixel_values[:, 1]
            lose_imgs = pixel_values[:, 2]
            
            optimizer.zero_grad()
            
            # Predict rewards
            # Reward is P(structural consistency | content, style_output)
            r_win = model(content_imgs, win_imgs)
            r_lose = model(content_imgs, lose_imgs)
            
            # Calculate loss
            loss = ranking_loss(r_win, r_lose)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    print("Training finished!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CycleReward Model")
    
    parser.add_argument('--json_path', type=str, default="./cycle_output/cycle_pref_data.json", 
                        help='Path to the triplet JSON file generated in the previous step')
    parser.add_argument('--output_dir', type=str, default="./reward_model_checkpoint", 
                        help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    
    train(args)