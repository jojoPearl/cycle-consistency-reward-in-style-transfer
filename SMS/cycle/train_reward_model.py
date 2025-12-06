import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from tqdm import tqdm

# ================= 模型定义 =================
class CycleRewardModel(nn.Module):
    def __init__(self, base_model="openai/clip-vit-large-patch14"):
        super().__init__()
        # 使用 CLIP 作为视觉骨干
        self.clip = CLIPModel.from_pretrained(base_model)
        self.config = self.clip.config
        
        # 冻结大部分层，只训练最后几层和 MLP (适应小数据，防止过拟合)
        for param in self.clip.parameters():
            param.requires_grad = False
            
        # 解冻 Vision Model 的最后两个 Layer (可选)
        for param in self.clip.vision_model.encoder.layers[-1:].parameters():
            param.requires_grad = True

        # 打分头 (MLP Head)
        # 输入是 Content Embedding + Style Embedding (768 * 2)
        hidden_size = self.config.projection_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1) # 输出标量分数
        )

    def forward(self, content_pixel_values, style_pixel_values):
        # 1. 提取特征
        content_out = self.clip.get_image_features(pixel_values=content_pixel_values)
        style_out = self.clip.get_image_features(pixel_values=style_pixel_values)
        
        # 2. 归一化 (CLIP 特征通常需要归一化)
        content_out = content_out / content_out.norm(p=2, dim=-1, keepdim=True)
        style_out = style_out / style_out.norm(p=2, dim=-1, keepdim=True)
        
        # 3. 拼接特征
        combined = torch.cat([content_out, style_out], dim=1)
        
        # 4. 预测分数
        score = self.head(combined)
        return score

# ================= 数据集定义 =================
class PreferenceDataset(Dataset):
    def __init__(self, json_file, processor):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载三张图
        content_img = Image.open(item['content_path']).convert("RGB")
        win_img = Image.open(item['win_path']).convert("RGB")
        lose_img = Image.open(item['lose_path']).convert("RGB")
        
        # 预处理
        inputs = self.processor(
            images=[content_img, win_img, lose_img], 
            return_tensors="pt", 
            padding=True
        )
        
        # 返回像素值: [3, 3, 224, 224] -> 分别对应 content, win, lose
        return inputs['pixel_values']

# ================= 训练主流程 =================
def train():
    # 配置
    JSON_PATH = "./cycle_pref_data.json"
    print(f"Using preference data from: {JSON_PATH}")
    OUTPUT_DIR = "./reward_model_checkpoint"
    print(f"Saving checkpoints to: {OUTPUT_DIR}")
    BATCH_SIZE = 4 # 显存小就调小
    LR = 1e-5
    EPOCHS = 1 # 数据少多跑几轮，数据多跑1-2轮
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 初始化
    print("Initializing Model and Processor...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CycleRewardModel().to(DEVICE)
    
    # 2. 数据准备
    dataset = PreferenceDataset(JSON_PATH, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 3. 优化器 & Loss
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    
    # Bradley-Terry Loss (Log Sigmoid)
    # Loss = -log(sigmoid(r_win - r_lose))
    def ranking_loss(r_win, r_lose):
        return -torch.mean(torch.nn.functional.logsigmoid(r_win - r_lose))

    # 4. 训练循环
    print(f"Start Training with {len(dataset)} pairs...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for pixel_values in pbar:
            # pixel_values shape: [batch, 3, channels, h, w]
            pixel_values = pixel_values.to(DEVICE)
            
            # 拆分 Batch
            content_imgs = pixel_values[:, 0]
            win_imgs = pixel_values[:, 1]
            lose_imgs = pixel_values[:, 2]
            
            optimizer.zero_grad()
            
            # 前向传播：计算好图得分
            r_win = model(content_imgs, win_imgs)
            
            # 前向传播：计算坏图得分
            r_lose = model(content_imgs, lose_imgs)
            
            # 计算 Loss
            loss = ranking_loss(r_win, r_lose)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # 保存权重
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}.pth"))

    print("Training finished!")

if __name__ == "__main__":
    train()