import argparse
import json
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from reward_model import AdaINCycleRewardModel

device = "cuda" if torch.cuda.is_available() else "cpu"

class PreferenceDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Group by (content, style)
        groups = {}
        for item in data:
            key = (item["content_path"], item["style_path"])
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        
        # Build preference pairs: best vs worst per group
        self.pairs = []
        for group in groups.values():
            if len(group) < 2:
                continue
            group.sort(key=lambda x: x["reward"], reverse=True)
            best = group[0]
            for worse in group[1:]:
                self.pairs.append((best, worse))
        
        # Transform: resize to 224x224 (ResNet input size)
        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        best, worse = self.pairs[idx]
        content = Image.open(best["content_path"]).convert("RGB")
        preferred = Image.open(best["stylized_path"]).convert("RGB")
        rejected = Image.open(worse["stylized_path"]).convert("RGB")

        return (
            self.transform(content),
            self.transform(preferred),
            self.transform(rejected),
            best["reward"],
            worse["reward"]
        )

def collate_fn(batch):
    contents, prefs, rejs, r_prefs, r_rejs = zip(*batch)
    return (
        torch.stack(contents),
        torch.stack(prefs),
        torch.stack(rejs),
        torch.tensor(r_prefs, dtype=torch.float32),
        torch.tensor(r_rejs, dtype=torch.float32)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", type=str, default="data/cycle_scores.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="models/reward_model_resnet18.pth")
    args = parser.parse_args()

    dataset = PreferenceDataset(args.data_json)
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    model = AdaINCycleRewardModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Training on {len(dataset)} preference pairs")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for content, preferred, rejected, _, _ in pbar:
            content = content.to(device)
            preferred = preferred.to(device)
            rejected = rejected.to(device)

            optimizer.zero_grad()
            r_pref = model(content, preferred)
            r_rej = model(content, rejected)
            loss = -torch.log(torch.sigmoid(r_pref - r_rej)).mean()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"✅ Model saved to {args.save_path}")

if __name__ == "__main__":
    main()