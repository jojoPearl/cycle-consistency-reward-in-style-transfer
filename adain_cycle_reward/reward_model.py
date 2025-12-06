# reward_model.py
import torch
import torch.nn as nn
import timm
from torchvision import transforms

class AdaINCycleRewardModel(nn.Module):
    """
    Reward model using ResNet-18 backbone.
    Input images must be normalized with ImageNet stats.
    """
    def __init__(self, backbone="resnet18", freeze_backbone=False):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        # ResNet-18 outputs 512-dim feature after global avg pool
        self.head = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        # ImageNet normalization (required for pretrained ResNet)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, content, stylized):
        """
        Args:
            content, stylized: [B, 3, H, W] tensors in range [0, 1]
        Returns:
            reward: [B] scalar (higher = better content preservation)
        """
        # Normalize to ImageNet stats
        content = self.normalize(content)
        stylized = self.normalize(stylized)

        f_c = self.backbone(content)      # [B, 512]
        f_s = self.backbone(stylized)     # [B, 512]
        feat = torch.cat([f_c, f_s], dim=1)
        return self.head(feat).squeeze(-1)  # [B]