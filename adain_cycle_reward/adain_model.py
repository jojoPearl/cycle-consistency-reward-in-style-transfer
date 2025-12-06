# This code is adapted from the implementation of 
# "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"
# by Naoto Inoue (https://github.com/naoto0804/pytorch-AdaIN).
# Original paper: X. Huang and S. Belongie, ICCV 2017.

import torch
import torch.nn as nn

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
    nn.Tanh()
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std per channel over spatial dimensions."""
    assert feat.dim() == 4
    N, C = feat.size(0), feat.size(1)
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    """
    AdaIN: transfer style statistics (mean/std) from style to content.
    Args:
        content_feat: [N, C, H, W]
        style_feat:   [N, C, H, W]
    Returns:
        stylized_feat: [N, C, H, W]
    """
    assert content_feat.size()[:2] == style_feat.size()[:2], "Batch & channel must match"
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_content = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    stylized_feat = normalized_content * style_std.expand(size) + style_mean.expand(size)
    return stylized_feat

def inverse_adaptive_instance_normalization(stylized_feat, content_stats, style_stats):
    """
    Inverse AdaIN: recover original content feature from stylized feature.
    Used in cycle-consistency or reconstruction tasks.
    Args:
        stylized_feat: [N, C, H, W] — output of AdaIN
        content_stats: dict with 'mean', 'std' of original content
        style_stats:   dict with 'mean', 'std' of style
    Returns:
        recovered_content_feat: [N, C, H, W]
    """
    size = stylized_feat.size()
    c_mean = content_stats['mean']
    c_std = content_stats['std']
    s_mean = style_stats['mean']
    s_std = style_stats['std']

    # Step 1: Remove style statistics → get normalized content
    normalized_content = (stylized_feat - s_mean.expand(size)) / s_std.expand(size)
    # Step 2: Re-apply original content statistics
    recovered = normalized_content * c_std.expand(size) + c_mean.expand(size)
    return recovered

class AdaINModel(nn.Module):
    def __init__(self, vgg_path, decoder_path):
        super().__init__()

        # Encoder
        self.encoder = vgg
        self.encoder.load_state_dict(torch.load(vgg_path, weights_only=False))
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder = nn.Sequential(*list(self.encoder.children())[:31])

        # Decoder
        self.decoder = decoder
        self.decoder.load_state_dict(torch.load(decoder_path, weights_only=False))
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False

    def decode(self, x):
        x = self.decoder(x)
        x = torch.clamp((x + 1) / 2, 0, 1)
        return x

    def encode(self, x):
        return self.encoder(x)

    def forward(self, content, style, alpha=1.0):
        """
        Args:
            content/style: tensor in [0, 1], shape (B, 3, H, W)
            alpha: float in [0, 1]
        Returns:
            stylized: tensor in [0, 1]
            stats: dict of feature statistics
        """
        assert 0 <= alpha <= 1.0

        content_feat = self.encode(content)
        style_feat = self.encode(style)

        # Compute mean and std (AdaIN stats)
        c_mean, c_std = calc_mean_std(content_feat)
        s_mean, s_std = calc_mean_std(style_feat)

        # AdaIN fusion
        stylized_feat = adaptive_instance_normalization(content_feat, style_feat)

        # Optional interpolation with original content feature
        blended_feat = alpha * stylized_feat + (1 - alpha) * content_feat

        # Decode
        stylized = self.decode(blended_feat)

        stats = {
            'content': {'mean': c_mean, 'std': c_std},
            'style':   {'mean': s_mean, 'std': s_std}
        }
        return stylized, stats

    def inverse_adain(self, stylized_feat, stats):
        """
        Recover original content feature from stylized feature using stored stats.
        Useful for cycle reconstruction.
        """
        return inverse_adaptive_instance_normalization(
            stylized_feat,
            content_stats=stats['content'],
            style_stats=stats['style']
        )