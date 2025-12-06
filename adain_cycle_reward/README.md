# AdaIN Cycle Consistency Reward

Quantifying content preservation in AdaIN style transfer using invertible statistics and perceptual similarity.

<!-- ![teaser](assets/teaser.jpg) -->

This project implements a **cycle-consistency-based reward model** inspired by:

> **Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences**  
> Hyojin Bahng, Caroline Chan, Fredo Durand, Phillip Isola  
> *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025*

We adapt this idea to **image-to-image style transfer**, using **AdaIN's analytically invertible normalization** and **DREAMSim** as a self-supervised reward signal for content fidelity.

## 🎯 Goal
Evaluate how well AdaIN preserves content under different:
- Style images
- Content images
- Style strength (α)

Using **cycle consistency reward**:  
`reward = 1 - DREAMSim(original, reconstructed)`

Higher reward → better content structure preservation.

## 🛠️ Setup
```bash
git clone https://github.com/829183/adain_cycle_reward.git
cd adain_cycle_reward

# Create and activate conda environment
conda env create -f environment.yml
conda activate acrwd

# Alternatively, create manually:
# conda create -n acrwd python=3.10 -y
# conda activate acrwd
# pip install -r requirements.txt
```

## 📂 Data Preparation
- Place your **content images** in `data/content/` (e.g., photos, natural scenes)
- Place your **style images** in `data/style/` (e.g., paintings, artistic works)
> Supported formats: `.jpg`, `.png.` Images will be resized internally.

### Model
This project uses the **pre-trained model** from the AdaIN implementation:
- **File**: `decoder.pth`, `vgg_normalised.pth`
- **Source**: [naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN)
- **Release Version**: `v0.0.0`

> Do **not** substitute with other model variants — they may cause the model to fail to load correctly.

## 📚 References
- Huang, X., & Belongie, S. (2017). Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.
- Bahng, H., Chan, C., Durand, F., & Isola, P. (2025). Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.
- Fu, S., Tamir, N., Sundaram, S., Chai, L., Zhang, R., Dekel, T., & Isola, P. (2023). **DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data**. *Advances in Neural Information Processing Systems (NeurIPS)*. [GitHub](https://github.com/ssundaram21/dreamsim)
- Naoto, Y. (2017). **pytorch-AdaIN**: Unofficial PyTorch implementation of "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization". [GitHub](https://github.com/naoto0804/pytorch-AdaIN)