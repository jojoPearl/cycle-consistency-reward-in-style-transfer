# CycleReward: Learning Structural Consistency for Style Transfer

This repository implements **CycleReward**, a novel content preservation mechanism designed to solve the structural collapse problem in neural style transfer.

By integrating a learned **CLIP-based "Judge"**, we enforce strict semantic consistency during the training of style transfer models. We demonstrate the power of CycleReward by applying it to the **[SMS (Style Matching Score)](https://github.com/showlab/SMS)** framework, successfully distilling diffusion-quality style into a lightweight **Feed-Forward Network**.


> **Core Innovation: CycleReward**
> Traditional style transfer often destroys object shapes or hallucinations content. **CycleReward** acts as a learned critic, trained on human preferences, to penalize structural deviations during generation.
>
> In this implementation, CycleReward guides a lightweight generator ($G_\theta$) to learn the style manifold from a Stable Diffusion Teacher (via SDS Loss), ensuring **millisecond-level inference** without sacrificing content fidelity.

-----------------------------------------------------

## 🌟 Key Features

- **⚖️ CycleReward Consistency (Main)**: Integrates a specialized "Judge" model (CLIP-Siamese) trained on human preference triplets. It actively penalizes the generator whenever the stylized output loses the semantic structure of the original content.
- **🛡️ Robust Distillation Pipeline**: To support CycleReward, we implement a **"Robust Mode"** for Score Distillation Sampling (SDS). This prevents gradient explosion (NaN/Magenta Noise) when distilling diffusion priors, ensuring the Reward Model receives valid inputs.
    - *Features: Gradient Normalization, Latent Stats Monitoring, L1 Warmup.*
- **⚡ Real-Time Application**: Demonstrates CycleReward's efficiency by training a ResNet generator for one-step, real-time inference.

----------------------------------------------------

## 🚀 Get Started

### 1. Environment Setup

```bash
# Clone this repository
git clone https://github.com/jojoPearl/cycle-consistency-reward-in-style-transfer.git
cd SMS

# Create Conda Environment
conda create -n acrwd python=3.10 -y
conda activate acrwd

# Install Dependencies
pip install -r requirements.txt
````

### 2\. Prepare Data

#### Content Dataset

Download the COCO Validation set (or use any folder of content images) to train the generator under CycleReward supervision.

```bash
mkdir -p data/content
wget [http://images.cocodataset.org/zips/val2017.zip](http://images.cocodataset.org/zips/val2017.zip)
unzip val2017.zip -d data/content/
```

#### CycleReward Model (Required)

Place your trained CycleReward Model checkpoint here. This is the "Judge" that will guide the training.

```bash
mkdir reward_model_checkpoint
# cp /path/to/your/reward_model.pth ./reward_model_checkpoint/epoch_1.pth
```

#### Style LoRAs

Download your desired style LoRA (e.g., Fechin Oil Painting) for the backend style teacher.

```bash
mkdir lora_ckpt
wget "[https://civitai.com/api/download/models/90795?type=Model&format=SafeTensor](https://civitai.com/api/download/models/90795?type=Model&format=SafeTensor)" -O ./lora_ckpt/fechin.safetensors
```

-----

## 🎨 Training with CycleReward

We use a robust training script that combines the **CycleReward** signal with **SDS (Style)** loss.

### Robust Training Command

The script `train_feedforward_final_fix.py` optimizes the generator. Note that `--lambda_cycle` controls the strength of the CycleReward influence.

```bash
python train_feedforward.py \
  --content_dir "./data/content" \
  --reward_model_path "./reward_model_checkpoint/epoch_1.pth" \
  --lora_path "./lora_ckpt/fechin.safetensors" \
  --output_dir "./output/oil_painting_model" \
  --batch_size 1 \
  --lr 1e-5 \
  --epochs 20 \
  --lambda_cycle 5.0
```

**Key Arguments:**

  * `--lambda_cycle 5.0`: **Crucial.** Controls the weight of the CycleReward. Higher values force the model to adhere more strictly to the original content structure.
  * `--lr 1e-5`: Low learning rate ensures the Reward Model can guide the optimization stably.

### Sanity Check (Single Image)

Verify that CycleReward is correctly preventing structure collapse.

```bash
python train_feedforward.py \
  --content_dir "./data/debug_content" \
  --output_dir "./debug_result" \
  --epochs 100 \
  --save_interval 10
```

-----

## ⚡ Inference (Application)

Once the generator is trained under CycleReward supervision, use this script for inference.

```bash
python inference.py
```

**Configuration (Inside `inference.py`):**

```python
MODEL_PATH = "./output/oil_painting_model/fix_final.pth"
INPUT_DIR = "./data/test_images"
OUTPUT_DIR = "./results"
```

-----

## 📝 Technical Details: Stability & CycleReward

To ensure the **CycleReward** model receives valid inputs for evaluation, we must prevent the generator from collapsing into noise during the early stages of SDS distillation.

Our training pipeline addresses this via:

1.  **L1 Warmup**: Forces the generator to learn content reconstruction (Epochs 0-5) before applying style loss.
2.  **Gradient Normalization**: Scales teacher gradients to a fixed standard deviation (0.1).
3.  **Latent Guardrails**: Ignores gradients if latent statistics drift too far from $N(0,1)$.


## 👏 Acknowledgements

  - **[SMS Official Repo](https://github.com/showlab/SMS)**: For the core theory of Style Matching Score.
  - **[Cycle consistency Reward Repo](https://github.com/hjbahng/cyclereward)**: For the core theory of Reward model training.
  - **HuggingFace Diffusers**: For the Stable Diffusion pipeline.

<!-- end list -->

```
```
