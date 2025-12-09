#!/bin/bash

set -e 

echo "========================================"
echo "Task 1 Started: Generating Candidates"
date
python cycle/generate_candidates.py \
  --input_dir "./data/content" \
  --output_dir "./cycle/output_candidates" \
  --sd_path "Lykon/dreamshaper-8" \
  --lora_path "./lora_ckpt/fechin.safetensors" \
  --style_prompt "oil painting, thick brushstrokes, chiaroscuro lighting, textured canvas" \
  --device "cuda:0"

echo "========================================"
echo "Task 2 Started: Training Reward Model"
date
python cycle/generate_triplets.py \
  --input_dir "./cycle/output_candidates" \
  --output_dir "./cycle/reconstructed_images" \
  --output_json "./cycle/cycle_pref_data.json" \
  --style_prompt "oil" \
  --device "cuda:0"

echo "========================================"
echo "Task 3 Started: Training Reward Model"
date
python train_reward.py \
  --json_path "./cycle_output/cycle_pref_data.json" \
  --output_dir "./reward_model_checkpoint" \
  --batch_size 4 \
  --epochs 1
echo "========================================"

echo "========================================"
echo "Task 4 Started: Training FeedForward"
date
python cycle/train_feedforward.py

echo "========================================"
echo "All Tasks Completed Successfully!"
date