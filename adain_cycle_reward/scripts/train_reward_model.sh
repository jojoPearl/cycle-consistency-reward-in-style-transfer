#!/bin/bash

set -e

SCRIPT_DIR="/home/bjia-25/workspace/papers/gen/adain_cycle_reward"
cd "$SCRIPT_DIR"
echo "Current Workplace: $(pwd)"

CONDA_ENV_NAME="acrwd"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
echo "Conda activated: $CONDA_ENV_NAME"

DATA_JSON="./data/cycle_scores.json"
if [ ! -f "$DATA_JSON" ]; then
    echo "Error: $DATA_JSON not exists please run ./generate_cycle_scores.sh first." 
    exit 1
fi

MODEL_SAVE_PATH="./models/reward_model_resnet18.pth"
echo "Reward Model (ResNet-18)..."
python train_reward_model.py \
  --data_json "$DATA_JSON" \
  --epochs 3 \
  --batch_size 32 \
  --lr 1e-4 \
  --save_path "$MODEL_SAVE_PATH"

echo "Train Finished. Save model to path: $MODEL_SAVE_PATH"