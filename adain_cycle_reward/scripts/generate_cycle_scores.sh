#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration (Keep the original path for consistency with the user's setup) ---
SCRIPT_DIR="/home/bjia-25/workspace/papers/gen/adain_cycle_reward"

# Change to the script's directory
cd "$SCRIPT_DIR"
echo "Current working directory: $(pwd)"

# Conda environment name to activate
CONDA_ENV_NAME="acrwd"
# Path to the content images
CONTENT_DIR="./data/content"
# Path to the style images
STYLE_DIR="./data/style"
# Path where the generated scores JSON file will be saved
OUTPUT_JSON="./data/cycle_scores.json"
# Directory for saving stylized images (although not used directly in the Python command, keeping it as a note)
STYLIZED_DIR="./data/stylized"
# ---------------------

# Activate the Conda environment
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
echo "Activated Conda environment: $CONDA_ENV_NAME"


# Start generating stylized images and reward scores
echo "Starting generation of stylized images and cycle scores/rewards..."
python generate_cycle_scores.py \
  --content_dir "$CONTENT_DIR" \
  --style_dir "$STYLE_DIR" \
  --output_json "$OUTPUT_JSON" \
  --alphas 0.3 0.5 0.7 0.9 1.0 \
  --max_samples 2000

# Generation complete message
echo "Generation finished! Results saved to: $OUTPUT_JSON"