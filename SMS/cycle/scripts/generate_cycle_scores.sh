set -e

echo "Starting generation of stylized images and cycle scores/rewards..."
python cycle/train_feedforward.py   --content_dir "./data/content"   --reward_model_path "./reward_model_checkpoint/epoch_1.pth"   --style_prompt "oil painting, dramatic lighting, thick brushstrokes"   --lambda_style 10.0   --lambda_content 5.0
echo "Generation finished! Results saved to: ./data/cycle_scores.json"