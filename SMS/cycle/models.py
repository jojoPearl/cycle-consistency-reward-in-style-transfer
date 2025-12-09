from huggingface_hub import snapshot_download

print("Downloading InstructPix2Pix...")
local_path = snapshot_download(
    repo_id="timbrooks/instruct-pix2pix",
    local_dir="./models/instruct-pix2pix",
    local_dir_use_symlinks=False,
    resume_download=True
)
print(f"Downloaded to: {local_path}")