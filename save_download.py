from huggingface_hub import hf_hub_download
import os
import shutil

# Replace with your actual Hugging Face token
hf_token = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"  # Get from https://huggingface.co/settings/tokens

# Dataset details
repo_id = "ai4bharat/IndicVoices"
filename = "assamese/valid/15.wav"  # Path to the specific WAV file

# Download the file
local_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset",
    token=hf_token  # Required for authenticated access
)

# Rename and move to voices/ directory
target_dir = "voices"
os.makedirs(target_dir, exist_ok=True)
target_path = os.path.join(target_dir, "indian_female.wav")
shutil.copy(local_path, target_path)

print(f"File downloaded and saved as: {target_path}")
