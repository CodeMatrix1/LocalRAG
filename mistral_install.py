from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from pathlib import Path
import os
from dotenv import load_dotenv

# Download embedding model (cached locally)
print("Downloading embedding model...")
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

# Download Mistral model
print("Downloading Mistral model...")
mistral_dir = Path("models/mistral")
load_dotenv() 
snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
    local_dir=mistral_dir,
    allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"],
    resume_download=True
)

print("Models downloaded successfully!")
