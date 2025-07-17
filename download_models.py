#!/usr/bin/env python3
"""
Download official Kyutai models for MoshiAI Voice Assistant
"""
import os
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_kyutai_models():
    """Download all required Kyutai models"""
    
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    models = [
        {
            "repo_id": "kyutai/stt-1b-en_fr",
            "cache_dir": models_dir / "stt",
            "name": "Kyutai STT 1B English/French"
        },
        {
            "repo_id": "kyutai/tts-1.6b-en_fr", 
            "cache_dir": models_dir / "tts",
            "name": "Kyutai TTS 1.6B English/French"
        },
        {
            "repo_id": "kyutai/moshika-pytorch-bf16",
            "cache_dir": models_dir / "llm",
            "name": "Moshi LLM"
        }
    ]
    
    for model in models:
        try:
            logger.info(f"Downloading {model['name']}...")
            
            path = snapshot_download(
                repo_id=model["repo_id"],
                cache_dir=str(model["cache_dir"]),
                resume_download=True,
                local_files_only=False
            )
            
            logger.info(f"✅ {model['name']} downloaded to: {path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to download {model['name']}: {e}")
            continue
    
    logger.info("🎉 All models downloaded successfully!")

if __name__ == "__main__":
    download_kyutai_models()
