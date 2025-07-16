#!/usr/bin/env python3
"""
Download official Kyutai models for Unmute.sh
"""
import os
from huggingface_hub import snapshot_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_kyutai_models():
    """Download all required Kyutai models"""
    
    models = [
        {
            "repo_id": "kyutai/stt-1b-en_fr",
            "cache_dir": "./models/stt",
            "name": "Kyutai STT 1B English/French"
        },
        {
            "repo_id": "kyutai/tts-1.6b-en_fr", 
            "cache_dir": "./models/tts",
            "name": "Kyutai TTS 1.6B English/French"
        },
        {
            "repo_id": "kyutai/moshika-pytorch-bf16",
            "cache_dir": "./models/llm",
            "name": "Moshi LLM (Female Voice)"
        }
    ]
    
    for model in models:
        try:
            logger.info(f"Downloading {model['name']}...")
            
            path = snapshot_download(
                repo_id=model["repo_id"],
                cache_dir=model["cache_dir"],
                resume_download=True
            )
            
            logger.info(f"✅ {model['name']} downloaded to: {path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to download {model['name']}: {e}")
            raise

if __name__ == "__main__":
    download_kyutai_models()
