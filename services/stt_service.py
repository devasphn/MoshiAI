import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class KyutaiSTTService:
    """Production-ready Kyutai STT Service"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        
    async def initialize(self, models_dir: Path):
        """Initialize STT model and tokenizer"""
        try:
            logger.info("Initializing Kyutai STT Service...")
            
            # Import required modules
            try:
                from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
                import sentencepiece as spm
            except ImportError as e:
                logger.error(f"Required dependencies not installed: {e}")
                return False
            
            # Find model directory
            model_dir = models_dir / "stt"
            stt_path = None
            
            for path in model_dir.rglob("*"):
                if "kyutai" in str(path) and "stt" in str(path):
                    stt_path = path
                    break
            
            if not stt_path:
                logger.error("STT model directory not found")
                return False
            
            # Load model and processor
            try:
                self.processor = AutoProcessor.from_pretrained(str(stt_path))
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    str(stt_path),
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto" if self.device.type == "cuda" else None
                )
                self.model.eval()
                
                logger.info("✅ STT model loaded successfully")
                self.is_initialized = True
                return True
                
            except Exception as e:
                logger.error(f"Failed to load STT model: {e}")
                return False
                
        except Exception as e:
            logger.error(f"STT initialization error: {e}")
            return False
    
    async def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio to text"""
        if not self.is_initialized or len(audio_data) == 0:
            return ""
        
        try:
            # Preprocess audio
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Process with model
            inputs = self.processor(
                audio_data,
                sampling_rate=16000,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs["input_features"],
                    max_length=200,
                    do_sample=False,
                    temperature=0.0
                )
            
            # Decode result
            transcription = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()
            
            logger.info(f"STT transcription: '{transcription}'")
            return transcription
            
        except Exception as e:
            logger.error(f"STT transcription error: {e}")
            return ""
