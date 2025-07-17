import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional
import asyncio

logger = logging.getLogger(__name__)

class KyutaiSTTService:
    """Real Kyutai STT Service using actual model inference"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.stt_model = None
        self.tokenizer = None
        self.is_initialized = False
        
    async def initialize(self, models_dir: Path):
        """Initialize real STT model"""
        try:
            logger.info("Initializing Real Kyutai STT Service...")
            
            # Find STT model directory
            model_dir = models_dir / "stt" / "models--kyutai--stt-1b-en_fr" / "snapshots"
            stt_path = None
            
            if model_dir.exists():
                snapshot_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
                if snapshot_dirs:
                    stt_path = snapshot_dirs[0]
                    logger.info(f"Found STT model at: {stt_path}")
            
            if not stt_path:
                logger.error("STT model not found")
                return False
            
            # Load tokenizer
            try:
                import sentencepiece as spm
                tokenizer_file = stt_path / "tokenizer_en_fr_audio_8000.model"
                if tokenizer_file.exists():
                    self.tokenizer = smp.SentencePieceProcessor()
                    self.tokenizer.load(str(tokenizer_file))
                    logger.info("✅ STT tokenizer loaded")
            except Exception as e:
                logger.warning(f"Tokenizer loading failed: {e}")
            
            # Initialize actual STT model
            try:
                # Use transformers with proper model type
                from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
                
                # Load with proper configuration
                self.processor = AutoProcessor.from_pretrained(
                    str(stt_path),
                    trust_remote_code=True
                )
                
                self.stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    str(stt_path),
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto" if self.device.type == "cuda" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                self.stt_model.eval()
                logger.info("✅ Real STT model loaded successfully")
                self.is_initialized = True
                return True
                
            except Exception as e:
                logger.error(f"STT model loading failed: {e}")
                # Try Whisper as high-quality fallback
                return await self._initialize_whisper_fallback()
                
        except Exception as e:
            logger.error(f"STT initialization error: {e}")
            return False
    
    async def _initialize_whisper_fallback(self):
        """Initialize Whisper as high-quality fallback"""
        try:
            import whisper
            
            def load_whisper():
                return whisper.load_model("base", device=self.device)
            
            loop = asyncio.get_event_loop()
            self.stt_model = await loop.run_in_executor(None, load_whisper)
            
            logger.info("✅ Whisper STT loaded as fallback")
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Whisper fallback failed: {e}")
            return False
    
    async def transcribe(self, audio_data: np.ndarray) -> str:
        """Real transcription using actual models"""
        if not self.is_initialized or len(audio_data) == 0:
            return ""
        
        try:
            # Run in executor to prevent blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._transcribe_sync, audio_data)
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    def _transcribe_sync(self, audio_data: np.ndarray) -> str:
        """Synchronous transcription"""
        try:
            # Preprocess audio
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Use real model if available
            if hasattr(self, 'processor') and self.processor is not None:
                return self._transcribe_with_transformers(audio_data)
            elif hasattr(self.stt_model, 'transcribe'):
                return self._transcribe_with_whisper(audio_data)
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Sync transcription error: {e}")
            return ""
    
    def _transcribe_with_transformers(self, audio_data: np.ndarray) -> str:
        """Transcribe using transformers model"""
        try:
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
                generated_ids = self.stt_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.0
                )
            
            # Decode result
            transcription = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()
            
            return transcription
            
        except Exception as e:
            logger.error(f"Transformers transcription error: {e}")
            return ""
    
    def _transcribe_with_whisper(self, audio_data: np.ndarray) -> str:
        """Transcribe using Whisper"""
        try:
            # Whisper expects audio at 16kHz
            result = self.stt_model.transcribe(audio_data)
            return result["text"].strip()
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return ""
