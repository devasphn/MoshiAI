import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from huggingface_hub import snapshot_download
import warnings

# --- Global Configuration ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 Using device: {device}")
models_base_dir = Path("./models")

# --- Service Classes ---

class KyutaiSTTService:
    """Kyutai STT Service"""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_initialized = False

    async def initialize(self):
        try:
            logger.info("Loading Kyutai STT Model...")
            from moshi.models import loaders
            import sentencepiece as spm

            model_dir = models_base_dir / "stt" / "models--kyutai--stt-1b-en_fr"
            snapshot_path = next(model_dir.glob("**/snapshots/*/"), None)
            if not snapshot_path: raise FileNotFoundError(f"STT snapshot not found in {model_dir}")

            logger.info(f"Loading STT from: {snapshot_path}")
            model_file = snapshot_path / "mimi-pytorch-e351c8d8@125.safetensors"
            tokenizer_file = snapshot_path / "tokenizer_en_fr_audio_8000.model"

            if not model_file.exists() or not tokenizer_file.exists(): raise FileNotFoundError("STT model/tokenizer missing.")

            self.model = loaders.get_mimi(str(model_file), device=device)
            self.model.eval()

            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(str(tokenizer_file))
            self.is_initialized = True
            logger.info("✅ Kyutai STT Model loaded successfully!")
        except Exception as e:
            logger.error(f"STT initialization failed: {e}", exc_info=True)
            self.is_initialized = False

    async def transcribe(self, audio_data: np.ndarray) -> str:
        if not self.is_initialized or len(audio_data) == 0: return ""
        try:
            audio_tensor = torch.from_numpy(audio_data).to(device).unsqueeze(0)
            transcribed_ids = self.model.generate(audio_tensor, self.tokenizer.eos_id())[0].cpu().numpy()
            transcription = self.tokenizer.decode(transcribed_ids.tolist())
            logger.info(f"STT Result: '{transcription}'")
            return transcription
        except Exception as e:
            logger.error(f"STT transcription error: {e}", exc_info=True)
            return "I heard you, but there was an error."

class KyutaiTTSService:
    """Kyutai TTS Service with Manual Model Loading"""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_initialized = False

    async def initialize(self):
        try:
            logger.info("Loading Kyutai TTS Model...")
            # CRITICAL FIX: Manually import the model architecture and load the state dict.
            from moshi.models.dsm_tts import DSMTTS
            from moshi.models.config import DSMTTSConfig
            from safetensors.torch import load_file
            import sentencepiece as spm

            model_dir = models_base_dir / "tts" / "models--kyutai--tts-1.6b-en_fr"
            snapshot_path = next(model_dir.glob("**/snapshots/*/"), None)
            if not snapshot_path: raise FileNotFoundError(f"TTS snapshot not found in {model_dir}")

            logger.info(f"Loading TTS from: {snapshot_path}")
            model_file = snapshot_path / "dsm_tts_1e68beda@240.safetensors"
            tokenizer_file = snapshot_path / "tokenizer_spm_8k_en_fr_audio.model"

            if not model_file.exists() or not tokenizer_file.exists(): raise FileNotFoundError("TTS model/tokenizer missing.")

            # 1. Instantiate the model from its config.
            config = DSMTTSConfig()
            self.model = DSMTTS(config).to(device)

            # 2. Load the weights from the .safetensors file.
            state_dict = load_file(model_file, device=str(device))
            
            # 3. Load the state dict into the model architecture. strict=False is key.
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()

            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(str(tokenizer_file))
            
            self.is_initialized = True
            logger.info("✅ Kyutai TTS Model loaded successfully!")
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}", exc_info=True)
            self.is_initialized = False

    async def synthesize(self, text: str) -> np.ndarray:
        if not text.strip(): return np.array([])
        if not self.is_initialized: return self._generate_fallback_voice(text)
        try:
            logger.info(f"Synthesizing: '{text}'")
            tokenized_text = torch.tensor([self.tokenizer.encode(text)], device=device)
            with torch.no_grad():
                output_dict = self.model.generate(tokenized_text)
                audio_output = output_dict['audio'][0].cpu().numpy()
            return audio_output
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}", exc_info=True)
            return self._generate_fallback_voice("I have an error in my speech module.")

    def _generate_fallback_voice(self, text: str) -> np.ndarray:
        sr = 24000
        duration = max(0.5, len(text) * 0.1)
        t = np.linspace(0., duration, int(sr * duration), endpoint=False)
        return (np.sin(2 * np.pi * 220.0 * t) * 0.3).astype(np.float32)

class MoshiLLMService:
    """Moshi LLM Service"""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_initialized = False

    async def initialize(self):
        try:
            logger.info("Loading Moshi LLM...")
            from moshi.models import loaders
            import sentencepiece as spm
            
            model_dir = models_base_dir / "llm" / "models--kyutai--moshika-pytorch-bf16"
            snapshot_path = next(model_dir.glob("**/snapshots/*/"), None)
            if not snapshot_path: raise FileNotFoundError(f"LLM snapshot not found in {model_dir}")

            logger.info(f"Loading LLM from: {snapshot_path}")
            model_file = snapshot_path / "model.safetensors"
            tokenizer_file = snapshot_path / "tokenizer_spm_32k_3.model"
            if not model_file.exists() or not tokenizer_file.exists(): raise FileNotFoundError("LLM model/tokenizer missing.")

            self.model = loaders.get_moshi_lm(str(model_file), device=device)
            self.model.eval()

            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(str(tokenizer_file))
            self.is_initialized = True
            logger.info("✅ Moshi LLM loaded successfully!")
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}", exc_info=True)
            self.is_initialized = False

    async def generate_response(self, text: str) -> str:
        if not text.strip(): return "Could you please repeat that?"
        if not self.is_initialized: return "The LLM is in fallback mode."
        try:
            prompt = f"User: {text}\nAssistant:"
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids, max_new_tokens=60, pad_token_id=self.tokenizer.eos_id()
                )
            response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            logger.info(f"LLM Response: '{response}'")
            return response
        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
            return "I am having trouble thinking right now."

# --- Main System Orchestrator ---
class UnmuteSystem:
    def __init__(self):
        self.stt_service = KyutaiSTTService()
        self.tts_service = KyutaiTTSService()
        self.llm_service = MoshiLLMService()

    async def initialize(self):
        await asyncio.gather(
            self.stt_service.initialize(),
            self.tts_service.initialize(),
            self.llm_service.initialize()
        )
        logger.info("✅ All Unmute.sh Services Initialized!")

    async def process_audio(self, audio_data: np.ndarray) -> Dict[str, Any]:
        try:
            transcription = await self.stt_service.transcribe(audio_data)
            if not transcription.strip(): return {"error": "No speech detected"}

            response_text = await self.llm_service.generate_response(transcription)
            response_audio = await self.tts_service.synthesize(response_text)
            
            return {
                "transcription": transcription,
                "response_text": response_text,
                "response_audio": response_audio.tolist(),
            }
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return {"error": "An internal error occurred."}

# --- FastAPI Application ---
unmute_system = UnmuteSystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await unmute_system.initialize()
    yield
    logger.info("🛑 Shutting down.")

app = FastAPI(title="Unmute Voice Assistant", version="5.0.0-final", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request): return templates.TemplateResponse("index.html", {"request": request})

@app.get("/status")
async def get_status():
    return {
        "stt_initialized": unmute_system.stt_service.is_initialized,
        "tts_initialized": unmute_system.tts_service.is_initialized,
        "llm_initialized": unmute_system.llm_service.is_initialized,
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"🔌 New WebSocket connection: {session_id}")
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "audio":
                audio_data = np.array(data.get("audio", []), dtype=np.float32)
                if audio_data.size == 0: continue
                result = await unmute_system.process_audio(audio_data)
                await websocket.send_json(result)
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}", exc_info=True)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
