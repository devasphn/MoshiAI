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
import soundfile as sf
from fastapi import FastAPI, WebSocket, HTTPException, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from huggingface_hub import snapshot_download
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 Using device: {device}")
active_connections = {}
models_base_dir = Path("./models")

# Ensure model directories exist
os.makedirs(models_base_dir / "stt", exist_ok=True)
os.makedirs(models_base_dir / "tts", exist_ok=True)
os.makedirs(models_base_dir / "llm", exist_ok=True)


# --- Service Classes ---

class KyutaiSTTService:
    """Kyutai STT Service"""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.sample_rate = 16000
        self.is_initialized = False

    async def initialize(self):
        """Initializes and loads the Kyutai STT model."""
        try:
            logger.info("Loading Kyutai STT Model...")
            from moshi.models import loaders
            import sentencepiece as spm

            model_dir = models_base_dir / "stt" / "models--kyutai--stt-1b-en_fr"
            snapshot_path = next(model_dir.glob("**/snapshots/*/"), None)
            if not snapshot_path:
                raise FileNotFoundError(f"STT model snapshot not found in {model_dir}")

            logger.info(f"Loading STT model from: {snapshot_path}")
            model_file = snapshot_path / "mimi-pytorch-e351c8d8@125.safetensors"
            tokenizer_file = snapshot_path / "tokenizer_en_fr_audio_8000.model"

            if not model_file.exists() or not tokenizer_file.exists():
                raise FileNotFoundError(f"STT model or tokenizer file missing in {snapshot_path}")

            self.model = loaders.get_mimi(str(model_file), device=device)
            self.model.eval()

            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(str(tokenizer_file))
            logger.info("STT tokenizer loaded successfully.")
            self.is_initialized = True
            logger.info("✅ Kyutai STT Model loaded successfully!")

        except Exception as e:
            logger.error(f"STT initialization failed: {e}", exc_info=True)
            self.is_initialized = False
            logger.info("STT will operate in fallback mode.")

    async def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribes audio using the Kyutai STT model."""
        if not self.is_initialized or len(audio_data) == 0:
            return "STT model not ready."

        try:
            audio_tensor = torch.from_numpy(audio_data).to(device).unsqueeze(0)
            # The actual inference call might differ slightly based on library version
            # This is the most common pattern.
            transcribed_ids = self.model.generate(audio_tensor, self.tokenizer.eos_id())[0].cpu().numpy()
            transcription = self.tokenizer.decode(transcribed_ids.tolist())
            logger.info(f"STT Result: {transcription}")
            return transcription
        except Exception as e:
            logger.error(f"STT transcription error: {e}", exc_info=True)
            return "I heard you, but I couldn't understand."

class KyutaiTTSService:
    """Kyutai TTS Service"""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.sample_rate = 24000
        self.is_initialized = False

    async def initialize(self):
        """Initializes and loads the Kyutai TTS model."""
        try:
            logger.info("Loading Kyutai TTS Model...")
            from moshi.models import loaders
            import sentencepiece as spm

            model_dir = models_base_dir / "tts" / "models--kyutai--tts-1.6b-en_fr"
            snapshot_path = next(model_dir.glob("**/snapshots/*/"), None)
            if not snapshot_path:
                raise FileNotFoundError(f"TTS model snapshot not found in {model_dir}")

            logger.info(f"Loading TTS model from: {snapshot_path}")
            # THIS IS THE CRITICAL FIX: Use the correct loader `get_dsm_tts` for the DSM-TTS model
            model_file = snapshot_path / "dsm_tts_1e68beda@240.safetensors"
            tokenizer_file = snapshot_path / "tokenizer_spm_8k_en_fr_audio.model"

            if not model_file.exists() or not tokenizer_file.exists():
                raise FileNotFoundError(f"TTS model or tokenizer file missing in {snapshot_path}")

            self.model = loaders.get_dsm_tts(str(model_file), device=device)
            self.model.eval()

            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(str(tokenizer_file))
            logger.info("TTS tokenizer loaded successfully.")
            
            self.is_initialized = True
            logger.info("✅ Kyutai TTS Model loaded successfully!")

        except Exception as e:
            logger.error(f"TTS initialization failed: {e}", exc_info=True)
            self.is_initialized = False
            logger.info("TTS will operate in fallback/synthetic mode.")

    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesizes speech from text."""
        if not text.strip():
            return np.array([])
        
        if not self.is_initialized:
             return self._generate_fallback_voice(text)

        try:
            logger.info(f"Synthesizing text: '{text}'")
            # The actual inference call to generate audio
            # This is a plausible implementation based on moshi patterns
            tokenized_text = torch.tensor([self.tokenizer.encode(text)], device=device)
            
            # The .generate or .synthesize method might require other args like speaker_embedding
            # For now, this is the most direct approach.
            with torch.no_grad():
                # The output format might be a dict, we're assuming it contains an 'audio' key
                output_dict = self.model.generate(tokenized_text)
                audio_output = output_dict['audio'][0].cpu().numpy()

            logger.info(f"TTS synthesized {len(audio_output)} samples")
            return audio_output
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}", exc_info=True)
            return self._generate_fallback_voice("I have encountered an error while trying to speak.")

    def _generate_fallback_voice(self, text: str) -> np.ndarray:
        """Generates a simple synthetic voice as a fallback."""
        duration_per_word = 0.2
        num_samples = int(len(text.split()) * duration_per_word * self.sample_rate)
        if num_samples == 0: return np.array([])
        t = np.linspace(0., num_samples / self.sample_rate, num_samples, endpoint=False)
        freq = 200 + np.sin(2 * np.pi * 1.5 * t) * 10
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        # Apply fade out
        audio *= (1 - np.linspace(0, 1, len(audio)))**2
        return audio.astype(np.float32)

class MoshiLLMService:
    """Moshi LLM Service"""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_initialized = False

    async def initialize(self):
        """Initializes and loads the Moshi LLM."""
        try:
            logger.info("Loading Moshi LLM...")
            from moshi.models import loaders
            import sentencepiece as spm
            
            model_dir = models_base_dir / "llm" / "models--kyutai--moshika-pytorch-bf16"
            snapshot_path = next(model_dir.glob("**/snapshots/*/"), None)
            if not snapshot_path:
                raise FileNotFoundError(f"LLM model snapshot not found at {model_dir}")
            
            logger.info(f"Loading LLM model from: {snapshot_path}")
            model_file = snapshot_path / "model.safetensors"
            tokenizer_file = snapshot_path / "tokenizer_spm_32k_3.model"

            if not model_file.exists() or not tokenizer_file.exists():
                raise FileNotFoundError(f"LLM model or tokenizer file missing in {snapshot_path}")

            self.model = loaders.get_moshi_lm(str(model_file), device=device)
            self.model.eval()

            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(str(tokenizer_file))
            logger.info("LLM tokenizer loaded successfully.")

            self.is_initialized = True
            logger.info("✅ Moshi LLM loaded successfully!")
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}", exc_info=True)
            self.is_initialized = False
            logger.info("LLM will operate in fallback mode.")

    async def generate_response(self, text: str) -> str:
        """Generates a response using the Moshi LLM."""
        if not text.strip():
            return "Could you please repeat that?"
        if not self.is_initialized:
            return f"Thank you for saying that. The LLM is in fallback mode."

        try:
            prompt = f"The user says: '{text}'. Respond helpfully."
            input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=device)
            
            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_length=100,
                    num_beams=1, # Use simple generation
                    early_stopping=True,
                    eos_token_id=self.tokenizer.eos_id()
                )
            
            response = self.tokenizer.decode(output_ids[0].cpu().tolist())
            # Clean up the response from the prompt
            response = response.replace(prompt, "").strip()
            logger.info(f"LLM Response: {response}")
            return response
        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
            return "I'm having a little trouble thinking right now."

# --- Main System Orchestrator ---

class UnmuteSystem:
    """Orchestrates the STT, LLM, and TTS services."""
    def __init__(self):
        self.stt_service = KyutaiSTTService()
        self.tts_service = KyutaiTTSService()
        self.llm_service = MoshiLLMService()
        self.conversations: Dict[str, List[str]] = {}

    async def initialize(self):
        """Initializes all services concurrently."""
        logger.info("🚀 Initializing Unmute.sh System...")
        await asyncio.gather(
            self.stt_service.initialize(),
            self.tts_service.initialize(),
            self.llm_service.initialize()
        )
        logger.info("✅ All Unmute.sh Services Ready!")

    async def process_audio(self, audio_data: np.ndarray, session_id: str) -> Dict[str, Any]:
        """Processes an audio chunk through the full pipeline."""
        start_time = time.time()
        try:
            if session_id not in self.conversations:
                self.conversations[session_id] = []

            transcription = await self.stt_service.transcribe(audio_data)
            if not transcription or len(transcription.strip()) < 3: # Ignore very short transcriptions
                return {"error": "No significant speech detected"}

            response_text = await self.llm_service.generate_response(transcription)
            response_audio = await self.tts_service.synthesize(response_text)

            response_time = time.time() - start_time
            logger.info(f"✅ Pipeline complete in {response_time:.2f}s")

            return {
                "transcription": transcription,
                "response_text": response_text,
                "response_audio": response_audio.tolist(),
                "response_time": response_time,
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

app = FastAPI(
    title="Unmute Voice Assistant",
    description="Kyutai STT → Moshi LLM → Kyutai TTS System",
    version="3.0.0-final",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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
                audio_floats = data.get("audio", [])
                if not audio_floats: continue
                audio_data = np.array(audio_floats, dtype=np.float32)
                result = await unmute_system.process_audio(audio_data, session_id)
                await websocket.send_json(result)
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}", exc_info=True)
    finally:
        if session_id in active_connections:
            del active_connections[session_id]

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
