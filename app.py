import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from collections import deque
import threading

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import services
from services.stt_service import KyutaiSTTService
from services.tts_service import KyutaiTTSService
from services.llm_service import MoshiLLMService
from utils.audio_utils import detect_speech_activity

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_dir = Path("./models")

class AudioBuffer:
    """Fixed audio buffer without duplication"""
    
    def __init__(self, sample_rate: int = 16000, max_duration: float = 5.0):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)
        self.buffer = deque(maxlen=self.max_samples)
        self.lock = threading.Lock()
        
    def add_audio(self, audio_data: np.ndarray):
        with self.lock:
            self.buffer.extend(audio_data)
    
    def get_and_clear(self) -> np.ndarray:
        """Get audio and immediately clear to prevent duplicates"""
        with self.lock:
            if len(self.buffer) == 0:
                return np.array([])
            audio = np.array(list(self.buffer))
            self.buffer.clear()  # Clear immediately
            return audio
    
    def get_recent(self, duration: float = 0.5) -> np.ndarray:
        samples = int(self.sample_rate * duration)
        with self.lock:
            if len(self.buffer) < samples:
                return np.array(list(self.buffer))
            return np.array(list(self.buffer)[-samples:])

class MoshiAISystem:
    """Fixed system without duplicates"""
    
    def __init__(self):
        self.stt_service = KyutaiSTTService(device)
        self.tts_service = KyutaiTTSService(device)
        self.llm_service = MoshiLLMService(device)
        self.is_ready = False
        self.active_sessions = {}
        
    async def initialize(self):
        """Initialize real Kyutai models"""
        logger.info("🔄 Initializing real Kyutai system...")
        
        try:
            # Initialize in sequence to avoid conflicts
            stt_ok = await self.stt_service.initialize(models_dir)
            tts_ok = await self.tts_service.initialize(models_dir)
            llm_ok = await self.llm_service.initialize(models_dir)
            
            if not (stt_ok and tts_ok and llm_ok):
                raise Exception("Failed to initialize real Kyutai models")
            
            self.is_ready = True
            logger.info("✅ Real Kyutai system ready")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize real Kyutai: {e}")
            raise
    
    def create_session(self, session_id: str) -> Dict[str, Any]:
        session = {
            "id": session_id,
            "audio_buffer": AudioBuffer(),
            "is_processing": False,
            "last_transcription": "",
            "last_process_time": 0,
            "silence_start": None
        }
        self.active_sessions[session_id] = session
        return session
    
    def remove_session(self, session_id: str):
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
    
    async def process_audio_stream(self, session_id: str, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Fixed processing without duplicates"""
        if not self.is_ready or session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        # Prevent concurrent processing
        if session["is_processing"]:
            return None
        
        # Add audio to buffer
        session["audio_buffer"].add_audio(audio_data)
        
        # Check for silence (end of speech)
        recent_audio = session["audio_buffer"].get_recent(0.8)
        has_voice = detect_speech_activity(recent_audio, threshold=0.003)
        
        if not has_voice:
            if session["silence_start"] is None:
                session["silence_start"] = time.time()
            
            # Check silence duration
            silence_duration = time.time() - session["silence_start"]
            if silence_duration >= 0.8:  # End of speech
                # Get and clear buffer to prevent duplicates
                buffered_audio = session["audio_buffer"].get_and_clear()
                
                if len(buffered_audio) > 8000:  # Minimum audio
                    # Prevent processing same audio twice
                    current_time = time.time()
                    if current_time - session["last_process_time"] < 2.0:
                        return None
                    
                    session["is_processing"] = True
                    session["last_process_time"] = current_time
                    session["silence_start"] = None
                    
                    try:
                        result = await self._process_utterance(session, buffered_audio)
                        return result
                    finally:
                        session["is_processing"] = False
        else:
            session["silence_start"] = None
        
        return None
    
    async def _process_utterance(self, session: Dict[str, Any], audio_data: np.ndarray) -> Dict[str, Any]:
        """Process utterance with real models"""
        start_time = time.time()
        
        try:
            logger.info(f"🎯 Processing {len(audio_data)} samples with real Kyutai")
            
            # Real STT
            stt_start = time.time()
            transcription = await self.stt_service.transcribe(audio_data)
            stt_time = time.time() - stt_start
            
            if not transcription or transcription == session["last_transcription"]:
                return {"error": "No valid transcription"}
            
            session["last_transcription"] = transcription
            
            # LLM
            llm_start = time.time()
            response_text = await self.llm_service.generate_response(transcription)
            llm_time = time.time() - llm_start
            
            # Real TTS
            tts_start = time.time()
            response_audio = await self.tts_service.synthesize(response_text)
            tts_time = time.time() - tts_start
            
            total_time = time.time() - start_time
            
            logger.info(f"✅ Pipeline: {total_time:.2f}s (STT: {stt_time:.2f}s, LLM: {llm_time:.2f}s, TTS: {tts_time:.2f}s)")
            
            return {
                "transcription": transcription,
                "response_text": response_text,
                "response_audio": response_audio.tolist(),
                "timing": {
                    "total": total_time,
                    "stt": stt_time,
                    "llm": llm_time,
                    "tts": tts_time
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            return {"error": str(e)}

# Initialize system
moshi_system = MoshiAISystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await moshi_system.initialize()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    session = moshi_system.create_session(session_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "audio":
                audio_array = np.array(data.get("audio", []), dtype=np.float32)
                if len(audio_array) > 0:
                    result = await moshi_system.process_audio_stream(session_id, audio_array)
                    if result:
                        await websocket.send_json(result)
                        
    except WebSocketDisconnect:
        pass
    finally:
        moshi_system.remove_session(session_id)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    if "RUNPOD_POD_ID" in os.environ:
        pod_id = os.environ["RUNPOD_POD_ID"]
        logger.info(f"🌐 RunPod URL: https://{pod_id}-{port}.proxy.runpod.net")
    
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
