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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_dir = Path("./models")

logger.info(f"🚀 Starting MoshiAI Voice Assistant on {device}")

class AudioBuffer:
    """Fixed audio buffer with proper VAD"""
    
    def __init__(self, sample_rate: int = 16000, max_duration: float = 8.0):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)
        self.buffer = []
        self.lock = threading.Lock()
        
    def add_audio(self, audio_data: np.ndarray):
        """Add audio data with size limits"""
        with self.lock:
            self.buffer.extend(audio_data)
            
            # Keep buffer within reasonable size
            if len(self.buffer) > self.max_samples:
                excess = len(self.buffer) - self.max_samples
                self.buffer = self.buffer[excess:]
    
    def get_audio(self) -> np.ndarray:
        """Get current buffer as numpy array"""
        with self.lock:
            if len(self.buffer) == 0:
                return np.array([])
            return np.array(self.buffer, dtype=np.float32)
    
    def get_recent_audio(self, duration: float = 1.0) -> np.ndarray:
        """Get recent audio for VAD"""
        samples = int(self.sample_rate * duration)
        with self.lock:
            if len(self.buffer) < samples:
                return np.array(self.buffer, dtype=np.float32)
            return np.array(self.buffer[-samples:], dtype=np.float32)
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()
    
    def get_length(self) -> int:
        """Get buffer length"""
        with self.lock:
            return len(self.buffer)

class MoshiAISystem:
    """Fixed system orchestrator"""
    
    def __init__(self):
        self.stt_service = KyutaiSTTService(device)
        self.tts_service = KyutaiTTSService(device)
        self.llm_service = MoshiLLMService(device)
        self.is_ready = False
        self.active_sessions = {}
        
    async def initialize(self):
        """Initialize all services"""
        logger.info("🔄 Initializing MoshiAI system...")
        
        results = await asyncio.gather(
            self.stt_service.initialize(models_dir),
            self.tts_service.initialize(models_dir),
            self.llm_service.initialize(models_dir),
            return_exceptions=True
        )
        
        stt_ok = results[0] is True
        tts_ok = results[1] is True
        llm_ok = results[2] is True
        
        self.is_ready = True
        
        logger.info(f"✅ MoshiAI System initialized:")
        logger.info(f"   STT: {'✅' if stt_ok else '⚠️  (fallback)'}")
        logger.info(f"   TTS: {'✅' if tts_ok else '⚠️  (fallback)'}")
        logger.info(f"   LLM: {'✅' if llm_ok else '⚠️  (fallback)'}")
    
    def create_session(self, session_id: str) -> Dict[str, Any]:
        """Create session with fixed logic"""
        session = {
            "id": session_id,
            "audio_buffer": AudioBuffer(),
            "last_activity": time.time(),
            "is_processing": False,
            "last_transcription": "",
            "conversation_history": [],
            "silence_start": None,
            "speech_detected": False,
            "processed_utterances": set()  # Track processed utterances
        }
        self.active_sessions[session_id] = session
        logger.info(f"🎯 Created session: {session_id}")
        return session
    
    def remove_session(self, session_id: str):
        """Remove session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"🗑️  Removed session: {session_id}")
    
    async def process_audio_stream(self, session_id: str, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Fixed audio processing logic"""
        if not self.is_ready or session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        if session["is_processing"]:
            return None
        
        try:
            # Add audio to buffer
            session["audio_buffer"].add_audio(audio_data)
            session["last_activity"] = time.time()
            
            # Get current buffer
            buffered_audio = session["audio_buffer"].get_audio()
            buffer_length = len(buffered_audio)
            
            # Need minimum audio
            min_samples = int(16000 * 0.5)  # 500ms minimum
            if buffer_length < min_samples:
                return None
            
            # Voice activity detection
            has_voice = detect_speech_activity(buffered_audio, threshold=0.002)
            recent_audio = session["audio_buffer"].get_recent_audio(0.8)
            recent_has_voice = detect_speech_activity(recent_audio, threshold=0.002)
            
            # Speech state tracking
            if has_voice and not session["speech_detected"]:
                session["speech_detected"] = True
                session["silence_start"] = None
                logger.debug(f"🎤 Session {session_id}: Speech started")
                
            elif session["speech_detected"] and not recent_has_voice:
                if session["silence_start"] is None:
                    session["silence_start"] = time.time()
                    logger.debug(f"🤫 Session {session_id}: Silence detected")
                
                # Check for end of speech
                silence_duration = time.time() - session["silence_start"]
                if silence_duration >= 0.8:  # 800ms silence = end of speech
                    # Create unique utterance ID based on buffer content
                    utterance_hash = hash(buffered_audio.tobytes())
                    
                    if utterance_hash not in session["processed_utterances"]:
                        logger.info(f"🎯 Session {session_id}: Processing new utterance ({buffer_length} samples)")
                        
                        session["is_processing"] = True
                        session["processed_utterances"].add(utterance_hash)
                        
                        try:
                            result = await self._process_complete_utterance(session, buffered_audio)
                            session["audio_buffer"].clear()
                            session["speech_detected"] = False
                            session["silence_start"] = None
                            return result
                        finally:
                            session["is_processing"] = False
                    else:
                        logger.debug(f"🔄 Session {session_id}: Duplicate utterance detected, skipping")
                        
            elif session["speech_detected"] and recent_has_voice:
                # Reset silence timer if voice returns
                session["silence_start"] = None
            
            # Force processing if buffer gets too large
            max_samples = int(16000 * 10)  # 10 seconds max
            if buffer_length > max_samples:
                logger.info(f"⚠️  Session {session_id}: Force processing large buffer")
                session["is_processing"] = True
                try:
                    result = await self._process_complete_utterance(session, buffered_audio)
                    session["audio_buffer"].clear()
                    session["speech_detected"] = False
                    return result
                finally:
                    session["is_processing"] = False
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Audio processing error in session {session_id}: {e}")
            session["is_processing"] = False
            return None
    
    async def _process_complete_utterance(self, session: Dict[str, Any], audio_data: np.ndarray) -> Dict[str, Any]:
        """Process utterance with real models"""
        start_time = time.time()
        session_id = session["id"]
        
        try:
            # STT with real model
            stt_start = time.time()
            transcription = await self.stt_service.transcribe(audio_data)
            stt_time = time.time() - stt_start
            
            logger.info(f"📝 STT result: '{transcription}' ({stt_time:.2f}s)")
            
            if not transcription or len(transcription.strip()) < 1:
                return {"error": "Could not transcribe audio"}
            
            # Skip if same as last transcription
            if transcription == session["last_transcription"]:
                logger.debug(f"🔄 Duplicate transcription skipped")
                return {"error": "Duplicate transcription"}
            
            session["last_transcription"] = transcription
            
            # LLM generation
            llm_start = time.time()
            response_text = await self.llm_service.generate_response(transcription)
            llm_time = time.time() - llm_start
            
            logger.info(f"🤖 LLM response: '{response_text[:50]}...' ({llm_time:.2f}s)")
            
            # TTS synthesis
            tts_start = time.time()
            response_audio = await self.tts_service.synthesize(response_text)
            tts_time = time.time() - tts_start
            
            logger.info(f"🔊 TTS generated {len(response_audio)} samples ({tts_time:.2f}s)")
            
            total_time = time.time() - start_time
            
            # Update conversation history
            session["conversation_history"].append({
                "user": transcription,
                "assistant": response_text,
                "timestamp": time.time()
            })
            
            logger.info(f"🎯 Pipeline completed: {total_time:.2f}s (STT: {stt_time:.2f}s, LLM: {llm_time:.2f}s, TTS: {tts_time:.2f}s)")
            
            return {
                "transcription": transcription,
                "response_text": response_text,
                "response_audio": response_audio.tolist() if len(response_audio) > 0 else [],
                "timing": {
                    "total": total_time,
                    "stt": stt_time,
                    "llm": llm_time,
                    "tts": tts_time
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Utterance processing error in session {session_id}: {e}")
            return {"error": "Processing failed"}

# Initialize system
moshi_system = MoshiAISystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle"""
    await moshi_system.initialize()
    yield
    logger.info("🛑 MoshiAI shutting down")

# Create FastAPI app
app = FastAPI(
    title="MoshiAI Voice Assistant",
    description="Production-ready voice assistant",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "models_ready": moshi_system.is_ready,
        "active_sessions": len(moshi_system.active_sessions),
        "services": {
            "stt": moshi_system.stt_service.is_initialized,
            "tts": moshi_system.tts_service.is_initialized,
            "llm": moshi_system.llm_service.is_initialized
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket with fixed logic"""
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    session = moshi_system.create_session(session_id)
    
    logger.info(f"🔌 New session: {session_id}")
    
    try:
        await websocket.send_json({
            "type": "connection",
            "session_id": session_id,
            "status": "connected"
        })
        
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                
                if data.get("type") == "audio":
                    audio_array = np.array(data.get("audio", []), dtype=np.float32)
                    
                    if len(audio_array) > 0:
                        result = await moshi_system.process_audio_stream(session_id, audio_array)
                        if result:
                            await websocket.send_json(result)
                
                elif data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                
                elif data.get("type") == "reset":
                    session["audio_buffer"].clear()
                    session["last_transcription"] = ""
                    session["speech_detected"] = False
                    session["processed_utterances"].clear()
                    await websocket.send_json({"type": "reset_complete"})
                    
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "keepalive"})
                continue
                
    except WebSocketDisconnect:
        logger.info(f"🔌 Session {session_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error in session {session_id}: {e}")
    finally:
        moshi_system.remove_session(session_id)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    if "RUNPOD_POD_ID" in os.environ:
        pod_id = os.environ["RUNPOD_POD_ID"]
        logger.info(f"🌐 RunPod URL: https://{pod_id}-{port}.proxy.runpod.net")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
