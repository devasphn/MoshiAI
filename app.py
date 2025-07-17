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
from utils.audio_utils import detect_speech_activity, preprocess_audio

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
    """Real-time audio buffer for streaming processing"""
    
    def __init__(self, sample_rate: int = 16000, buffer_duration: float = 2.0):
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_duration)
        self.buffer = deque(maxlen=self.buffer_size)
        self.lock = threading.Lock()
        
    def add_audio(self, audio_data: np.ndarray):
        """Add audio data to buffer"""
        with self.lock:
            self.buffer.extend(audio_data)
    
    def get_audio(self) -> np.ndarray:
        """Get current audio buffer as numpy array"""
        with self.lock:
            if len(self.buffer) == 0:
                return np.array([])
            return np.array(list(self.buffer))
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()

class MoshiAISystem:
    """Main system orchestrator with streaming support"""
    
    def __init__(self):
        self.stt_service = KyutaiSTTService(device)
        self.tts_service = KyutaiTTSService(device)
        self.llm_service = MoshiLLMService(device)
        self.is_ready = False
        self.active_sessions = {}
        
    async def initialize(self):
        """Initialize all services"""
        logger.info("🔄 Initializing MoshiAI system...")
        
        # Initialize services concurrently
        results = await asyncio.gather(
            self.stt_service.initialize(models_dir),
            self.tts_service.initialize(models_dir),
            self.llm_service.initialize(models_dir),
            return_exceptions=True
        )
        
        # Check results
        stt_ok = results[0] is True
        tts_ok = results[1] is True
        llm_ok = results[2] is True
        
        self.is_ready = True
        
        logger.info(f"✅ MoshiAI System initialized:")
        logger.info(f"   STT: {'✅' if stt_ok else '⚠️  (fallback)'}")
        logger.info(f"   TTS: {'✅' if tts_ok else '⚠️  (fallback)'}")
        logger.info(f"   LLM: {'✅' if llm_ok else '⚠️  (fallback)'}")
    
    def create_session(self, session_id: str) -> Dict[str, Any]:
        """Create a new session with audio buffer"""
        session = {
            "id": session_id,
            "audio_buffer": AudioBuffer(),
            "last_activity": time.time(),
            "is_processing": False,
            "last_transcription": "",
            "conversation_history": []
        }
        self.active_sessions[session_id] = session
        return session
    
    def remove_session(self, session_id: str):
        """Remove a session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
    
    async def process_audio_stream(self, session_id: str, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process streaming audio data"""
        if not self.is_ready or session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        # Prevent concurrent processing
        if session["is_processing"]:
            return None
        
        try:
            # Add audio to buffer
            session["audio_buffer"].add_audio(audio_data)
            session["last_activity"] = time.time()
            
            # Check if we have enough audio and voice activity
            buffered_audio = session["audio_buffer"].get_audio()
            
            if len(buffered_audio) < 8000:  # Need at least 0.5 seconds
                return None
            
            # Voice activity detection
            if not detect_speech_activity(buffered_audio):
                return None
            
            # Check for silence (end of utterance)
            recent_audio = buffered_audio[-8000:]  # Last 0.5 seconds
            if not detect_speech_activity(recent_audio):
                # End of utterance detected, process the audio
                session["is_processing"] = True
                
                try:
                    result = await self._process_complete_utterance(session, buffered_audio)
                    session["audio_buffer"].clear()
                    return result
                finally:
                    session["is_processing"] = False
            
            return None
            
        except Exception as e:
            logger.error(f"Audio stream processing error: {e}")
            session["is_processing"] = False
            return None
    
    async def _process_complete_utterance(self, session: Dict[str, Any], audio_data: np.ndarray) -> Dict[str, Any]:
        """Process a complete utterance"""
        start_time = time.time()
        
        try:
            # STT: Speech to Text
            stt_start = time.time()
            transcription = await self.stt_service.transcribe(audio_data)
            stt_time = time.time() - stt_start
            
            if not transcription or len(transcription.strip()) < 1:
                return {"error": "Could not transcribe audio"}
            
            # Avoid processing duplicate transcriptions
            if transcription == session["last_transcription"]:
                return {"error": "Duplicate transcription"}
            
            session["last_transcription"] = transcription
            
            # LLM: Generate Response
            llm_start = time.time()
            response_text = await self.llm_service.generate_response(transcription)
            llm_time = time.time() - llm_start
            
            # TTS: Text to Speech
            tts_start = time.time()
            response_audio = await self.tts_service.synthesize(response_text)
            tts_time = time.time() - tts_start
            
            total_time = time.time() - start_time
            
            # Update conversation history
            session["conversation_history"].append({
                "user": transcription,
                "assistant": response_text,
                "timestamp": time.time()
            })
            
            logger.info(f"🎯 Pipeline completed in {total_time:.2f}s (STT: {stt_time:.2f}s, LLM: {llm_time:.2f}s, TTS: {tts_time:.2f}s)")
            
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
            logger.error(f"Utterance processing error: {e}")
            return {"error": "Processing failed"}

# Initialize system
moshi_system = MoshiAISystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    await moshi_system.initialize()
    yield
    logger.info("🛑 MoshiAI shutting down")

# Create FastAPI app
app = FastAPI(
    title="MoshiAI Voice Assistant",
    description="Production-ready voice assistant using Kyutai models",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
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
    """WebSocket endpoint for real-time streaming communication"""
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    session = moshi_system.create_session(session_id)
    
    logger.info(f"🔌 New streaming session: {session_id}")
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection",
            "session_id": session_id,
            "status": "connected"
        })
        
        while True:
            try:
                # Set timeout to prevent hanging
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                
                if data.get("type") == "audio":
                    audio_array = np.array(data.get("audio", []), dtype=np.float32)
                    
                    if len(audio_array) > 0:
                        # Process audio stream
                        result = await moshi_system.process_audio_stream(session_id, audio_array)
                        
                        if result:
                            await websocket.send_json(result)
                
                elif data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                
                elif data.get("type") == "reset":
                    session["audio_buffer"].clear()
                    session["last_transcription"] = ""
                    await websocket.send_json({"type": "reset_complete"})
                    
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "keepalive"})
                continue
                
    except WebSocketDisconnect:
        logger.info(f"🔌 Session {session_id} disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error in session {session_id}: {e}")
    finally:
        moshi_system.remove_session(session_id)

if __name__ == "__main__":
    # Auto-detect port for RunPod
    port = int(os.environ.get("PORT", 8000))
    
    # Check if running on RunPod
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
