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
    
    def __init__(self, sample_rate: int = 16000, buffer_duration: float = 3.0):
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_duration)
        self.buffer = deque(maxlen=self.buffer_size)
        self.lock = threading.Lock()
        self.total_samples = 0
        
    def add_audio(self, audio_data: np.ndarray):
        """Add audio data to buffer"""
        with self.lock:
            self.buffer.extend(audio_data)
            self.total_samples += len(audio_data)
    
    def get_audio(self) -> np.ndarray:
        """Get current audio buffer as numpy array"""
        with self.lock:
            if len(self.buffer) == 0:
                return np.array([])
            return np.array(list(self.buffer))
    
    def get_recent_audio(self, duration: float = 0.5) -> np.ndarray:
        """Get recent audio data"""
        samples = int(self.sample_rate * duration)
        with self.lock:
            if len(self.buffer) < samples:
                return np.array(list(self.buffer))
            return np.array(list(self.buffer)[-samples:])
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()
            self.total_samples = 0

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
            "conversation_history": [],
            "silence_start": None,
            "audio_chunks_received": 0
        }
        self.active_sessions[session_id] = session
        logger.info(f"🎯 Created new session: {session_id}")
        return session
    
    def remove_session(self, session_id: str):
        """Remove a session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"🗑️  Removed session: {session_id}")
    
    async def process_audio_stream(self, session_id: str, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process streaming audio data with corrected logic"""
        if not self.is_ready or session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        # Log audio reception
        session["audio_chunks_received"] += 1
        logger.debug(f"📥 Session {session_id}: Received audio chunk {session['audio_chunks_received']} with {len(audio_data)} samples")
        
        # Prevent concurrent processing
        if session["is_processing"]:
            logger.debug(f"⏳ Session {session_id}: Already processing, skipping chunk")
            return None
        
        try:
            # Add audio to buffer
            session["audio_buffer"].add_audio(audio_data)
            session["last_activity"] = time.time()
            
            # Get buffered audio
            buffered_audio = session["audio_buffer"].get_audio()
            
            # Need minimum audio length (0.3 seconds)
            min_samples = int(16000 * 0.3)
            if len(buffered_audio) < min_samples:
                logger.debug(f"🔊 Session {session_id}: Buffer too small ({len(buffered_audio)} < {min_samples})")
                return None
            
            # Check for voice activity in the buffer
            has_voice_activity = detect_speech_activity(buffered_audio)
            
            if not has_voice_activity:
                logger.debug(f"🔇 Session {session_id}: No voice activity detected")
                return None
            
            # CORRECTED LOGIC: Check for end of utterance
            recent_audio = session["audio_buffer"].get_recent_audio(0.8)  # Last 0.8 seconds
            recent_has_voice = detect_speech_activity(recent_audio)
            
            # If recent audio has no voice activity, consider it end of utterance
            if not recent_has_voice:
                if session["silence_start"] is None:
                    session["silence_start"] = time.time()
                    logger.debug(f"🤫 Session {session_id}: Silence detected, starting timer")
                    return None
                
                # If silence persists for 0.5 seconds, process utterance
                silence_duration = time.time() - session["silence_start"]
                if silence_duration >= 0.5:
                    logger.info(f"🎤 Session {session_id}: End of utterance detected, processing...")
                    session["is_processing"] = True
                    session["silence_start"] = None
                    
                    try:
                        result = await self._process_complete_utterance(session, buffered_audio)
                        session["audio_buffer"].clear()
                        return result
                    finally:
                        session["is_processing"] = False
                        
            else:
                # Reset silence timer if voice activity returns
                if session["silence_start"] is not None:
                    logger.debug(f"🔊 Session {session_id}: Voice activity resumed, resetting silence timer")
                    session["silence_start"] = None
            
            # If buffer is getting too large, force processing
            if len(buffered_audio) > 16000 * 10:  # 10 seconds max
                logger.info(f"⚠️  Session {session_id}: Buffer full, force processing")
                session["is_processing"] = True
                
                try:
                    result = await self._process_complete_utterance(session, buffered_audio)
                    session["audio_buffer"].clear()
                    return result
                finally:
                    session["is_processing"] = False
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Audio stream processing error in session {session_id}: {e}")
            session["is_processing"] = False
            return None
    
    async def _process_complete_utterance(self, session: Dict[str, Any], audio_data: np.ndarray) -> Dict[str, Any]:
        """Process a complete utterance"""
        start_time = time.time()
        session_id = session["id"]
        
        try:
            logger.info(f"🎯 Processing utterance for session {session_id} with {len(audio_data)} samples")
            
            # STT: Speech to Text
            stt_start = time.time()
            transcription = await self.stt_service.transcribe(audio_data)
            stt_time = time.time() - stt_start
            
            logger.info(f"📝 STT result for session {session_id}: '{transcription}' (took {stt_time:.2f}s)")
            
            if not transcription or len(transcription.strip()) < 1:
                return {"error": "Could not transcribe audio"}
            
            # Avoid processing duplicate transcriptions
            if transcription == session["last_transcription"]:
                logger.debug(f"🔄 Duplicate transcription detected for session {session_id}")
                return {"error": "Duplicate transcription"}
            
            session["last_transcription"] = transcription
            
            # LLM: Generate Response
            llm_start = time.time()
            response_text = await self.llm_service.generate_response(transcription)
            llm_time = time.time() - llm_start
            
            logger.info(f"🤖 LLM response for session {session_id}: '{response_text[:50]}...' (took {llm_time:.2f}s)")
            
            # TTS: Text to Speech
            tts_start = time.time()
            response_audio = await self.tts_service.synthesize(response_text)
            tts_time = time.time() - tts_start
            
            logger.info(f"🔊 TTS generated {len(response_audio)} audio samples (took {tts_time:.2f}s)")
            
            total_time = time.time() - start_time
            
            # Update conversation history
            session["conversation_history"].append({
                "user": transcription,
                "assistant": response_text,
                "timestamp": time.time()
            })
            
            logger.info(f"🎯 Pipeline completed for session {session_id} in {total_time:.2f}s (STT: {stt_time:.2f}s, LLM: {llm_time:.2f}s, TTS: {tts_time:.2f}s)")
            
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
                    session["silence_start"] = None
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
