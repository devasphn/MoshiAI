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
    """Production-grade audio buffer with semantic VAD support"""
    
    def __init__(self, sample_rate: int = 16000, buffer_duration: float = 4.0):
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_duration)
        self.buffer = deque(maxlen=self.buffer_size)
        self.lock = threading.Lock()
        self.total_samples = 0
        self.energy_history = deque(maxlen=20)  # Track energy levels
        
    def add_audio(self, audio_data: np.ndarray):
        """Add audio data and compute energy"""
        with self.lock:
            self.buffer.extend(audio_data)
            self.total_samples += len(audio_data)
            
            # Compute energy for this chunk
            energy = np.mean(audio_data ** 2) if len(audio_data) > 0 else 0
            self.energy_history.append(energy)
    
    def get_audio(self) -> np.ndarray:
        """Get current audio buffer"""
        with self.lock:
            if len(self.buffer) == 0:
                return np.array([])
            return np.array(list(self.buffer))
    
    def get_recent_audio(self, duration: float = 0.8) -> np.ndarray:
        """Get recent audio for VAD analysis"""
        samples = int(self.sample_rate * duration)
        with self.lock:
            if len(self.buffer) < samples:
                return np.array(list(self.buffer))
            return np.array(list(self.buffer)[-samples:])
    
    def get_energy_trend(self) -> float:
        """Get average energy over recent history"""
        with self.lock:
            if len(self.energy_history) == 0:
                return 0.0
            return np.mean(list(self.energy_history))
    
    def clear(self):
        """Clear buffer and history"""
        with self.lock:
            self.buffer.clear()
            self.energy_history.clear()
            self.total_samples = 0

class MoshiAISystem:
    """Production system orchestrator with semantic VAD"""
    
    def __init__(self):
        self.stt_service = KyutaiSTTService(device)
        self.tts_service = KyutaiTTSService(device)
        self.llm_service = MoshiLLMService(device)
        self.is_ready = False
        self.active_sessions = {}
        
    async def initialize(self):
        """Initialize all services concurrently"""
        logger.info("🔄 Initializing MoshiAI system...")
        
        # Initialize services in parallel
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
        """Create session with semantic VAD state"""
        session = {
            "id": session_id,
            "audio_buffer": AudioBuffer(),
            "last_activity": time.time(),
            "is_processing": False,
            "last_transcription": "",
            "conversation_history": [],
            "silence_start": None,
            "speech_start": None,
            "audio_chunks_received": 0,
            "vad_state": "silence",  # silence, speech, end_detected
            "energy_threshold": 0.003  # Dynamic threshold
        }
        self.active_sessions[session_id] = session
        logger.info(f"🎯 Created session with semantic VAD: {session_id}")
        return session
    
    def remove_session(self, session_id: str):
        """Remove session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"🗑️  Removed session: {session_id}")
    
    async def process_audio_stream(self, session_id: str, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process audio with semantic VAD (like unmute.sh)"""
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
            session["audio_chunks_received"] += 1
            
            # Get buffered audio
            buffered_audio = session["audio_buffer"].get_audio()
            
            # Need minimum audio for processing
            min_samples = int(16000 * 0.4)  # 400ms minimum
            if len(buffered_audio) < min_samples:
                return None
            
            # Semantic VAD state machine
            current_energy = session["audio_buffer"].get_energy_trend()
            has_voice = detect_speech_activity(buffered_audio, threshold=session["energy_threshold"])
            
            # Update dynamic threshold
            if current_energy > 0:
                session["energy_threshold"] = 0.7 * session["energy_threshold"] + 0.3 * current_energy * 0.5
            
            # VAD state transitions
            current_state = session["vad_state"]
            
            if current_state == "silence" and has_voice:
                # Start of speech detected
                session["vad_state"] = "speech"
                session["speech_start"] = time.time()
                session["silence_start"] = None
                logger.debug(f"🎤 Session {session_id}: Speech started")
                
            elif current_state == "speech" and not has_voice:
                # Potential end of speech
                if session["silence_start"] is None:
                    session["silence_start"] = time.time()
                    logger.debug(f"🤫 Session {session_id}: Silence detected")
                
                # Check silence duration for end-of-speech
                silence_duration = time.time() - session["silence_start"]
                speech_duration = time.time() - (session["speech_start"] or time.time())
                
                # Semantic VAD: longer speech gets longer pause tolerance
                if speech_duration < 2.0:
                    required_silence = 0.6  # Short utterances need shorter pause
                elif speech_duration < 4.0:
                    required_silence = 0.8  # Medium utterances
                else:
                    required_silence = 1.0  # Long utterances get more tolerance
                
                if silence_duration >= required_silence:
                    # End of speech confirmed
                    session["vad_state"] = "end_detected"
                    logger.info(f"🎯 Session {session_id}: End of speech detected (speech: {speech_duration:.1f}s, silence: {silence_duration:.1f}s)")
                    
                    session["is_processing"] = True
                    try:
                        result = await self._process_complete_utterance(session, buffered_audio)
                        session["audio_buffer"].clear()
                        session["vad_state"] = "silence"
                        session["speech_start"] = None
                        session["silence_start"] = None
                        return result
                    finally:
                        session["is_processing"] = False
                        
            elif current_state == "speech" and has_voice:
                # Continue speech - reset silence timer
                session["silence_start"] = None
                
            # Force processing if buffer gets too large
            if len(buffered_audio) > 16000 * 12:  # 12 seconds max
                logger.info(f"⚠️  Session {session_id}: Force processing due to buffer size")
                session["is_processing"] = True
                try:
                    result = await self._process_complete_utterance(session, buffered_audio)
                    session["audio_buffer"].clear()
                    session["vad_state"] = "silence"
                    return result
                finally:
                    session["is_processing"] = False
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Audio processing error in session {session_id}: {e}")
            session["is_processing"] = False
            return None
    
    async def _process_complete_utterance(self, session: Dict[str, Any], audio_data: np.ndarray) -> Dict[str, Any]:
        """Process complete utterance through the pipeline"""
        start_time = time.time()
        session_id = session["id"]
        
        try:
            logger.info(f"🎯 Processing utterance for session {session_id} ({len(audio_data)} samples)")
            
            # STT with flush trick (like unmute.sh)
            stt_start = time.time()
            transcription = await self.stt_service.transcribe(audio_data)
            stt_time = time.time() - stt_start
            
            logger.info(f"📝 STT: '{transcription}' ({stt_time:.2f}s)")
            
            if not transcription or len(transcription.strip()) < 1:
                return {"error": "Could not transcribe audio"}
            
            # Avoid duplicate processing
            if transcription == session["last_transcription"]:
                return {"error": "Duplicate transcription"}
            
            session["last_transcription"] = transcription
            
            # LLM generation
            llm_start = time.time()
            response_text = await self.llm_service.generate_response(transcription)
            llm_time = time.time() - llm_start
            
            logger.info(f"🤖 LLM: '{response_text[:50]}...' ({llm_time:.2f}s)")
            
            # TTS synthesis (streaming like unmute.sh)
            tts_start = time.time()
            response_audio = await self.tts_service.synthesize(response_text)
            tts_time = time.time() - tts_start
            
            logger.info(f"🔊 TTS: {len(response_audio)} samples ({tts_time:.2f}s)")
            
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
    """Application lifecycle management"""
    await moshi_system.initialize()
    yield
    logger.info("🛑 MoshiAI shutting down")

# Create FastAPI application
app = FastAPI(
    title="MoshiAI Voice Assistant",
    description="Production-ready voice assistant using Kyutai architecture",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
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
    """Serve main interface"""
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
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    session = moshi_system.create_session(session_id)
    
    logger.info(f"🔌 New session: {session_id}")
    
    try:
        # Send connection confirmation
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
                    session["vad_state"] = "silence"
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
    # Auto-detect port for RunPod
    port = int(os.environ.get("PORT", 8000))
    
    # RunPod URL detection
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
