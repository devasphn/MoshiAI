import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, Any
from contextlib import asynccontextmanager

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

class MoshiAISystem:
    """Main system orchestrator"""
    
    def __init__(self):
        self.stt_service = KyutaiSTTService(device)
        self.tts_service = KyutaiTTSService(device)
        self.llm_service = MoshiLLMService(device)
        self.is_ready = False
        
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
        
        self.is_ready = True  # System is functional even with fallbacks
        
        logger.info(f"✅ MoshiAI System initialized:")
        logger.info(f"   STT: {'✅' if stt_ok else '⚠️  (fallback)'}")
        logger.info(f"   TTS: {'✅' if tts_ok else '⚠️  (fallback)'}")
        logger.info(f"   LLM: {'✅' if llm_ok else '⚠️  (fallback)'}")
        
    async def process_conversation(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Process complete conversation pipeline"""
        if not self.is_ready:
            return {"error": "System not ready"}
        
        try:
            start_time = time.time()
            
            # Voice Activity Detection
            if not detect_speech_activity(audio_data):
                return {"error": "No speech detected"}
            
            # STT: Speech to Text
            stt_start = time.time()
            transcription = await self.stt_service.transcribe(audio_data)
            stt_time = time.time() - stt_start
            
            if not transcription or len(transcription.strip()) < 2:
                return {"error": "Could not transcribe audio"}
            
            # LLM: Generate Response
            llm_start = time.time()
            response_text = await self.llm_service.generate_response(transcription)
            llm_time = time.time() - llm_start
            
            # TTS: Text to Speech
            tts_start = time.time()
            response_audio = await self.tts_service.synthesize(response_text)
            tts_time = time.time() - tts_start
            
            total_time = time.time() - start_time
            
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
            logger.error(f"Pipeline error: {e}")
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
    logger.info(f"🔌 New session: {session_id}")
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_json()
            
            if data.get("type") == "audio":
                audio_array = np.array(data.get("audio", []), dtype=np.float32)
                
                if len(audio_array) > 0:
                    # Process through pipeline
                    result = await moshi_system.process_conversation(audio_array)
                    
                    # Send result back
                    await websocket.send_json(result)
                    
    except WebSocketDisconnect:
        logger.info(f"🔌 Session {session_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error in session {session_id}: {e}")

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
