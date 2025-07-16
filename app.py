import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, WebSocket, HTTPException, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from huggingface_hub import hf_hub_download, snapshot_download
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
active_connections = {}

class KyutaiSTTService:
    """Official Kyutai STT Service"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.sample_rate = 16000
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize actual Kyutai STT model"""
        try:
            logger.info("Loading Official Kyutai STT Model...")
            
            # Download STT model from Hugging Face
            try:
                from moshi.models import get_stt_model
                
                # Download the actual Kyutai STT model
                model_path = snapshot_download(
                    repo_id="kyutai/stt-1b-en_fr",
                    cache_dir="./models/stt"
                )
                
                logger.info(f"STT model downloaded to: {model_path}")
                
                # Load the model
                self.model = get_stt_model(model_path)
                self.model.to(device)
                self.model.eval()
                
                self.is_initialized = True
                logger.info("✅ Official Kyutai STT Model loaded successfully!")
                
            except Exception as e:
                logger.error(f"Failed to load Kyutai STT: {e}")
                raise
                
        except Exception as e:
            logger.error(f"STT initialization failed: {e}")
            raise
    
    async def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe using official Kyutai STT"""
        if not self.is_initialized:
            raise RuntimeError("STT model not initialized")
        
        try:
            # Process with official Kyutai STT
            with torch.no_grad():
                audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0).to(device)
                transcription = self.model.transcribe(audio_tensor)
                
            logger.info(f"STT Result: {transcription}")
            return transcription
            
        except Exception as e:
            logger.error(f"STT transcription error: {e}")
            raise

class KyutaiTTSService:
    """Official Kyutai TTS Service"""
    
    def __init__(self):
        self.model = None
        self.sample_rate = 24000
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize actual Kyutai TTS model"""
        try:
            logger.info("Loading Official Kyutai TTS Model...")
            
            try:
                from moshi.models import get_tts_model
                
                # Download the actual Kyutai TTS model
                model_path = snapshot_download(
                    repo_id="kyutai/tts-1.6b-en_fr",
                    cache_dir="./models/tts"
                )
                
                logger.info(f"TTS model downloaded to: {model_path}")
                
                # Load the model
                self.model = get_tts_model(model_path)
                self.model.to(device)
                self.model.eval()
                
                self.is_initialized = True
                logger.info("✅ Official Kyutai TTS Model loaded successfully!")
                
            except Exception as e:
                logger.error(f"Failed to load Kyutai TTS: {e}")
                raise
                
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            raise
    
    async def synthesize(self, text: str, voice_id: str = "default") -> np.ndarray:
        """Synthesize using official Kyutai TTS"""
        if not self.is_initialized:
            raise RuntimeError("TTS model not initialized")
        
        try:
            # Process with official Kyutai TTS
            with torch.no_grad():
                audio_output = self.model.synthesize(text, voice_id=voice_id)
                
            logger.info(f"TTS synthesized: {len(audio_output)} samples")
            return audio_output
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            raise

class MoshiLLMService:
    """Official Moshi LLM Service"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize actual Moshi LLM"""
        try:
            logger.info("Loading Official Moshi LLM...")
            
            try:
                from moshi.models import get_moshi_lm
                
                # Download the actual Moshi LLM
                model_path = snapshot_download(
                    repo_id="kyutai/moshika-pytorch-bf16",
                    cache_dir="./models/llm"
                )
                
                logger.info(f"LLM model downloaded to: {model_path}")
                
                # Load the model
                self.model = get_moshi_lm(model_path)
                self.model.to(device)
                self.model.eval()
                
                self.is_initialized = True
                logger.info("✅ Official Moshi LLM loaded successfully!")
                
            except Exception as e:
                logger.error(f"Failed to load Moshi LLM: {e}")
                raise
                
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            raise
    
    async def generate_response(self, text: str, conversation_history: List[Dict] = None) -> str:
        """Generate response using official Moshi LLM"""
        if not self.is_initialized:
            raise RuntimeError("LLM model not initialized")
        
        try:
            # Process with official Moshi LLM
            with torch.no_grad():
                response = self.model.generate(
                    text,
                    max_length=512,
                    temperature=0.8,
                    top_p=0.9
                )
                
            logger.info(f"LLM Response: {response}")
            return response
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise

class OfficialUnmuteSystem:
    """Official Unmute.sh System Implementation"""
    
    def __init__(self):
        self.stt_service = KyutaiSTTService()
        self.tts_service = KyutaiTTSService()
        self.llm_service = MoshiLLMService()
        self.conversations = {}
        
    async def initialize(self):
        """Initialize all official services"""
        logger.info("🚀 Initializing Official Unmute.sh System...")
        
        try:
            # Initialize all services
            await self.stt_service.initialize()
            await self.tts_service.initialize()
            await self.llm_service.initialize()
            
            logger.info("✅ All Official Unmute.sh Services Ready!")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    async def process_audio(self, audio_data: np.ndarray, session_id: str) -> Dict[str, Any]:
        """Process audio through official Unmute.sh pipeline"""
        try:
            # Step 1: Official STT
            logger.info("🎤 Processing with Official Kyutai STT...")
            transcription = await self.stt_service.transcribe(audio_data)
            
            # Step 2: Official LLM
            logger.info("🧠 Processing with Official Moshi LLM...")
            response_text = await self.llm_service.generate_response(transcription)
            
            # Step 3: Official TTS
            logger.info("🗣️ Processing with Official Kyutai TTS...")
            response_audio = await self.tts_service.synthesize(response_text)
            
            timestamp = time.time()
            
            logger.info("✅ Official Unmute.sh Pipeline Complete!")
            
            return {
                "transcription": transcription,
                "response_text": response_text,
                "response_audio": response_audio.tolist(),
                "timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"Official pipeline error: {e}")
            raise

# Initialize official system
unmute_system = OfficialUnmuteSystem()

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Official Unmute.sh Voice Assistant...")
    await unmute_system.initialize()
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Official Unmute.sh Voice Assistant...")

# Create FastAPI app
app = FastAPI(
    title="Official Unmute.sh Voice Assistant",
    description="Official Kyutai STT → Moshi LLM → Kyutai TTS System",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/status")
async def get_status():
    return {
        "status": "running",
        "stt_initialized": unmute_system.stt_service.is_initialized,
        "tts_initialized": unmute_system.tts_service.is_initialized,
        "llm_initialized": unmute_system.llm_service.is_initialized,
        "device": str(device),
        "system": "Official Unmute.sh"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    active_connections[session_id] = websocket
    
    logger.info(f"🔌 New Official WebSocket connection: {session_id}")
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "audio":
                audio_data = np.array(message["audio"], dtype=np.float32)
                
                # Process through official pipeline
                result = await unmute_system.process_audio(audio_data, session_id)
                
                # Send transcription
                await websocket.send_text(json.dumps({
                    "type": "transcription",
                    "text": result["transcription"],
                    "timestamp": result["timestamp"]
                }))
                
                # Send response
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "text": result["response_text"],
                    "audio": result["response_audio"],
                    "timestamp": result["timestamp"]
                }))
                
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        if session_id in active_connections:
            del active_connections[session_id]

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
