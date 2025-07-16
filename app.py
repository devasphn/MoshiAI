import asyncio
import json
import logging
import os
import torch
import numpy as np
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from huggingface_hub import hf_hub_download
from moshi.models import loaders, LMGen  # Correct imports from moshi.models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="MoshiAI Voice Assistant")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables for models
mimi = None
lm_gen = None

# Emotion and speaking style configurations
EMOTIONS = {
    "happy": "I'm feeling really happy and cheerful today!",
    "sad": "*sighs* I'm feeling a bit down right now...",
    "excited": "Oh wow! I'm so excited about this!",
    "calm": "I'm feeling very peaceful and relaxed...",
    "frustrated": "Ugh, this is quite frustrating...",
    "curious": "Hmm, that's really interesting! Tell me more...",
    "surprised": "Oh my! That's quite surprising!",
    "whispering": "*whispers* Let me tell you a secret...",
    "singing": "♪ La la la, I love to sing! ♪",
    "formal": "I shall address this matter with utmost professionalism.",
    "casual": "Hey there! What's up?",
    "dramatic": "This is... *dramatic pause* ...absolutely incredible!",
    "giggling": "*giggles* That's so funny!",
    "thoughtful": "Hmm... let me think about this carefully...",
    "energetic": "I'm feeling so full of energy right now!"
}

async def initialize_models():
    """Initialize Moshi models using official loaders"""
    global mimi, lm_gen
    
    try:
        logger.info("Loading Moshi models...")
        
        # Download weights from Hugging Face
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
        
        # Load Mimi codec
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(8)  # Optimized for Moshi
        
        # Load Moshi LM model
        moshi_lm = loaders.get_moshi_lm(moshi_weight, device=device)
        lm_gen = LMGen(moshi_lm, temp=0.8, temp_text=0.7)  # Sampling parameters
        
        logger.info("Models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

class VoiceAssistant:
    def __init__(self):
        self.current_emotion = "happy"
        self.conversation_history = []
        self.is_muted = False
        
    def set_emotion(self, emotion: str):
        """Set the current emotion/speaking style"""
        if emotion in EMOTIONS:
            self.current_emotion = emotion
            return True
        return False
    
    def apply_emotion_to_text(self, text: str) -> str:
        """Apply emotional context to text sparingly for natural flow"""
        if np.random.random() < 0.3:  # 30% chance to add emotional tag for better timing
            if self.current_emotion == "whispering":
                return f"*whispers* {text}"
            elif self.current_emotion == "singing":
                return f"♪ {text} ♪"
            elif self.current_emotion == "dramatic":
                return f"*dramatically* {text}"
            elif self.current_emotion == "giggling":
                return f"*giggles* {text}"
            elif self.current_emotion == "sad":
                return f"*sighs* {text}"
            elif self.current_emotion == "excited":
                return f"*excitedly* {text}!"
        return text
    
    async def process_speech(self, audio_data: bytes) -> dict:
        """Process incoming speech and generate response using Moshi API"""
        try:
            # Convert audio bytes to tensor
            audio_tensor = torch.frombuffer(audio_data, dtype=torch.float32).to(mimi.device)
            
            # Encode audio with Mimi
            with torch.no_grad():
                codes = mimi.encode(audio_tensor.unsqueeze(0))
            
            # Generate transcription/response using LMGen
            with torch.no_grad(), lm_gen.streaming(1):
                tokens_out = lm_gen.step(codes)
                if tokens_out is not None:
                    transcription = " ".join(map(str, tokens_out[:, 1].tolist()))  # Simplified token-to-text
            
            # Check for emotion commands
            if transcription.lower().startswith("speak with"):
                emotion_match = None
                for emotion in EMOTIONS:
                    if emotion in transcription.lower():
                        emotion_match = emotion
                        break
                
                if emotion_match:
                    self.set_emotion(emotion_match)
                    response_text = f"Switching to {emotion_match} mode. {EMOTIONS[emotion_match]}"
                else:
                    response_text = "I understand you want me to change my speaking style. Available emotions include: happy, sad, excited, calm, whispering, singing, dramatic, and more!"
            else:
                # Generate response
                response_text = self.apply_emotion_to_text(transcription)  # Apply emotion to generated text
            
            # Decode to audio output
            with torch.no_grad():
                audio_output = mimi.decode(tokens_out[:, 1:])
            
            return {
                "user_text": transcription,
                "ai_text": response_text,
                "audio": audio_output.cpu().numpy().tobytes(),
                "emotion": self.current_emotion
            }
            
        except Exception as e:
            logger.error(f"Error processing speech: {e}")
            return {"error": str(e)}

# Initialize voice assistant
voice_assistant = VoiceAssistant()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    await initialize_models()

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time voice communication"""
    await websocket.accept()
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Process speech
            result = await voice_assistant.process_speech(data)
            
            # Send response
            await websocket.send_text(json.dumps({
                "type": "response",
                "data": result
            }))
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@app.post("/set_emotion")
async def set_emotion(emotion: str):
    """Set current emotion/speaking style"""
    success = voice_assistant.set_emotion(emotion)
    return {"success": success, "current_emotion": voice_assistant.current_emotion}

@app.get("/emotions")
async def get_emotions():
    """Get available emotions"""
    return {"emotions": list(EMOTIONS.keys())}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
