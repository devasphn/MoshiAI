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
from moshi import MimiModel, LMModel, LMGen, loaders

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="MoshiAI Voice Assistant")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables for models
mimi_model = None
lm_model = None
lm_gen = None

# Emotion configurations (unchanged)
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
    """Initialize Moshi models (updated for 0.2.10)"""
    global mimi_model, lm_model, lm_gen
    
    try:
        logger.info("Loading Moshi models...")
        
        # Load Mimi codec
        mimi_model = loaders.get_mimi(device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Load LM model
        lm_model = loaders.get_lm(device="cuda" if torch.cuda.is_available() else "cpu")
        lm_gen = LMGen(lm_model)
        
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
        if emotion in EMOTIONS:
            self.current_emotion = emotion
            return True
        return False
    
    def apply_emotion_to_text(self, text: str) -> str:
        if np.random.random() < 0.3:  # Sparing application for natural timing
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
        try:
            # Convert audio bytes to tensor
            audio_tensor = torch.frombuffer(audio_data, dtype=torch.float32)
            
            # Speech-to-text (using Mimi for encoding)
            with torch.no_grad():
                encoded = mimi_model.encode(audio_tensor.unsqueeze(0))
                transcription = lm_gen.generate(encoded)  # Simplified for 0.2.10
            
            # Emotion handling and response generation (unchanged logic)
            if transcription.lower().startswith("speak with"):
                emotion_match = next((e for e in EMOTIONS if e in transcription.lower()), None)
                if emotion_match:
                    self.set_emotion(emotion_match)
                    response_text = f"Switching to {emotion_match} mode. {EMOTIONS[emotion_match]}"
                else:
                    response_text = "Available emotions: happy, sad, excited, etc."
            else:
                response_text = lm_gen.generate(transcription)
                response_text = self.apply_emotion_to_text(response_text)
            
            # Text-to-speech
            audio_output = mimi_model.decode(lm_gen.step(response_text))
            
            return {
                "user_text": transcription,
                "ai_text": response_text,
                "audio": audio_output.numpy().tobytes(),
                "emotion": self.current_emotion
            }
            
        except Exception as e:
            logger.error(f"Error processing speech: {e}")
            return {"error": str(e)}

# Initialize voice assistant
voice_assistant = VoiceAssistant()

@app.on_event("startup")
async def startup_event():
    await initialize_models()

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            result = await voice_assistant.process_speech(data)
            await websocket.send_text(json.dumps({"type": "response", "data": result}))
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@app.post("/set_emotion")
async def set_emotion(emotion: str):
    success = voice_assistant.set_emotion(emotion)
    return {"success": success, "current_emotion": voice_assistant.current_emotion}

@app.get("/emotions")
async def get_emotions():
    return {"emotions": list(EMOTIONS.keys())}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
