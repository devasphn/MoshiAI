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
import soundfile as sf
from transformers import (
    KyutaiSpeechToTextProcessor,
    KyutaiSpeechToTextForConditionalGeneration
)
from moshi.models import loaders, TTSModel

# FastAPI app config
app = FastAPI(title="MoshiAI Voice Assistant")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")
logger.info("Starting MoshiAI Voice Assistant...")

# Globals
device = "cuda" if torch.cuda.is_available() else "cpu"
stt_model = None
stt_processor = None
tts_model = None
emotion = "happy"

# Emotion dictionary
EMOTIONS = {
    "happy": "I'm feeling great today!",
    "sad": "*sighs* It's a down day...",
    "excited": "Oh wow, that sounds amazing!",
    "calm": "Let's take things slowly and peacefully.",
    "frustrated": "Ugh, this is frustrating!",
    "giggling": "*giggles*",
    "dramatic": "*dramatically* Oh no...",
    "whispering": "*whispers* Let me tell you a secret...",
}

# Load Models
async def initialize_models():
    global stt_model, stt_processor, tts_model
    try:
        logger.info("Initializing Kyutai STT model...")
        stt_processor = KyutaiSpeechToTextProcessor.from_pretrained("kyutai/stt-1b-en_fr")
        stt_model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained("kyutai/stt-1b-en_fr").to(device)

        logger.info("Initializing Kyutai TTS model...")
        tts_ckpt = loaders.CheckpointInfo.from_hf_repo("kyutai/tts-1.6b-en_fr")
        tts_model = TTSModel.from_checkpoint_info(tts_ckpt, device=device, dtype=torch.float16)

        logger.info("Models loaded successfully!")

    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

class VoiceAssistant:
    def __init__(self):
        self.current_emotion = "happy"

    def set_emotion(self, new_emotion: str):
        if new_emotion in EMOTIONS:
            self.current_emotion = new_emotion
            logger.info(f"Emotion set to: {new_emotion}")
            return True
        return False

    def apply_emotion_to_text(self, text: str):
        if np.random.random() < 0.35:
            prefix = EMOTIONS.get(self.current_emotion, "")
            return f"{prefix} {text}"
        return text

    async def transcribe(self, audio_bytes: bytes) -> str:
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
        inputs = stt_processor(audio_np, sampling_rate=24000, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = stt_model.generate(**inputs)
            transcription = stt_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logger.info(f"User said: {transcription}")
        return transcription

    def synthesize_speech(self, text: str) -> bytes:
        conditioned_text = self.apply_emotion_to_text(text)
        logger.info(f"AI response: {conditioned_text}")
        audio_tensor = tts_model.synthesize(conditioned_text, temp=0.8)
        return audio_tensor.cpu().numpy().tobytes()

voice_assistant = VoiceAssistant()

@app.on_event("startup")
async def startup_event():
    await initialize_models()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            audio_data = await websocket.receive_bytes()
            transcription = await voice_assistant.transcribe(audio_data)
            response_text = generate_response(transcription)
            audio_out = voice_assistant.synthesize_speech(response_text)
            await websocket.send_text(json.dumps({
                "type": "response",
                "data": {
                    "user_text": transcription,
                    "ai_text": response_text,
                    "audio": audio_out.decode("latin1"),  # Safe transfer of bytes
                    "emotion": voice_assistant.current_emotion
                }
            }))
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@app.post("/set_emotion")
async def set_emotion(payload: dict):
    success = voice_assistant.set_emotion(payload.get("emotion", ""))
    return {"success": success, "emotion": voice_assistant.current_emotion}

@app.get("/emotions")
async def get_emotions():
    return {"emotions": list(EMOTIONS.keys())}

def generate_response(user_input: str) -> str:
    # Here you would normally call your LLM — for now, respond simply for testing
    if "time" in user_input.lower():
        return "It's always a good time to talk!"
    return f"You said: {user_input}"

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
