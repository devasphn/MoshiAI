import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import soundfile as sf
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="MoshiAI Voice Assistant")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_cache = {}
active_connections = {}

class AudioProcessor:
    """Handles audio processing for speech-to-text and text-to-speech"""
    
    def __init__(self):
        self.stt_model = None
        self.stt_processor = None
        self.tts_model = None
        self.tts_processor = None
        self.sample_rate = 16000
        
    async def initialize_models(self):
        """Initialize Kyutai STT and TTS models"""
        try:
            logger.info("Initializing Kyutai STT model...")
            # Load STT model (kyutai/stt-1b-en_fr)
            self.stt_processor = AutoProcessor.from_pretrained("kyutai/stt-1b-en_fr")
            self.stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                "kyutai/stt-1b-en_fr",
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
            ).to(device)
            
            logger.info("Initializing Kyutai TTS model...")
            # Load TTS model (kyutai/tts-1.6b-en_fr)  
            self.tts_processor = AutoProcessor.from_pretrained("kyutai/tts-1.6b-en_fr")
            self.tts_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                "kyutai/tts-1.6b-en_fr",
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
            ).to(device)
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    async def speech_to_text(self, audio_data: np.ndarray) -> str:
        """Convert speech to text using Kyutai STT"""
        try:
            # Ensure audio is in correct format
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Resample if necessary
            if len(audio_data) > 0:
                audio_tensor = torch.tensor(audio_data).float()
                
                # Process with STT model
                inputs = self.stt_processor(
                    audio_tensor, 
                    sampling_rate=self.sample_rate, 
                    return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    outputs = self.stt_model.generate(**inputs, max_length=512)
                    
                transcription = self.stt_processor.decode(outputs[0], skip_special_tokens=True)
                return transcription.strip()
            
            return ""
            
        except Exception as e:
            logger.error(f"STT error: {e}")
            return ""
    
    async def text_to_speech(self, text: str, emotion: str = "neutral") -> np.ndarray:
        """Convert text to speech using Kyutai TTS"""
        try:
            if not text.strip():
                return np.array([])
            
            # Prepare text with emotion context
            emotional_text = self._add_emotion_context(text, emotion)
            
            # Process with TTS model
            inputs = self.tts_processor(
                emotional_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(device)
            
            with torch.no_grad():
                outputs = self.tts_model.generate(
                    **inputs,
                    max_length=8192,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True
                )
                
            # Convert to audio
            audio_output = self.tts_processor.decode(outputs[0], skip_special_tokens=True)
            
            # Generate synthetic audio array (placeholder for actual TTS output)
            duration = len(text) * 0.1  # Approximate duration
            num_samples = int(duration * self.sample_rate)
            audio_array = np.random.normal(0, 0.1, num_samples).astype(np.float32)
            
            return audio_array
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return np.array([])
    
    def _add_emotion_context(self, text: str, emotion: str) -> str:
        """Add emotional context to text"""
        emotion_prefixes = {
            "happy": "[speaking with joy and enthusiasm] ",
            "sad": "[speaking with sadness and melancholy] ",
            "excited": "[speaking with high energy and excitement] ",
            "calm": "[speaking calmly and peacefully] ",
            "angry": "[speaking with intensity and frustration] ",
            "whisper": "[whispering softly] ",
            "dramatic": "[speaking dramatically with emphasis] ",
            "confident": "[speaking with confidence and authority] ",
            "nervous": "[speaking nervously and hesitantly] ",
            "neutral": ""
        }
        
        prefix = emotion_prefixes.get(emotion, "")
        return f"{prefix}{text}"

class ConversationManager:
    """Manages conversation state and responses"""
    
    def __init__(self):
        self.conversations = {}
        self.current_emotion = "neutral"
        
    async def process_message(self, user_input: str, session_id: str) -> str:
        """Process user message and generate response"""
        try:
            # Initialize conversation if new session
            if session_id not in self.conversations:
                self.conversations[session_id] = {
                    "history": [],
                    "context": ""
                }
            
            # Add user message to history
            self.conversations[session_id]["history"].append({
                "role": "user",
                "content": user_input,
                "timestamp": time.time()
            })
            
            # Generate AI response
            response = await self._generate_response(user_input, session_id)
            
            # Add AI response to history
            self.conversations[session_id]["history"].append({
                "role": "assistant", 
                "content": response,
                "timestamp": time.time()
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I'm sorry, I encountered an error processing your message."
    
    async def _generate_response(self, user_input: str, session_id: str) -> str:
        """Generate contextual response"""
        try:
            # Get conversation context
            context = self._get_conversation_context(session_id)
            
            # Generate response based on input and context
            if "hello" in user_input.lower() or "hi" in user_input.lower():
                return "Hello! I'm MoshiAI, your voice assistant. How can I help you today?"
            
            elif "how are you" in user_input.lower():
                return "I'm doing great! Thanks for asking. I'm here and ready to chat with you."
            
            elif "what can you do" in user_input.lower():
                return "I can have conversations with you using voice! I can listen to what you say and respond with natural speech. Try asking me questions or just chat with me!"
            
            elif "goodbye" in user_input.lower() or "bye" in user_input.lower():
                return "Goodbye! It was nice talking with you. Feel free to come back anytime!"
            
            else:
                # Generate contextual response
                response = f"That's interesting! You mentioned: '{user_input}'. I'd love to hear more about that. What would you like to discuss further?"
                
                # Add context-aware elements
                if len(context) > 0:
                    response += " Based on our conversation, I think this connects to what we discussed earlier."
                
                return response
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm having trouble generating a response right now. Could you try again?"
    
    def _get_conversation_context(self, session_id: str) -> str:
        """Get conversation context for response generation"""
        if session_id not in self.conversations:
            return ""
        
        history = self.conversations[session_id]["history"]
        if len(history) < 2:
            return ""
        
        # Get last few messages for context
        recent_messages = history[-4:]
        context = " ".join([msg["content"] for msg in recent_messages])
        return context
    
    def set_emotion(self, emotion: str):
        """Set current emotion for responses"""
        self.current_emotion = emotion
    
    def get_emotion(self) -> str:
        """Get current emotion"""
        return self.current_emotion

# Initialize processors
audio_processor = AudioProcessor()
conversation_manager = ConversationManager()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting MoshiAI Voice Assistant...")
    await audio_processor.initialize_models()
    logger.info("MoshiAI Voice Assistant ready!")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/status")
async def get_status():
    """Get system status"""
    return {
        "status": "running",
        "device": str(device),
        "models_loaded": audio_processor.stt_model is not None,
        "active_connections": len(active_connections)
    }

@app.post("/set_emotion")
async def set_emotion(request: Request):
    """Set emotion for TTS"""
    data = await request.json()
    emotion = data.get("emotion", "neutral")
    conversation_manager.set_emotion(emotion)
    return {"status": "success", "emotion": emotion}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    active_connections[session_id] = websocket
    
    logger.info(f"New WebSocket connection: {session_id}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "audio":
                # Process audio data
                audio_data = np.array(message["audio"], dtype=np.float32)
                
                # Speech to text
                transcription = await audio_processor.speech_to_text(audio_data)
                
                if transcription:
                    # Send transcription to client
                    await websocket.send_text(json.dumps({
                        "type": "transcription",
                        "text": transcription
                    }))
                    
                    # Generate response
                    response_text = await conversation_manager.process_message(
                        transcription, session_id
                    )
                    
                    # Convert response to speech
                    current_emotion = conversation_manager.get_emotion()
                    response_audio = await audio_processor.text_to_speech(
                        response_text, current_emotion
                    )
                    
                    # Send response to client
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "text": response_text,
                        "audio": response_audio.tolist(),
                        "emotion": current_emotion
                    }))
            
            elif message["type"] == "emotion":
                # Update emotion
                emotion = message.get("emotion", "neutral")
                conversation_manager.set_emotion(emotion)
                
                await websocket.send_text(json.dumps({
                    "type": "emotion_updated",
                    "emotion": emotion
                }))
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clean up connection
        if session_id in active_connections:
            del active_connections[session_id]
        logger.info(f"WebSocket connection closed: {session_id}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
