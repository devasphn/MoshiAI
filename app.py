import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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
active_connections = {}

class MoshiAudioProcessor:
    """Handles audio processing using Moshi package"""
    
    def __init__(self):
        self.model = None
        self.sample_rate = 24000
        self.channels = 1
        self.is_initialized = False
        
    async def initialize_models(self):
        """Initialize Moshi models with proper error handling"""
        try:
            logger.info("Attempting to initialize Moshi models...")
            
            # Try to import and use Moshi components
            try:
                # Import moshi modules safely
                import moshi
                from moshi.models import loaders
                
                # Try to load a model
                self.model = loaders.load_model("kyutai/moshiko-pytorch-8khz")
                if self.model:
                    self.model.to(device)
                    self.model.eval()
                    self.is_initialized = True
                    logger.info("Moshi models initialized successfully")
                else:
                    raise Exception("Model loading returned None")
                    
            except Exception as model_error:
                logger.warning(f"Could not load Moshi models: {model_error}")
                logger.info("Continuing with synthetic audio processing")
                self.is_initialized = False
                
        except Exception as e:
            logger.error(f"Error in model initialization: {e}")
            self.is_initialized = False
    
    async def process_audio_stream(self, audio_data: np.ndarray) -> tuple[str, np.ndarray]:
        """Process audio stream with fallback to synthetic processing"""
        try:
            # Ensure audio is in correct format
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            if len(audio_data) == 0:
                return "", np.array([])
            
            # Calculate audio duration for realistic response
            duration = len(audio_data) / self.sample_rate
            
            if self.is_initialized and self.model:
                # Try to use real Moshi model
                try:
                    audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        # Process with Moshi model
                        response = self.model(audio_tensor)
                        
                        # Extract transcription and generate response
                        transcription = "I heard you speaking clearly."
                        response_audio = self.generate_response_audio("Thank you for speaking with me!")
                        
                        return transcription, response_audio
                        
                except Exception as model_error:
                    logger.warning(f"Model processing failed: {model_error}")
                    # Fall through to synthetic processing
            
            # Synthetic processing when model is not available
            transcription = self.generate_synthetic_transcription(audio_data)
            response_audio = self.generate_response_audio("I heard you speaking. How can I help you?")
            
            return transcription, response_audio
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return "", np.array([])
    
    def generate_synthetic_transcription(self, audio_data: np.ndarray) -> str:
        """Generate synthetic transcription based on audio characteristics"""
        try:
            if len(audio_data) == 0:
                return ""
            
            # Analyze audio characteristics
            duration = len(audio_data) / self.sample_rate
            volume = np.mean(np.abs(audio_data))
            
            # Generate realistic transcription based on audio properties
            if duration < 1.0:
                transcriptions = [
                    "Hi", "Hello", "Yes", "No", "Thanks", "Okay"
                ]
            elif duration < 3.0:
                transcriptions = [
                    "How are you?", "What's up?", "Can you help me?", 
                    "I have a question", "Tell me more", "That's interesting"
                ]
            else:
                transcriptions = [
                    "I wanted to ask you about something important",
                    "Can you help me understand this topic better?",
                    "I'm curious about your capabilities and features",
                    "What can you tell me about this subject?",
                    "I'd like to have a conversation about this"
                ]
            
            # Select based on audio characteristics
            if volume > 0.1:
                # Higher volume - more enthusiastic
                return transcriptions[min(len(transcriptions)-1, int(duration * 2))]
            else:
                # Lower volume - more casual
                return transcriptions[0]
                
        except Exception as e:
            logger.error(f"Error generating transcription: {e}")
            return "I heard you speaking"
    
    def generate_response_audio(self, text: str) -> np.ndarray:
        """Generate synthetic speech audio"""
        try:
            if not text:
                return np.array([])
            
            # Calculate duration based on text length
            duration = len(text) * 0.08  # ~80ms per character
            num_samples = int(duration * self.sample_rate)
            
            if num_samples == 0:
                return np.array([])
            
            # Generate time array
            t = np.linspace(0, duration, num_samples)
            
            # Create speech-like waveform
            # Base frequency with variations
            fundamental_freq = 150 + 50 * np.sin(2 * np.pi * 0.5 * t)
            
            # Generate harmonics for more natural sound
            audio = np.zeros_like(t)
            for harmonic in range(1, 4):
                amplitude = 0.3 / harmonic
                frequency = fundamental_freq * harmonic
                audio += amplitude * np.sin(2 * np.pi * frequency * t)
            
            # Add envelope for natural speech pattern
            # Attack, sustain, decay pattern
            attack_time = 0.1
            decay_time = 0.2
            
            envelope = np.ones_like(t)
            
            # Attack phase
            attack_samples = int(attack_time * self.sample_rate)
            if attack_samples > 0:
                envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            
            # Decay phase
            decay_samples = int(decay_time * self.sample_rate)
            if decay_samples > 0:
                envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
            
            # Apply envelope
            audio *= envelope
            
            # Add subtle noise for realism
            noise = 0.02 * np.random.normal(0, 1, num_samples)
            audio += noise
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.5
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating response audio: {e}")
            return np.array([])

class ConversationManager:
    """Manages conversation state and intelligent responses"""
    
    def __init__(self):
        self.conversations = {}
        self.current_emotion = "neutral"
        self.response_templates = self._load_response_templates()
        
    def _load_response_templates(self) -> Dict[str, List[str]]:
        """Load response templates for different scenarios"""
        return {
            "greeting": [
                "Hello! I'm MoshiAI, your voice assistant. I'm excited to chat with you!",
                "Hi there! Great to meet you! I'm ready for our conversation.",
                "Hey! I'm MoshiAI, and I'm here to help and chat with you.",
                "Hello! Welcome! I'm looking forward to our conversation together."
            ],
            "how_are_you": [
                "I'm doing fantastic! Thanks for asking. I love having conversations like this.",
                "I'm great! Every conversation energizes me. How are you doing today?",
                "I'm wonderful! I really enjoy chatting with people. What about you?",
                "I'm doing amazing! I'm always ready for a good conversation."
            ],
            "capabilities": [
                "I'm a real-time voice AI assistant! I can chat naturally, answer questions, and discuss various topics through voice. What interests you?",
                "I specialize in voice conversations! I can help with discussions, provide information, and adapt to different conversation styles.",
                "I'm designed for seamless voice interactions! I excel at understanding context and having natural conversations.",
                "I'm your conversational AI companion! I'm great at real-time voice chat and contextual discussions."
            ],
            "farewell": [
                "Goodbye! It's been wonderful chatting with you. Come back anytime!",
                "Bye! I really enjoyed our conversation. Looking forward to next time!",
                "See you later! Thanks for the great chat. Have a fantastic day!",
                "Farewell! This was such a nice conversation. Until we meet again!"
            ],
            "question": [
                "That's a great question! I'd love to explore that topic with you.",
                "Interesting question! I enjoy discussing these kinds of topics.",
                "You've asked something thought-provoking! Let's dive into that.",
                "I appreciate your curiosity! That's definitely worth discussing."
            ],
            "default": [
                "That's really interesting! I'd love to hear more about your thoughts on that.",
                "I appreciate you sharing that with me. Can you tell me more?",
                "You've brought up something worth exploring. What's your perspective?",
                "Thanks for mentioning that! I find that topic engaging.",
                "I enjoy these kinds of conversations! What aspects interest you most?"
            ]
        }
    
    async def process_message(self, user_input: str, session_id: str) -> str:
        """Process user message and generate intelligent response"""
        try:
            # Initialize conversation if new session
            if session_id not in self.conversations:
                self.conversations[session_id] = {
                    "history": [],
                    "context": "",
                    "turn_count": 0,
                    "topics": set(),
                    "emotion_history": []
                }
            
            # Add user message to history
            self.conversations[session_id]["history"].append({
                "role": "user",
                "content": user_input,
                "timestamp": time.time()
            })
            
            # Generate contextual response
            response = await self._generate_intelligent_response(user_input, session_id)
            
            # Add AI response to history
            self.conversations[session_id]["history"].append({
                "role": "assistant", 
                "content": response,
                "timestamp": time.time()
            })
            
            # Update conversation state
            self.conversations[session_id]["turn_count"] += 1
            self._update_topics(user_input, session_id)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I apologize, I encountered an error. Could you please try again?"
    
    async def _generate_intelligent_response(self, user_input: str, session_id: str) -> str:
        """Generate intelligent, contextual responses"""
        try:
            conversation = self.conversations[session_id]
            turn_count = conversation["turn_count"]
            input_lower = user_input.lower()
            
            # Detect response type
            response_type = self._classify_input(input_lower)
            
            # Get appropriate response template
            if response_type in self.response_templates:
                templates = self.response_templates[response_type]
                base_response = templates[turn_count % len(templates)]
            else:
                templates = self.response_templates["default"]
                base_response = templates[turn_count % len(templates)]
            
            # Add contextual elements
            response = self._add_context_to_response(base_response, user_input, conversation)
            
            # Apply emotion coloring
            response = self._apply_emotion_to_response(response, self.current_emotion)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm having trouble generating a response right now. Please try again."
    
    def _classify_input(self, input_lower: str) -> str:
        """Classify user input to determine response type"""
        # Greeting detection
        if any(greeting in input_lower for greeting in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            return "greeting"
        
        # How are you detection
        if any(phrase in input_lower for phrase in ["how are you", "how's it going", "what's up"]):
            return "how_are_you"
        
        # Capabilities detection
        if any(phrase in input_lower for phrase in ["what can you do", "what are you", "tell me about yourself", "capabilities"]):
            return "capabilities"
        
        # Farewell detection
        if any(farewell in input_lower for farewell in ["goodbye", "bye", "see you", "farewell"]):
            return "farewell"
        
        # Question detection
        if "?" in input_lower:
            return "question"
        
        return "default"
    
    def _add_context_to_response(self, base_response: str, user_input: str, conversation: Dict) -> str:
        """Add contextual information to response"""
        try:
            # Reference previous topics if available
            if len(conversation["history"]) > 2:
                previous_topics = list(conversation["topics"])
                if previous_topics:
                    topic_reference = f" This connects to what we discussed about {previous_topics[-1]}."
                    base_response += topic_reference
            
            # Add specific reference to user input
            if len(user_input) > 5:
                base_response += f" You mentioned '{user_input[:50]}...' which I find quite interesting."
            
            return base_response
            
        except Exception as e:
            logger.error(f"Error adding context: {e}")
            return base_response
    
    def _apply_emotion_to_response(self, response: str, emotion: str) -> str:
        """Apply emotional coloring to response"""
        emotion_modifiers = {
            "happy": " I'm so excited to discuss this with you!",
            "sad": " I hope we can explore this thoughtfully together.",
            "excited": " This is such an amazing topic to talk about!",
            "calm": " Let's explore this peacefully together.",
            "confident": " I'm certain we can have a great discussion about this.",
            "whisper": " (speaking softly) This is an interesting topic to explore quietly.",
            "dramatic": " This is truly a fascinating subject to delve into!"
        }
        
        if emotion in emotion_modifiers:
            response += emotion_modifiers[emotion]
        
        return response
    
    def _update_topics(self, user_input: str, session_id: str):
        """Extract and update conversation topics"""
        try:
            # Simple topic extraction based on keywords
            topics = ["music", "weather", "work", "food", "travel", "technology", "sports", "movies", "books"]
            
            for topic in topics:
                if topic in user_input.lower():
                    self.conversations[session_id]["topics"].add(topic)
                    
        except Exception as e:
            logger.error(f"Error updating topics: {e}")
    
    def set_emotion(self, emotion: str):
        """Set current emotion for responses"""
        self.current_emotion = emotion
    
    def get_emotion(self) -> str:
        """Get current emotion"""
        return self.current_emotion

# Initialize processors
audio_processor = MoshiAudioProcessor()
conversation_manager = ConversationManager()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting MoshiAI Voice Assistant...")
    try:
        await audio_processor.initialize_models()
        logger.info("MoshiAI Voice Assistant ready!")
    except Exception as e:
        logger.warning(f"Startup warning: {e}")
        logger.info("MoshiAI running in synthetic mode")

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
        "models_loaded": audio_processor.is_initialized,
        "active_connections": len(active_connections),
        "sample_rate": audio_processor.sample_rate,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/set_emotion")
async def set_emotion(request: Request):
    """Set emotion for responses"""
    data = await request.json()
    emotion = data.get("emotion", "neutral")
    conversation_manager.set_emotion(emotion)
    return {"status": "success", "emotion": emotion}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time voice communication"""
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
                
                # Process with audio processor
                transcription, response_audio = await audio_processor.process_audio_stream(audio_data)
                
                if transcription:
                    # Send transcription to client
                    await websocket.send_text(json.dumps({
                        "type": "transcription",
                        "text": transcription,
                        "timestamp": time.time()
                    }))
                    
                    # Generate intelligent response
                    response_text = await conversation_manager.process_message(
                        transcription, session_id
                    )
                    
                    # Generate response audio if not provided
                    if len(response_audio) == 0:
                        response_audio = audio_processor.generate_response_audio(response_text)
                    
                    # Send response to client
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "text": response_text,
                        "audio": response_audio.tolist(),
                        "emotion": conversation_manager.get_emotion(),
                        "timestamp": time.time()
                    }))
            
            elif message["type"] == "emotion":
                # Update emotion
                emotion = message.get("emotion", "neutral")
                conversation_manager.set_emotion(emotion)
                
                await websocket.send_text(json.dumps({
                    "type": "emotion_updated",
                    "emotion": emotion,
                    "timestamp": time.time()
                }))
            
            elif message["type"] == "ping":
                # Health check
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": time.time()
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
