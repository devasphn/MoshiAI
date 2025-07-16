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

# Import from official Moshi package
from moshi.models import loaders
from moshi.models.compression import MoshiDecoder
from moshi.models.lm import MoshiLM
from moshi import pipeline
import moshi.moshi_server as server

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
    """Handles audio processing using official Moshi models"""
    
    def __init__(self):
        self.model = None
        self.decoder = None
        self.sample_rate = 24000
        self.channels = 1
        self.is_initialized = False
        
    async def initialize_models(self):
        """Initialize Moshi models"""
        try:
            logger.info("Initializing Moshi models...")
            
            # Load the pretrained Moshi model
            self.model = loaders.load_model("kyutai/moshiko-pytorch-8khz")
            self.model.to(device)
            self.model.eval()
            
            # Initialize decoder
            self.decoder = MoshiDecoder(self.model)
            
            self.is_initialized = True
            logger.info("Moshi models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Moshi models: {e}")
            # Fallback initialization
            self.is_initialized = False
            raise
    
    async def process_audio_stream(self, audio_data: np.ndarray) -> tuple[str, np.ndarray]:
        """Process audio stream with Moshi"""
        try:
            if not self.is_initialized:
                await self.initialize_models()
            
            # Ensure audio is in correct format
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Resample to 24kHz if needed
            if len(audio_data) > 0:
                # Convert to tensor
                audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
                
                # Process with Moshi
                with torch.no_grad():
                    # Generate response
                    response = self.model.generate(
                        audio_tensor,
                        max_length=1000,
                        temperature=0.8,
                        top_p=0.9
                    )
                    
                    # Decode audio response
                    audio_output = self.decoder.decode(response)
                    
                    # Extract text (placeholder - actual implementation depends on model output)
                    text_output = "I heard you speaking. How can I help you?"
                    
                    return text_output, audio_output.cpu().numpy()
            
            return "", np.array([])
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return "", np.array([])
    
    def generate_synthetic_response(self, text: str) -> np.ndarray:
        """Generate synthetic audio response"""
        try:
            # Generate synthetic audio (placeholder for actual TTS)
            duration = len(text) * 0.08  # ~80ms per character
            num_samples = int(duration * self.sample_rate)
            
            # Create synthetic speech-like audio
            t = np.linspace(0, duration, num_samples)
            frequency = 200 + 50 * np.sin(2 * np.pi * 0.5 * t)  # Variable frequency
            audio = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            # Add some noise and envelope
            envelope = np.exp(-t * 2) * (1 - np.exp(-t * 10))
            audio = audio * envelope + 0.05 * np.random.normal(0, 1, num_samples)
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating synthetic response: {e}")
            return np.array([])

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
                    "context": "",
                    "turn_count": 0
                }
            
            # Add user message to history
            self.conversations[session_id]["history"].append({
                "role": "user",
                "content": user_input,
                "timestamp": time.time()
            })
            
            # Generate AI response
            response = await self._generate_contextual_response(user_input, session_id)
            
            # Add AI response to history
            self.conversations[session_id]["history"].append({
                "role": "assistant", 
                "content": response,
                "timestamp": time.time()
            })
            
            # Increment turn count
            self.conversations[session_id]["turn_count"] += 1
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I apologize, I encountered an error processing your message."
    
    async def _generate_contextual_response(self, user_input: str, session_id: str) -> str:
        """Generate contextual response based on conversation history"""
        try:
            conversation = self.conversations[session_id]
            turn_count = conversation["turn_count"]
            
            # Convert input to lowercase for matching
            input_lower = user_input.lower()
            
            # Greeting responses
            if any(greeting in input_lower for greeting in ["hello", "hi", "hey", "good morning", "good afternoon"]):
                greetings = [
                    "Hello! I'm MoshiAI, excited to chat with you! What's on your mind?",
                    "Hi there! Great to meet you! I'm ready for our conversation.",
                    "Hey! I'm MoshiAI, your voice assistant. How can I help you today?",
                    "Hello! Welcome to our conversation! I'm here to chat and assist you."
                ]
                return greetings[turn_count % len(greetings)]
            
            # How are you responses
            elif any(phrase in input_lower for phrase in ["how are you", "how's it going", "what's up"]):
                responses = [
                    "I'm doing fantastic! Thanks for asking. I love having conversations like this one.",
                    "I'm great! Every conversation is exciting for me. How are you doing?",
                    "I'm wonderful! I'm always energized by good conversations. What about you?",
                    "I'm doing amazing! I really enjoy our chat. How has your day been?"
                ]
                return responses[turn_count % len(responses)]
            
            # Capability questions
            elif any(phrase in input_lower for phrase in ["what can you do", "what are you capable of", "tell me about yourself"]):
                capabilities = [
                    "I'm a real-time voice AI assistant! I can have natural conversations, answer questions, and chat about various topics using voice. What interests you most?",
                    "I specialize in voice conversations! I can discuss topics, answer questions, and engage in natural dialogue. I'm particularly good at understanding context and emotions.",
                    "I'm designed for seamless voice interactions! I can help with conversations, provide information, and adapt my speaking style to match different emotions and contexts.",
                    "I'm your conversational AI companion! I excel at real-time voice chat, understanding context, and providing helpful responses. What would you like to explore?"
                ]
                return capabilities[turn_count % len(capabilities)]
            
            # Farewell responses
            elif any(farewell in input_lower for farewell in ["goodbye", "bye", "see you", "farewell"]):
                farewells = [
                    "Goodbye! It's been wonderful chatting with you. Come back anytime!",
                    "Bye! I really enjoyed our conversation. Looking forward to talking again!",
                    "See you later! Thanks for the great chat. Have a fantastic day!",
                    "Farewell! This was such a nice conversation. Until next time!"
                ]
                return farewells[turn_count % len(farewells)]
            
            # Question responses
            elif "?" in user_input:
                question_responses = [
                    f"That's a great question about '{user_input}'. I find that topic really interesting! What specifically would you like to know more about?",
                    f"You've asked about '{user_input}' - I'd love to explore that with you! Can you tell me more about what sparked your curiosity?",
                    f"Interesting question! '{user_input}' is something I enjoy discussing. What aspects of this topic are most important to you?",
                    f"I appreciate your question about '{user_input}'. That's definitely worth talking about! What's your perspective on this?"
                ]
                return question_responses[turn_count % len(question_responses)]
            
            # Topic-based responses
            elif any(topic in input_lower for topic in ["music", "song", "artist", "album"]):
                return f"Music is such a wonderful topic! You mentioned '{user_input}'. I'd love to hear about your musical preferences. What genres or artists do you enjoy most?"
            
            elif any(topic in input_lower for topic in ["weather", "temperature", "sunny", "rainy", "cloudy"]):
                return f"Weather can really affect our mood! You brought up '{user_input}'. I find it fascinating how weather influences our daily lives. How does weather typically affect you?"
            
            elif any(topic in input_lower for topic in ["work", "job", "career", "profession"]):
                return f"Work and career are such important parts of life! You mentioned '{user_input}'. I'm curious about your professional interests. What aspects of work do you find most fulfilling?"
            
            elif any(topic in input_lower for topic in ["food", "cooking", "recipe", "restaurant"]):
                return f"Food is one of life's great pleasures! You talked about '{user_input}'. I love hearing about culinary experiences. What are some of your favorite dishes or cuisines?"
            
            elif any(topic in input_lower for topic in ["travel", "vacation", "trip", "journey"]):
                return f"Travel opens up so many possibilities! You mentioned '{user_input}'. I find travel stories fascinating. What destinations have caught your interest lately?"
            
            # Emotional responses
            elif any(emotion in input_lower for emotion in ["happy", "excited", "joy", "great", "wonderful"]):
                return f"I can hear the positivity in what you shared: '{user_input}'! Your enthusiasm is contagious. What's making you feel so good today?"
            
            elif any(emotion in input_lower for emotion in ["sad", "upset", "disappointed", "frustrated"]):
                return f"I can sense you're going through something with '{user_input}'. I'm here to listen and support you. Would you like to talk more about what's bothering you?"
            
            # Technology/AI responses
            elif any(tech in input_lower for tech in ["ai", "artificial intelligence", "technology", "computer", "robot"]):
                return f"Technology is evolving so rapidly! You brought up '{user_input}'. As an AI, I find these discussions particularly engaging. What aspects of technology interest you most?"
            
            # Default contextual responses
            else:
                context_responses = [
                    f"That's really interesting! You said '{user_input}' - I'd love to dive deeper into that. Can you tell me more about your thoughts on this?",
                    f"I appreciate you sharing '{user_input}' with me. That sounds important to you. What made you think about this topic?",
                    f"You've brought up something thought-provoking: '{user_input}'. I'm curious to understand your perspective better. What's the most important aspect of this to you?",
                    f"Thanks for mentioning '{user_input}'. That's definitely worth exploring further. How does this relate to your current interests or experiences?",
                    f"I find what you said about '{user_input}' quite engaging. It's clear you've given this some thought. What would you like to discuss about it next?",
                    f"You've touched on something meaningful with '{user_input}'. I enjoy these kinds of conversations. What aspects of this topic resonate most with you?"
                ]
                
                # Add context from previous conversation
                if len(conversation["history"]) > 2:
                    context_responses.append(f"Building on our conversation, you mentioned '{user_input}'. This connects nicely to what we discussed earlier. How do you see these ideas fitting together?")
                
                return context_responses[turn_count % len(context_responses)]
                
        except Exception as e:
            logger.error(f"Error generating contextual response: {e}")
            return "I'm having trouble generating a response right now. Could you please try again?"
    
    def set_emotion(self, emotion: str):
        """Set current emotion for responses"""
        self.current_emotion = emotion
    
    def get_emotion(self) -> str:
        """Get current emotion"""
        return self.current_emotion
    
    def get_conversation_stats(self, session_id: str) -> Dict[str, Any]:
        """Get conversation statistics"""
        if session_id not in self.conversations:
            return {"turn_count": 0, "message_count": 0}
        
        conversation = self.conversations[session_id]
        return {
            "turn_count": conversation["turn_count"],
            "message_count": len(conversation["history"]),
            "last_activity": conversation["history"][-1]["timestamp"] if conversation["history"] else None
        }

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
        logger.warning(f"Could not initialize Moshi models: {e}")
        logger.info("MoshiAI will run in fallback mode")

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
        "sample_rate": audio_processor.sample_rate
    }

@app.post("/set_emotion")
async def set_emotion(request: Request):
    """Set emotion for TTS"""
    data = await request.json()
    emotion = data.get("emotion", "neutral")
    conversation_manager.set_emotion(emotion)
    return {"status": "success", "emotion": emotion}

@app.get("/conversations/{session_id}/stats")
async def get_conversation_stats(session_id: str):
    """Get conversation statistics"""
    stats = conversation_manager.get_conversation_stats(session_id)
    return stats

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
                
                if audio_processor.is_initialized:
                    # Use Moshi for processing
                    transcription, response_audio = await audio_processor.process_audio_stream(audio_data)
                else:
                    # Fallback processing
                    transcription = "I heard you speaking."
                    response_audio = np.array([])
                
                if transcription:
                    # Send transcription to client
                    await websocket.send_text(json.dumps({
                        "type": "transcription",
                        "text": transcription
                    }))
                    
                    # Generate response text
                    response_text = await conversation_manager.process_message(
                        transcription, session_id
                    )
                    
                    # Generate response audio if not provided by Moshi
                    if len(response_audio) == 0:
                        response_audio = audio_processor.generate_synthetic_response(response_text)
                    
                    # Send response to client
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "text": response_text,
                        "audio": response_audio.tolist(),
                        "emotion": conversation_manager.get_emotion()
                    }))
            
            elif message["type"] == "emotion":
                # Update emotion
                emotion = message.get("emotion", "neutral")
                conversation_manager.set_emotion(emotion)
                
                await websocket.send_text(json.dumps({
                    "type": "emotion_updated",
                    "emotion": emotion
                }))
            
            elif message["type"] == "ping":
                # Respond to ping
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
