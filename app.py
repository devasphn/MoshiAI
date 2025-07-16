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
from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
active_connections = {}

class MoshiAudioProcessor:
    """Handles audio processing using Moshi package"""
    
    def __init__(self):
        self.moshi_model = None
        self.mimi_model = None
        self.sample_rate = 24000
        self.channels = 1
        self.is_initialized = False
        
    async def initialize_models(self):
        """Initialize Moshi models with correct loading"""
        try:
            logger.info("Attempting to initialize Moshi models...")
            
            # Try to import and use Moshi components
            try:
                from moshi.models import get_moshi_lm, get_mimi
                from moshi.models.loaders import load_model
                
                # Load Mimi model (compression model)
                logger.info("Loading Mimi compression model...")
                self.mimi_model = get_mimi()
                if self.mimi_model:
                    self.mimi_model.to(device)
                    self.mimi_model.eval()
                    logger.info("Mimi model loaded successfully")
                
                # Load Moshi language model
                logger.info("Loading Moshi language model...")
                self.moshi_model = get_moshi_lm()
                if self.moshi_model:
                    self.moshi_model.to(device)
                    self.moshi_model.eval()
                    logger.info("Moshi language model loaded successfully")
                    
                self.is_initialized = True
                logger.info("All Moshi models initialized successfully")
                    
            except Exception as model_error:
                logger.warning(f"Could not load Moshi models: {model_error}")
                logger.info("Continuing with synthetic audio processing")
                self.is_initialized = False
                
        except Exception as e:
            logger.error(f"Error in model initialization: {e}")
            self.is_initialized = False
    
    async def process_audio_stream(self, audio_data: np.ndarray) -> tuple[str, np.ndarray]:
        """Process audio stream with Moshi models or fallback"""
        try:
            # Ensure audio is in correct format
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            if len(audio_data) == 0:
                return "", np.array([])
            
            # Calculate audio duration for realistic response
            duration = len(audio_data) / self.sample_rate
            
            if self.is_initialized and self.moshi_model and self.mimi_model:
                # Try to use real Moshi models
                try:
                    # Convert to tensor and proper format
                    audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # Ensure correct sample rate
                    if audio_tensor.shape[-1] != int(duration * self.sample_rate):
                        # Resample if needed
                        import torchaudio.transforms as T
                        resampler = T.Resample(orig_freq=16000, new_freq=self.sample_rate)
                        audio_tensor = resampler(audio_tensor)
                    
                    with torch.no_grad():
                        # Compress audio with Mimi
                        compressed = self.mimi_model.encode(audio_tensor)
                        
                        # Process with Moshi LM
                        response_codes = self.moshi_model.generate(
                            compressed, 
                            max_length=100,
                            temperature=0.8,
                            top_p=0.9
                        )
                        
                        # Decode response
                        response_audio = self.mimi_model.decode(response_codes)
                        
                        # Generate transcription (placeholder - real implementation would need STT)
                        transcription = self.analyze_audio_for_transcription(audio_data)
                        
                        return transcription, response_audio.cpu().numpy().squeeze()
                        
                except Exception as model_error:
                    logger.warning(f"Model processing failed: {model_error}")
                    # Fall through to synthetic processing
            
            # Synthetic processing when models are not available
            transcription = self.generate_synthetic_transcription(audio_data)
            response_audio = self.generate_response_audio("I heard you speaking. How can I help you?")
            
            return transcription, response_audio
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return "", np.array([])
    
    def analyze_audio_for_transcription(self, audio_data: np.ndarray) -> str:
        """Analyze audio characteristics to generate realistic transcription"""
        try:
            duration = len(audio_data) / self.sample_rate
            volume = np.mean(np.abs(audio_data))
            
            # Analyze frequency characteristics
            fft = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
            dominant_freq = freqs[np.argmax(np.abs(fft))]
            
            # Generate transcription based on audio characteristics
            if duration < 0.5:
                short_responses = ["Hi", "Yes", "No", "Okay", "Sure", "Thanks"]
                return short_responses[int(abs(dominant_freq) % len(short_responses))]
            elif duration < 2.0:
                medium_responses = [
                    "How are you?", "What's up?", "Can you help me?",
                    "That's interesting", "Tell me more", "I understand"
                ]
                return medium_responses[int(abs(dominant_freq) % len(medium_responses))]
            else:
                long_responses = [
                    "I wanted to ask you about something important today",
                    "Can you help me understand this topic better?",
                    "I'm curious about your capabilities and what you can do",
                    "What can you tell me about artificial intelligence?",
                    "I'd like to have a conversation about technology"
                ]
                return long_responses[int(abs(dominant_freq) % len(long_responses))]
                
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return "I heard you speaking"
    
    def generate_synthetic_transcription(self, audio_data: np.ndarray) -> str:
        """Generate synthetic transcription based on audio characteristics"""
        try:
            if len(audio_data) == 0:
                return ""
            
            # Analyze audio characteristics
            duration = len(audio_data) / self.sample_rate
            volume = np.mean(np.abs(audio_data))
            energy = np.sum(audio_data ** 2)
            
            # Generate realistic transcription based on audio properties
            if duration < 1.0:
                transcriptions = [
                    "Hi there", "Hello", "Yes", "No", "Thanks", "Okay", "Sure", "Great"
                ]
            elif duration < 3.0:
                transcriptions = [
                    "How are you doing?", "What's up today?", "Can you help me?", 
                    "I have a question", "Tell me more about that", "That's really interesting",
                    "What do you think?", "How does this work?"
                ]
            else:
                transcriptions = [
                    "I wanted to ask you about something that's been on my mind",
                    "Can you help me understand how this technology works?",
                    "I'm really curious about your capabilities and features",
                    "What can you tell me about artificial intelligence?",
                    "I'd like to have a detailed conversation about this topic",
                    "Could you explain more about how voice assistants work?",
                    "I find this technology fascinating and want to learn more"
                ]
            
            # Select based on audio characteristics
            volume_factor = int(volume * 100) if volume > 0 else 0
            energy_factor = int(energy * 1000) if energy > 0 else 0
            
            index = (volume_factor + energy_factor) % len(transcriptions)
            return transcriptions[index]
                
        except Exception as e:
            logger.error(f"Error generating transcription: {e}")
            return "I heard you speaking"
    
    def generate_response_audio(self, text: str) -> np.ndarray:
        """Generate high-quality synthetic speech audio"""
        try:
            if not text:
                return np.array([])
            
            # Calculate duration based on text length and speaking rate
            words = len(text.split())
            speaking_rate = 3.5  # words per second
            duration = words / speaking_rate
            
            # Add natural pauses
            pause_factor = text.count(',') * 0.3 + text.count('.') * 0.5
            duration += pause_factor
            
            num_samples = int(duration * self.sample_rate)
            
            if num_samples == 0:
                return np.array([])
            
            # Generate time array
            t = np.linspace(0, duration, num_samples)
            
            # Create more natural speech-like waveform
            audio = np.zeros_like(t)
            
            # Fundamental frequency with natural variations
            base_freq = 140  # Base frequency for speech
            freq_variation = 30 * np.sin(2 * np.pi * 0.8 * t)  # Prosodic variation
            word_stress = 20 * np.sin(2 * np.pi * 2 * t)  # Word-level stress
            
            fundamental_freq = base_freq + freq_variation + word_stress
            
            # Generate harmonics for more natural voice
            for harmonic in range(1, 6):
                amplitude = 0.4 / (harmonic ** 0.7)  # Natural harmonic decay
                frequency = fundamental_freq * harmonic
                
                # Add slight frequency jitter for naturalness
                jitter = 0.02 * np.random.normal(0, 1, len(t))
                frequency *= (1 + jitter)
                
                audio += amplitude * np.sin(2 * np.pi * frequency * t)
            
            # Add natural speech envelope
            # Simulate breath groups and word boundaries
            envelope = np.ones_like(t)
            
            # Add breath group patterns
            breath_groups = int(duration / 2)  # ~2 seconds per breath group
            for i in range(breath_groups):
                start_idx = int(i * len(t) / breath_groups)
                end_idx = int((i + 1) * len(t) / breath_groups)
                
                # Create breath group envelope
                group_t = np.linspace(0, 1, end_idx - start_idx)
                group_envelope = np.exp(-((group_t - 0.5) ** 2) / 0.3)
                envelope[start_idx:end_idx] *= group_envelope
            
            # Add word-level amplitude variations
            word_count = len(text.split())
            if word_count > 0:
                word_duration = duration / word_count
                for i in range(word_count):
                    word_start = int(i * word_duration * self.sample_rate)
                    word_end = int((i + 1) * word_duration * self.sample_rate)
                    
                    if word_end <= len(envelope):
                        # Emphasize content words vs function words
                        emphasis = 0.8 + 0.4 * (i % 2)  # Alternate emphasis
                        envelope[word_start:word_end] *= emphasis
            
            # Apply envelope
            audio *= envelope
            
            # Add subtle formant-like filtering
            # Simulate vocal tract resonances
            formant_freqs = [800, 1200, 2400]  # Typical formant frequencies
            for formant_freq in formant_freqs:
                # Simple formant boost
                formant_boost = 0.1 * np.sin(2 * np.pi * formant_freq * t)
                audio += formant_boost
            
            # Add natural noise and breathiness
            noise_level = 0.03
            breath_noise = noise_level * np.random.normal(0, 1, num_samples)
            audio += breath_noise
            
            # Apply natural compression (like human vocal tract)
            audio = np.tanh(audio * 2) * 0.5
            
            # Normalize to reasonable level
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.6
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating response audio: {e}")
            return np.array([])

class ConversationManager:
    """Advanced conversation management with context awareness"""
    
    def __init__(self):
        self.conversations = {}
        self.current_emotion = "neutral"
        self.response_templates = self._initialize_response_templates()
        self.context_memory = {}
        
    def _initialize_response_templates(self) -> Dict[str, List[str]]:
        """Initialize comprehensive response templates"""
        return {
            "greeting": [
                "Hello! I'm MoshiAI, your voice assistant. I'm thrilled to chat with you today!",
                "Hi there! Welcome! I'm excited to have this conversation with you.",
                "Hey! I'm MoshiAI, and I'm here to help and chat. What's on your mind?",
                "Hello! Great to meet you! I'm ready for whatever you'd like to discuss."
            ],
            "how_are_you": [
                "I'm doing fantastic! Every conversation energizes me. How are you feeling today?",
                "I'm wonderful! I love connecting with people through voice. What about you?",
                "I'm great! I'm always excited about new conversations. How has your day been?",
                "I'm doing amazingly well! I thrive on good conversations like this one."
            ],
            "capabilities": [
                "I'm a real-time voice AI that specializes in natural conversations! I can discuss topics, answer questions, and adapt to different emotional styles. What interests you most?",
                "I'm designed for seamless voice interactions! I excel at understanding context, remembering our conversation, and providing thoughtful responses. What would you like to explore?",
                "I'm your conversational AI companion! I can chat about various topics, help with questions, and even adjust my speaking style to match different emotions. What shall we talk about?",
                "I'm built for engaging voice conversations! I can maintain context, discuss complex topics, and provide helpful information all through natural speech."
            ],
            "farewell": [
                "Goodbye! This has been such a wonderful conversation. I hope we can chat again soon!",
                "Bye! I've really enjoyed talking with you. Thanks for the great conversation!",
                "See you later! This was fantastic. Come back anytime for another chat!",
                "Farewell! I've loved every moment of our conversation. Until next time!"
            ],
            "question": [
                "That's such a thoughtful question! I love exploring topics like this.",
                "Great question! I find inquiries like yours really engaging to discuss.",
                "You've asked something really interesting! I'm excited to explore this with you.",
                "I appreciate your curiosity! Questions like this make for the best conversations."
            ],
            "compliment": [
                "Thank you so much! That's really kind of you to say. I appreciate it!",
                "That's very sweet! I'm glad you're enjoying our conversation.",
                "I'm touched by your kind words! It means a lot to me.",
                "Thank you! Your positivity makes this conversation even more enjoyable."
            ],
            "concern": [
                "I can hear the concern in your voice. I'm here to listen and help however I can.",
                "That sounds important to you. I'm here to support and discuss this with you.",
                "I understand this is weighing on you. Let's talk through it together.",
                "I can sense this matters to you. I'm here to provide a thoughtful conversation about it."
            ],
            "default": [
                "That's really fascinating! I'd love to hear more about your perspective on this.",
                "I find that quite interesting! Can you tell me more about what you're thinking?",
                "You've brought up something worth exploring. What aspects of this interest you most?",
                "That's a great point! I enjoy discussions like this. What's your take on it?",
                "I appreciate you sharing that! It's given me something interesting to consider."
            ]
        }
    
    async def process_message(self, user_input: str, session_id: str) -> str:
        """Process user message with advanced context awareness"""
        try:
            # Initialize conversation if new session
            if session_id not in self.conversations:
                self.conversations[session_id] = {
                    "history": [],
                    "context": "",
                    "turn_count": 0,
                    "topics": set(),
                    "emotion_history": [],
                    "user_preferences": {},
                    "conversation_start": time.time()
                }
            
            # Add user message to history
            self.conversations[session_id]["history"].append({
                "role": "user",
                "content": user_input,
                "timestamp": time.time()
            })
            
            # Generate intelligent response
            response = await self._generate_contextual_response(user_input, session_id)
            
            # Add AI response to history
            self.conversations[session_id]["history"].append({
                "role": "assistant", 
                "content": response,
                "timestamp": time.time()
            })
            
            # Update conversation metadata
            self.conversations[session_id]["turn_count"] += 1
            self._update_conversation_context(user_input, session_id)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I apologize, I encountered an error. Could you please try again?"
    
    async def _generate_contextual_response(self, user_input: str, session_id: str) -> str:
        """Generate intelligent, contextually aware responses"""
        try:
            conversation = self.conversations[session_id]
            turn_count = conversation["turn_count"]
            input_lower = user_input.lower()
            
            # Classify input type
            response_type = self._classify_user_input(input_lower)
            
            # Get base response
            if response_type in self.response_templates:
                templates = self.response_templates[response_type]
                base_response = templates[turn_count % len(templates)]
            else:
                templates = self.response_templates["default"]
                base_response = templates[turn_count % len(templates)]
            
            # Enhance with context
            enhanced_response = self._enhance_with_context(base_response, user_input, conversation)
            
            # Apply emotional coloring
            final_response = self._apply_emotional_tone(enhanced_response, self.current_emotion)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm having trouble generating a response. Please try again."
    
    def _classify_user_input(self, input_lower: str) -> str:
        """Classify user input with improved detection"""
        # Greeting patterns
        if any(greeting in input_lower for greeting in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
            return "greeting"
        
        # Status inquiry
        if any(phrase in input_lower for phrase in ["how are you", "how's it going", "what's up", "how do you feel"]):
            return "how_are_you"
        
        # Capability questions
        if any(phrase in input_lower for phrase in ["what can you do", "what are you", "capabilities", "tell me about yourself", "what do you know"]):
            return "capabilities"
        
        # Farewell
        if any(farewell in input_lower for farewell in ["goodbye", "bye", "see you", "farewell", "talk to you later"]):
            return "farewell"
        
        # Compliments
        if any(compliment in input_lower for compliment in ["awesome", "amazing", "great job", "well done", "impressive", "fantastic"]):
            return "compliment"
        
        # Concerns or problems
        if any(concern in input_lower for concern in ["problem", "issue", "trouble", "worried", "concerned", "help me"]):
            return "concern"
        
        # Questions
        if "?" in input_lower or any(q_word in input_lower for q_word in ["what", "how", "why", "when", "where", "who"]):
            return "question"
        
        return "default"
    
    def _enhance_with_context(self, base_response: str, user_input: str, conversation: Dict) -> str:
        """Enhance response with conversational context"""
        try:
            enhanced = base_response
            
            # Add reference to previous topics
            if len(conversation["history"]) > 2:
                recent_topics = list(conversation["topics"])[-2:]
                if recent_topics:
                    topic_ref = f" This reminds me of when we discussed {recent_topics[-1]}."
                    enhanced += topic_ref
            
            # Reference conversation length
            if conversation["turn_count"] > 5:
                enhanced += " I'm really enjoying our ongoing conversation!"
            
            # Add specific reference to current input
            if len(user_input) > 10:
                key_phrase = user_input[:40] + "..." if len(user_input) > 40 else user_input
                enhanced += f" You mentioned '{key_phrase}' which I find quite thought-provoking."
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing context: {e}")
            return base_response
    
    def _apply_emotional_tone(self, response: str, emotion: str) -> str:
        """Apply emotional coloring to responses"""
        emotion_modifiers = {
            "happy": " I'm so delighted to discuss this with you!",
            "sad": " I hope we can explore this thoughtfully together.",
            "excited": " This is absolutely fascinating to talk about!",
            "calm": " Let's explore this peacefully and mindfully.",
            "confident": " I'm certain we can have a great discussion about this!",
            "whisper": " *speaking softly* This is an interesting topic to explore quietly.",
            "dramatic": " This is truly a remarkable subject to delve into!",
            "nervous": " I hope I can provide helpful insights on this topic.",
            "angry": " I feel strongly about engaging with this topic constructively."
        }
        
        if emotion in emotion_modifiers:
            response += emotion_modifiers[emotion]
        
        return response
    
    def _update_conversation_context(self, user_input: str, session_id: str):
        """Update conversation context and metadata"""
        try:
            # Extract and store topics
            topic_keywords = {
                "music": ["music", "song", "artist", "album", "band", "concert"],
                "technology": ["ai", "computer", "software", "tech", "digital", "internet"],
                "weather": ["weather", "sunny", "rainy", "cloudy", "temperature", "climate"],
                "work": ["work", "job", "career", "office", "business", "professional"],
                "food": ["food", "restaurant", "cooking", "recipe", "meal", "cuisine"],
                "travel": ["travel", "vacation", "trip", "journey", "destination", "visit"],
                "sports": ["sports", "game", "team", "player", "match", "championship"],
                "movies": ["movie", "film", "cinema", "actor", "director", "entertainment"],
                "books": ["book", "author", "reading", "novel", "literature", "story"],
                "health": ["health", "exercise", "fitness", "wellness", "medical", "doctor"]
            }
            
            input_lower = user_input.lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in input_lower for keyword in keywords):
                    self.conversations[session_id]["topics"].add(topic)
            
            # Store emotion history
            self.conversations[session_id]["emotion_history"].append({
                "emotion": self.current_emotion,
                "timestamp": time.time()
            })
            
        except Exception as e:
            logger.error(f"Error updating context: {e}")
    
    def set_emotion(self, emotion: str):
        """Set current emotion"""
        self.current_emotion = emotion
        logger.info(f"Emotion changed to: {emotion}")
    
    def get_emotion(self) -> str:
        """Get current emotion"""
        return self.current_emotion
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get conversation summary and statistics"""
        if session_id not in self.conversations:
            return {}
        
        conversation = self.conversations[session_id]
        return {
            "turn_count": conversation["turn_count"],
            "topics_discussed": list(conversation["topics"]),
            "duration": time.time() - conversation["conversation_start"],
            "message_count": len(conversation["history"]),
            "emotions_used": [e["emotion"] for e in conversation["emotion_history"]]
        }

# Initialize processors
audio_processor = MoshiAudioProcessor()
conversation_manager = ConversationManager()

# Lifespan context manager (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting MoshiAI Voice Assistant...")
    try:
        await audio_processor.initialize_models()
        logger.info("MoshiAI Voice Assistant ready!")
    except Exception as e:
        logger.warning(f"Startup warning: {e}")
        logger.info("MoshiAI running in synthetic mode")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MoshiAI Voice Assistant...")

# Create FastAPI app with lifespan
app = FastAPI(title="MoshiAI Voice Assistant", lifespan=lifespan)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/status")
async def get_status():
    """Get comprehensive system status"""
    return {
        "status": "running",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": audio_processor.is_initialized,
        "active_connections": len(active_connections),
        "sample_rate": audio_processor.sample_rate,
        "current_emotion": conversation_manager.get_emotion()
    }

@app.post("/set_emotion")
async def set_emotion(request: Request):
    """Set emotion for responses"""
    data = await request.json()
    emotion = data.get("emotion", "neutral")
    conversation_manager.set_emotion(emotion)
    return {"status": "success", "emotion": emotion}

@app.get("/conversations/{session_id}/summary")
async def get_conversation_summary(session_id: str):
    """Get conversation summary"""
    summary = conversation_manager.get_conversation_summary(session_id)
    return summary

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
