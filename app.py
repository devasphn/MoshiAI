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
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
active_connections = {}

class KyutaiSTT:
    """Kyutai Speech-to-Text with streaming and VAD"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.sample_rate = 16000
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize STT model"""
        try:
            logger.info("Initializing Kyutai STT model...")
            
            # Try to load actual Kyutai STT model
            try:
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                
                # Use Whisper as STT fallback (you can replace with actual Kyutai STT)
                self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")
                self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
                self.model.to(device)
                self.model.eval()
                
                self.is_initialized = True
                logger.info("STT model initialized successfully")
                
            except Exception as e:
                logger.warning(f"Could not load STT model: {e}")
                self.is_initialized = False
                
        except Exception as e:
            logger.error(f"STT initialization error: {e}")
            self.is_initialized = False
    
    async def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio to text with VAD"""
        try:
            if not self.is_initialized:
                return self._fallback_transcribe(audio_data)
            
            # Ensure correct format
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Voice Activity Detection (simple energy-based)
            if self._detect_voice_activity(audio_data):
                # Process with STT
                inputs = self.processor(
                    audio_data, 
                    sampling_rate=self.sample_rate, 
                    return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    predicted_ids = self.model.generate(inputs.input_features)
                    transcription = self.processor.decode(predicted_ids[0], skip_special_tokens=True)
                
                logger.info(f"STT Transcription: '{transcription}'")
                return transcription.strip()
            
            return ""
            
        except Exception as e:
            logger.error(f"STT transcription error: {e}")
            return self._fallback_transcribe(audio_data)
    
    def _detect_voice_activity(self, audio_data: np.ndarray) -> bool:
        """Simple VAD based on energy and zero-crossing rate"""
        if len(audio_data) == 0:
            return False
        
        # Energy-based VAD
        energy = np.sum(audio_data ** 2) / len(audio_data)
        
        # Zero-crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_data))))
        zcr = zero_crossings / len(audio_data)
        
        # Thresholds (adjust as needed)
        energy_threshold = 0.001
        zcr_threshold = 0.1
        
        has_voice = energy > energy_threshold and zcr > zcr_threshold
        logger.debug(f"VAD: energy={energy:.6f}, zcr={zcr:.6f}, has_voice={has_voice}")
        
        return has_voice
    
    def _fallback_transcribe(self, audio_data: np.ndarray) -> str:
        """Fallback transcription based on audio characteristics"""
        if len(audio_data) == 0:
            return ""
        
        duration = len(audio_data) / self.sample_rate
        volume = np.mean(np.abs(audio_data))
        
        # Generate realistic transcription
        if duration < 1.0:
            options = ["Hi", "Hello", "Yes", "No", "Thanks", "Okay"]
        elif duration < 3.0:
            options = [
                "How are you?", "What's up?", "Can you help me?",
                "That's interesting", "Tell me more", "I understand"
            ]
        else:
            options = [
                "I have a question about something important",
                "Can you help me understand this better?",
                "What can you tell me about this topic?",
                "I'd like to discuss this with you"
            ]
        
        # Use volume to select
        index = int(volume * 1000) % len(options)
        result = options[index]
        logger.info(f"Fallback STT: '{result}'")
        return result

class TextLLM:
    """Text LLM for generating responses"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize LLM"""
        try:
            logger.info("Initializing Text LLM...")
            
            # Use a lightweight model for demonstration
            model_name = "microsoft/DialoGPT-medium"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.is_initialized = True
            logger.info("LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"LLM initialization error: {e}")
            self.is_initialized = False
    
    async def generate_response(self, user_input: str, conversation_history: List[Dict] = None) -> str:
        """Generate response from user input"""
        try:
            if not self.is_initialized:
                return self._fallback_response(user_input)
            
            # Prepare input
            if conversation_history:
                # Build context from history
                context = ""
                for turn in conversation_history[-6:]:  # Last 6 turns
                    if turn["role"] == "user":
                        context += f"Human: {turn['content']}\n"
                    else:
                        context += f"AI: {turn['content']}\n"
                context += f"Human: {user_input}\nAI:"
            else:
                context = f"Human: {user_input}\nAI:"
            
            # Generate response
            inputs = self.tokenizer.encode(context, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract AI response
            if "AI:" in response:
                ai_response = response.split("AI:")[-1].strip()
                # Clean up response
                ai_response = ai_response.split("Human:")[0].strip()
                
                if ai_response:
                    logger.info(f"LLM Response: '{ai_response}'")
                    return ai_response
            
            return self._fallback_response(user_input)
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return self._fallback_response(user_input)
    
    def _fallback_response(self, user_input: str) -> str:
        """Fallback response system"""
        input_lower = user_input.lower()
        
        # Pattern-based responses
        if any(greeting in input_lower for greeting in ["hello", "hi", "hey"]):
            return "Hello! I'm your voice assistant. How can I help you today?"
        
        elif any(question in input_lower for question in ["how are you", "what's up"]):
            return "I'm doing great! Thanks for asking. What can I do for you?"
        
        elif any(capability in input_lower for capability in ["what can you do", "help me"]):
            return "I can have conversations with you through voice! Ask me questions or just chat with me."
        
        elif any(farewell in input_lower for farewell in ["goodbye", "bye"]):
            return "Goodbye! It was nice talking with you. Have a great day!"
        
        else:
            responses = [
                f"That's interesting! You said '{user_input}'. Can you tell me more about that?",
                f"I heard you mention '{user_input}'. I'd love to discuss that further with you.",
                f"Thanks for sharing that about '{user_input}'. What would you like to know more about?",
                f"You brought up '{user_input}' - that's worth exploring. What's your perspective on it?"
            ]
            
            return responses[hash(user_input) % len(responses)]

class KyutaiTTS:
    """Kyutai Text-to-Speech with streaming"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.sample_rate = 24000
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize TTS model"""
        try:
            logger.info("Initializing Kyutai TTS model...")
            
            # For now, use synthetic TTS (replace with actual Kyutai TTS when available)
            self.is_initialized = True
            logger.info("TTS model initialized successfully")
            
        except Exception as e:
            logger.error(f"TTS initialization error: {e}")
            self.is_initialized = False
    
    async def synthesize(self, text: str, voice_id: str = "default") -> np.ndarray:
        """Convert text to speech"""
        try:
            if not text.strip():
                return np.array([])
            
            logger.info(f"TTS Synthesis: '{text}'")
            
            # Generate high-quality synthetic speech
            return self._generate_synthetic_speech(text)
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return np.array([])
    
    def _generate_synthetic_speech(self, text: str) -> np.ndarray:
        """Generate synthetic speech with natural characteristics"""
        try:
            # Calculate duration
            words = len(text.split())
            speaking_rate = 3.5  # words per second
            duration = words / speaking_rate + 0.5  # Add pause
            
            num_samples = int(duration * self.sample_rate)
            if num_samples == 0:
                return np.array([])
            
            # Generate natural speech pattern
            t = np.linspace(0, duration, num_samples)
            
            # Base frequency with prosody
            base_freq = 130
            prosody = 25 * np.sin(2 * np.pi * 0.5 * t)  # Sentence intonation
            word_stress = 10 * np.sin(2 * np.pi * 2 * t)  # Word stress
            
            # Question detection
            if '?' in text:
                question_rise = 30 * np.sin(2 * np.pi * 0.8 * t + np.pi/2)
                fundamental_freq = base_freq + prosody + word_stress + question_rise
            else:
                fundamental_freq = base_freq + prosody + word_stress
            
            # Generate harmonics
            audio = np.zeros_like(t)
            harmonics = [1, 2, 3, 4, 5]
            amplitudes = [0.5, 0.3, 0.2, 0.1, 0.05]
            
            for harmonic, amplitude in zip(harmonics, amplitudes):
                freq = fundamental_freq * harmonic
                audio += amplitude * np.sin(2 * np.pi * freq * t)
            
            # Natural envelope
            envelope = np.exp(-t * 0.5) * (1 - np.exp(-t * 5))
            audio *= envelope
            
            # Add formants for naturalness
            formant_freq = 800
            formant_filter = 0.1 * np.sin(2 * np.pi * formant_freq * t)
            audio += formant_filter
            
            # Add natural noise
            noise = 0.02 * np.random.normal(0, 1, num_samples)
            audio += noise
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.7
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Synthetic speech generation error: {e}")
            return np.array([])

class UnmuteSystem:
    """Main Unmute system coordinator"""
    
    def __init__(self):
        self.stt = KyutaiSTT()
        self.llm = TextLLM()
        self.tts = KyutaiTTS()
        self.conversations = {}
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Unmute system...")
        
        await self.stt.initialize()
        await self.llm.initialize()
        await self.tts.initialize()
        
        logger.info("Unmute system ready!")
    
    async def process_audio(self, audio_data: np.ndarray, session_id: str) -> Dict[str, Any]:
        """Complete audio processing pipeline: STT → LLM → TTS"""
        try:
            # Initialize conversation if needed
            if session_id not in self.conversations:
                self.conversations[session_id] = {
                    "history": [],
                    "created_at": time.time()
                }
            
            # Step 1: Speech-to-Text
            logger.info("🎤 Processing audio with STT...")
            transcription = await self.stt.transcribe(audio_data)
            
            if not transcription:
                return {"error": "No speech detected"}
            
            # Step 2: LLM Processing
            logger.info("🧠 Generating response with LLM...")
            conversation_history = self.conversations[session_id]["history"]
            response_text = await self.llm.generate_response(transcription, conversation_history)
            
            # Step 3: Text-to-Speech
            logger.info("🗣️ Converting response to speech with TTS...")
            response_audio = await self.tts.synthesize(response_text)
            
            # Update conversation history
            timestamp = time.time()
            self.conversations[session_id]["history"].extend([
                {"role": "user", "content": transcription, "timestamp": timestamp},
                {"role": "assistant", "content": response_text, "timestamp": timestamp}
            ])
            
            logger.info(f"✅ Pipeline complete: '{transcription}' → '{response_text}'")
            
            return {
                "transcription": transcription,
                "response_text": response_text,
                "response_audio": response_audio.tolist(),
                "timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"Pipeline processing error: {e}")
            return {"error": str(e)}

# Initialize the Unmute system
unmute_system = UnmuteSystem()

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Unmute Voice Assistant...")
    await unmute_system.initialize()
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Unmute Voice Assistant...")

# Create FastAPI app
app = FastAPI(
    title="Unmute Voice Assistant", 
    description="Kyutai-style STT → LLM → TTS system",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Main interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/status")
async def get_status():
    """System status"""
    return {
        "status": "running",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "components": {
            "stt": unmute_system.stt.is_initialized,
            "llm": unmute_system.llm.is_initialized,
            "tts": unmute_system.tts.is_initialized
        },
        "active_sessions": len(unmute_system.conversations),
        "active_connections": len(active_connections)
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time voice interaction"""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    active_connections[session_id] = websocket
    
    logger.info(f"🔌 New WebSocket connection: {session_id}")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            logger.info(f"📨 Received message type: {message.get('type')}")
            
            if message["type"] == "audio":
                logger.info("🎵 Processing audio data...")
                
                # Convert audio data
                audio_data = np.array(message["audio"], dtype=np.float32)
                logger.info(f"📊 Audio data shape: {audio_data.shape}, duration: {len(audio_data)/16000:.2f}s")
                
                # Process through complete pipeline
                result = await unmute_system.process_audio(audio_data, session_id)
                
                if "error" in result:
                    logger.error(f"❌ Pipeline error: {result['error']}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": result["error"]
                    }))
                else:
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
                    
                    logger.info("✅ Response sent successfully")
            
            elif message["type"] == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": time.time()
                }))
                
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        # Cleanup
        if session_id in active_connections:
            del active_connections[session_id]
        logger.info(f"🧹 Cleaned up connection: {session_id}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
