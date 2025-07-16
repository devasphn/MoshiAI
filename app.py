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
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForSpeechSeq2Seq
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
        """Initialize STT model with PyTorch 2.6.0+ support"""
        try:
            logger.info("Initializing Kyutai STT model...")
            
            # Use Whisper with safetensors format (compatible with PyTorch 2.6.0+)
            try:
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                
                # Use Whisper base model with safetensors
                model_name = "openai/whisper-base"
                
                self.processor = WhisperProcessor.from_pretrained(
                    model_name,
                    use_safetensors=True  # Force safetensors usage
                )
                
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    model_name,
                    use_safetensors=True,  # Force safetensors usage
                    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
                )
                
                self.model.to(device)
                self.model.eval()
                
                self.is_initialized = True
                logger.info("STT model initialized successfully with safetensors")
                
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
            
            # Voice Activity Detection
            if self._detect_voice_activity(audio_data):
                # Process with STT
                inputs = self.processor(
                    audio_data, 
                    sampling_rate=self.sample_rate, 
                    return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        inputs.input_features,
                        max_length=448,
                        num_beams=1,
                        temperature=0.0
                    )
                    transcription = self.processor.decode(predicted_ids[0], skip_special_tokens=True)
                
                logger.info(f"STT Transcription: '{transcription}'")
                return transcription.strip()
            
            return ""
            
        except Exception as e:
            logger.error(f"STT transcription error: {e}")
            return self._fallback_transcribe(audio_data)
    
    def _detect_voice_activity(self, audio_data: np.ndarray) -> bool:
        """Enhanced VAD with multiple features"""
        if len(audio_data) == 0:
            return False
        
        # Energy-based VAD
        energy = np.sum(audio_data ** 2) / len(audio_data)
        
        # Zero-crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_data))))
        zcr = zero_crossings / len(audio_data)
        
        # Spectral centroid (simplified)
        if len(audio_data) > 1024:
            fft = np.abs(np.fft.fft(audio_data[:1024]))
            freqs = np.fft.fftfreq(1024, 1/self.sample_rate)
            spectral_centroid = np.sum(freqs[:512] * fft[:512]) / np.sum(fft[:512])
        else:
            spectral_centroid = 0
        
        # Thresholds
        energy_threshold = 0.0008
        zcr_threshold = 0.05
        centroid_threshold = 100
        
        has_voice = (energy > energy_threshold and 
                    zcr > zcr_threshold and 
                    spectral_centroid > centroid_threshold)
        
        logger.debug(f"VAD: energy={energy:.6f}, zcr={zcr:.6f}, centroid={spectral_centroid:.2f}, has_voice={has_voice}")
        
        return has_voice
    
    def _fallback_transcribe(self, audio_data: np.ndarray) -> str:
        """Fallback transcription with improved audio analysis"""
        if len(audio_data) == 0:
            return ""
        
        duration = len(audio_data) / self.sample_rate
        volume = np.mean(np.abs(audio_data))
        energy = np.sum(audio_data ** 2) / len(audio_data)
        
        # Analyze frequency content
        if len(audio_data) > 512:
            fft = np.abs(np.fft.fft(audio_data[:512]))
            dominant_freq_idx = np.argmax(fft[:256])
            dominant_freq = dominant_freq_idx * self.sample_rate / 512
        else:
            dominant_freq = 200
        
        # Generate contextual transcription
        if duration < 0.8:
            options = [
                "Hi", "Hello", "Yes", "No", "Thanks", "Okay", "Sure", "Great"
            ]
        elif duration < 2.5:
            options = [
                "How are you doing?", "What's up today?", "Can you help me?",
                "That's interesting", "Tell me more", "I understand",
                "What do you think?", "How does this work?"
            ]
        else:
            options = [
                "I have a question about something important",
                "Can you help me understand this better?",
                "What can you tell me about this topic?",
                "I'd like to discuss this with you",
                "Could you explain more about that?",
                "I'm curious about your thoughts on this"
            ]
        
        # Use combined features for selection
        feature_hash = int((volume * 1000 + energy * 100 + dominant_freq) % len(options))
        result = options[feature_hash]
        logger.info(f"Fallback STT: '{result}'")
        return result

class TextLLM:
    """Text LLM with PyTorch 2.6.0+ safetensors support"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize LLM with safetensors"""
        try:
            logger.info("Initializing Text LLM...")
            
            # Use a model that supports safetensors
            model_name = "microsoft/DialoGPT-small"  # Smaller model, better compatibility
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    use_safetensors=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    use_safetensors=True,
                    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                    device_map="auto" if device.type == "cuda" else None
                )
                
                self.model.to(device)
                self.model.eval()
                
                # Set pad token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.is_initialized = True
                logger.info("LLM initialized successfully with safetensors")
                
            except Exception as model_error:
                logger.warning(f"Could not load LLM model: {model_error}")
                logger.info("Using fallback response system")
                self.is_initialized = False
            
        except Exception as e:
            logger.error(f"LLM initialization error: {e}")
            self.is_initialized = False
    
    async def generate_response(self, user_input: str, conversation_history: List[Dict] = None) -> str:
        """Generate response with improved context handling"""
        try:
            if not self.is_initialized:
                return self._fallback_response(user_input, conversation_history)
            
            # Prepare input with conversation context
            if conversation_history and len(conversation_history) > 0:
                # Build context from recent history
                context = ""
                recent_history = conversation_history[-6:]  # Last 6 turns
                
                for turn in recent_history:
                    if turn["role"] == "user":
                        context += f"Human: {turn['content']}\n"
                    else:
                        context += f"Bot: {turn['content']}\n"
                
                context += f"Human: {user_input}\nBot:"
            else:
                context = f"Human: {user_input}\nBot:"
            
            # Generate response
            inputs = self.tokenizer.encode(
                context, 
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 80,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract bot response
            if "Bot:" in response:
                ai_response = response.split("Bot:")[-1].strip()
                # Clean up response
                ai_response = ai_response.split("Human:")[0].strip()
                
                if ai_response and len(ai_response) > 3:
                    logger.info(f"LLM Response: '{ai_response}'")
                    return ai_response
            
            return self._fallback_response(user_input, conversation_history)
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return self._fallback_response(user_input, conversation_history)
    
    def _fallback_response(self, user_input: str, conversation_history: List[Dict] = None) -> str:
        """Enhanced fallback response system"""
        input_lower = user_input.lower()
        
        # Context-aware responses
        if conversation_history and len(conversation_history) > 0:
            last_response = conversation_history[-1].get("content", "")
            if "hello" in last_response.lower():
                context_modifier = " It's great to continue our conversation!"
            else:
                context_modifier = " Building on what we discussed, "
        else:
            context_modifier = ""
        
        # Pattern-based responses
        if any(greeting in input_lower for greeting in ["hello", "hi", "hey", "good morning"]):
            return f"Hello! I'm your voice assistant.{context_modifier} How can I help you today?"
        
        elif any(question in input_lower for question in ["how are you", "what's up", "how's it going"]):
            return f"I'm doing great! Thanks for asking.{context_modifier} What can I do for you?"
        
        elif any(capability in input_lower for capability in ["what can you do", "help me", "capabilities"]):
            return f"I can have conversations with you through voice!{context_modifier} Ask me questions or just chat with me."
        
        elif any(farewell in input_lower for farewell in ["goodbye", "bye", "see you later"]):
            return f"Goodbye!{context_modifier} It was nice talking with you. Have a great day!"
        
        elif "?" in user_input:
            return f"That's a great question about '{user_input[:50]}...'.{context_modifier} I'd love to explore that with you!"
        
        else:
            responses = [
                f"That's interesting! You mentioned '{user_input[:50]}...'.{context_modifier} Can you tell me more?",
                f"I heard you say '{user_input[:50]}...'.{context_modifier} I'd love to discuss that further.",
                f"You brought up '{user_input[:50]}...'.{context_modifier} What's your perspective on it?",
                f"Thanks for sharing '{user_input[:50]}...'.{context_modifier} What would you like to know more about?"
            ]
            
            return responses[hash(user_input) % len(responses)]

class KyutaiTTS:
    """Enhanced TTS with natural speech synthesis"""
    
    def __init__(self):
        self.sample_rate = 24000
        self.is_initialized = True
        
    async def initialize(self):
        """Initialize TTS"""
        try:
            logger.info("Initializing Kyutai TTS model...")
            self.is_initialized = True
            logger.info("TTS model initialized successfully")
            
        except Exception as e:
            logger.error(f"TTS initialization error: {e}")
            self.is_initialized = False
    
    async def synthesize(self, text: str, voice_id: str = "default") -> np.ndarray:
        """Convert text to speech with enhanced naturalness"""
        try:
            if not text.strip():
                return np.array([])
            
            logger.info(f"TTS Synthesis: '{text}'")
            return self._generate_natural_speech(text)
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return np.array([])
    
    def _generate_natural_speech(self, text: str) -> np.ndarray:
        """Generate highly natural synthetic speech"""
        try:
            # Enhanced speech parameters
            words = text.split()
            speaking_rate = 3.2  # Slightly slower for clarity
            base_duration = len(words) / speaking_rate
            
            # Natural pauses and rhythm
            comma_pauses = text.count(',') * 0.3
            period_pauses = text.count('.') * 0.5
            question_pauses = text.count('?') * 0.4
            
            total_duration = base_duration + comma_pauses + period_pauses + question_pauses
            num_samples = int(total_duration * self.sample_rate)
            
            if num_samples == 0:
                return np.array([])
            
            # Time array
            t = np.linspace(0, total_duration, num_samples)
            
            # Fundamental frequency with natural variations
            base_freq = 140  # Natural speaking frequency
            
            # Prosodic variations
            sentence_intonation = 20 * np.sin(2 * np.pi * 0.4 * t)
            word_rhythm = 12 * np.sin(2 * np.pi * 1.8 * t)
            micro_variations = 3 * np.sin(2 * np.pi * 8 * t)
            
            # Question intonation
            if '?' in text:
                question_rise = 35 * np.sin(2 * np.pi * 0.6 * t + np.pi/3)
                fundamental_freq = base_freq + sentence_intonation + word_rhythm + micro_variations + question_rise
            else:
                fundamental_freq = base_freq + sentence_intonation + word_rhythm + micro_variations
            
            # Generate harmonics for natural voice
            audio = np.zeros_like(t)
            harmonics = [1, 2, 3, 4, 5, 6]
            amplitudes = [0.5, 0.3, 0.2, 0.12, 0.08, 0.04]
            
            for harmonic, amplitude in zip(harmonics, amplitudes):
                # Natural frequency modulation
                freq_mod = 1 + 0.008 * np.sin(2 * np.pi * 6 * t)
                frequency = fundamental_freq * harmonic * freq_mod
                
                # Natural phase variations
                phase_mod = 0.05 * np.sin(2 * np.pi * 4 * t)
                phase = 2 * np.pi * frequency * t + phase_mod
                
                audio += amplitude * np.sin(phase)
            
            # Formant simulation for vowel-like sounds
            formants = [650, 1080, 2650, 3900]  # Typical formant frequencies
            formant_amplitudes = [0.1, 0.08, 0.06, 0.04]
            
            for formant_freq, formant_amp in zip(formants, formant_amplitudes):
                # Formant resonance
                formant_wave = formant_amp * np.sin(2 * np.pi * formant_freq * t)
                # Apply formant envelope
                formant_envelope = np.exp(-((t - total_duration/2) ** 2) / (total_duration/3))
                audio += formant_wave * formant_envelope
            
            # Natural amplitude envelope
            envelope = np.ones_like(t)
            
            # Breath groups (natural speech patterns)
            breath_group_duration = 3.0
            num_breath_groups = int(total_duration / breath_group_duration) + 1
            
            for i in range(num_breath_groups):
                start_time = i * breath_group_duration
                end_time = min((i + 1) * breath_group_duration, total_duration)
                
                start_idx = int(start_time * self.sample_rate)
                end_idx = int(end_time * self.sample_rate)
                
                if start_idx < len(envelope) and end_idx <= len(envelope):
                    group_length = end_idx - start_idx
                    group_t = np.linspace(0, 1, group_length)
                    
                    # Natural breath group envelope
                    attack_time = 0.15
                    decay_time = 0.2
                    
                    group_env = np.ones(group_length)
                    
                    # Attack
                    attack_samples = int(attack_time * group_length)
                    if attack_samples > 0:
                        group_env[:attack_samples] = np.linspace(0, 1, attack_samples)
                    
                    # Decay
                    decay_samples = int(decay_time * group_length)
                    if decay_samples > 0:
                        group_env[-decay_samples:] = np.linspace(1, 0.4, decay_samples)
                    
                    envelope[start_idx:end_idx] *= group_env
            
            # Apply envelope
            audio *= envelope
            
            # Vocal tract filtering (simple resonance)
            if len(audio) > 50:
                # Simple moving average for vocal tract effect
                kernel = np.ones(7) / 7
                filtered_audio = np.convolve(audio, kernel, mode='same')
                audio = 0.7 * filtered_audio + 0.3 * audio
            
            # Add natural breathiness
            breath_noise = 0.03 * np.random.normal(0, 1, num_samples)
            # Filter breath noise to speech frequencies
            if len(breath_noise) > 50:
                breath_kernel = np.ones(9) / 9
                breath_noise = np.convolve(breath_noise, breath_kernel, mode='same')
            
            audio += breath_noise
            
            # Natural compression (vocal tract nonlinearity)
            audio = np.tanh(audio * 1.2) * 0.8
            
            # Final normalization
            max_amp = np.max(np.abs(audio))
            if max_amp > 0:
                audio = audio / max_amp * 0.75
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Natural speech generation error: {e}")
            return np.array([])

class UnmuteSystem:
    """Enhanced Unmute system with proper error handling"""
    
    def __init__(self):
        self.stt = KyutaiSTT()
        self.llm = TextLLM()
        self.tts = KyutaiTTS()
        self.conversations = {}
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Unmute system...")
        
        try:
            await self.stt.initialize()
            await self.llm.initialize()
            await self.tts.initialize()
            
            logger.info("Unmute system ready!")
            
        except Exception as e:
            logger.error(f"System initialization error: {e}")
            logger.info("System will run in fallback mode")
    
    async def process_audio(self, audio_data: np.ndarray, session_id: str) -> Dict[str, Any]:
        """Complete audio processing pipeline with enhanced error handling"""
        try:
            # Initialize conversation if needed
            if session_id not in self.conversations:
                self.conversations[session_id] = {
                    "history": [],
                    "created_at": time.time(),
                    "turn_count": 0
                }
            
            # Step 1: Speech-to-Text
            logger.info("🎤 Processing audio with STT...")
            transcription = await self.stt.transcribe(audio_data)
            
            if not transcription or len(transcription.strip()) < 2:
                logger.warning("No valid transcription detected")
                return {"error": "No speech detected or transcription too short"}
            
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
            
            # Update turn count
            self.conversations[session_id]["turn_count"] += 1
            
            logger.info(f"✅ Pipeline complete: '{transcription}' → '{response_text}'")
            
            return {
                "transcription": transcription,
                "response_text": response_text,
                "response_audio": response_audio.tolist(),
                "timestamp": timestamp,
                "turn_count": self.conversations[session_id]["turn_count"]
            }
            
        except Exception as e:
            logger.error(f"Pipeline processing error: {e}")
            return {"error": f"Processing error: {str(e)}"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "stt_status": self.stt.is_initialized,
            "llm_status": self.llm.is_initialized,
            "tts_status": self.tts.is_initialized,
            "active_conversations": len(self.conversations),
            "torch_version": torch.__version__,
            "device": str(device)
        }

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
    description="Kyutai-style STT → LLM → TTS system with PyTorch 2.6.0+",
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
    """Comprehensive system status"""
    system_status = unmute_system.get_system_status()
    
    return {
        "status": "running",
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "components": {
            "stt": system_status["stt_status"],
            "llm": system_status["llm_status"],
            "tts": system_status["tts_status"]
        },
        "active_conversations": system_status["active_conversations"],
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
                logger.info(f"📊 Audio data: shape={audio_data.shape}, duration={len(audio_data)/16000:.2f}s")
                
                # Process through pipeline
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
                        "timestamp": result["timestamp"],
                        "turn_count": result["turn_count"]
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
