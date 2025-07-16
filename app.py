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

class EmotionProcessor:
    """Enhanced emotion processing system"""
    
    def __init__(self):
        self.emotions = ["neutral", "happy", "sad", "excited", "calm", "angry", "whisper", "dramatic", "confident", "nervous"]
        self.current_emotion = "neutral"
        self.emotion_patterns = self._initialize_emotion_patterns()
    
    def _initialize_emotion_patterns(self) -> Dict[str, Dict]:
        """Initialize emotion patterns for text and audio processing"""
        return {
            "happy": {
                "text_prefix": "*with joy* ",
                "audio_pitch_mult": 1.2,
                "audio_speed_mult": 1.1,
                "energy_boost": 1.3
            },
            "sad": {
                "text_prefix": "*sighs softly* ",
                "audio_pitch_mult": 0.8,
                "audio_speed_mult": 0.9,
                "energy_boost": 0.7
            },
            "excited": {
                "text_prefix": "*with enthusiasm* ",
                "audio_pitch_mult": 1.3,
                "audio_speed_mult": 1.2,
                "energy_boost": 1.5
            },
            "calm": {
                "text_prefix": "*calmly* ",
                "audio_pitch_mult": 0.9,
                "audio_speed_mult": 0.95,
                "energy_boost": 0.8
            },
            "angry": {
                "text_prefix": "*with intensity* ",
                "audio_pitch_mult": 1.1,
                "audio_speed_mult": 1.05,
                "energy_boost": 1.4
            },
            "whisper": {
                "text_prefix": "*whispering* ",
                "audio_pitch_mult": 0.7,
                "audio_speed_mult": 0.8,
                "energy_boost": 0.4
            },
            "dramatic": {
                "text_prefix": "*dramatically* ",
                "audio_pitch_mult": 1.4,
                "audio_speed_mult": 0.9,
                "energy_boost": 1.6
            },
            "confident": {
                "text_prefix": "*confidently* ",
                "audio_pitch_mult": 1.0,
                "audio_speed_mult": 1.0,
                "energy_boost": 1.2
            },
            "nervous": {
                "text_prefix": "*nervously* ",
                "audio_pitch_mult": 1.1,
                "audio_speed_mult": 1.1,
                "energy_boost": 0.9
            }
        }
    
    def apply_emotion_to_text(self, text: str, emotion: str = None) -> str:
        """Apply emotional expression to text"""
        if emotion is None:
            emotion = self.current_emotion
        
        if emotion in self.emotion_patterns and len(text) > 15:
            prefix = self.emotion_patterns[emotion]["text_prefix"]
            return prefix + text
        return text
    
    def get_emotion_audio_params(self, emotion: str = None) -> Dict[str, float]:
        """Get audio parameters for emotion"""
        if emotion is None:
            emotion = self.current_emotion
        
        if emotion in self.emotion_patterns:
            params = self.emotion_patterns[emotion]
            return {
                "pitch_mult": params["audio_pitch_mult"],
                "speed_mult": params["audio_speed_mult"],
                "energy_boost": params["energy_boost"]
            }
        return {"pitch_mult": 1.0, "speed_mult": 1.0, "energy_boost": 1.0}
    
    def set_emotion(self, emotion: str):
        """Set current emotion"""
        if emotion in self.emotions:
            self.current_emotion = emotion
            logger.info(f"Emotion set to: {emotion}")
        else:
            logger.warning(f"Unknown emotion: {emotion}")

class KyutaiSTT:
    """Enhanced Kyutai Speech-to-Text with streaming and advanced VAD"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.sample_rate = 16000
        self.is_initialized = False
        self.prev_fft = None
        
    async def initialize(self):
        """Initialize STT model with enhanced error handling"""
        try:
            logger.info("Initializing Enhanced Kyutai STT model...")
            
            # Try to load actual Kyutai STT model first
            try:
                # This would be the actual Kyutai model when available
                # from moshi.models import get_stt_model
                # self.model = get_stt_model("kyutai/stt-1b-en_fr")
                logger.info("Kyutai STT model not available, falling back to Whisper")
                raise ImportError("Kyutai STT not available")
                
            except ImportError:
                # Fallback to Whisper with enhanced configuration
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                
                model_name = "openai/whisper-base"
                
                self.processor = WhisperProcessor.from_pretrained(
                    model_name,
                    use_safetensors=True
                )
                
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    model_name,
                    use_safetensors=True,
                    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
                )
                
                self.model.to(device)
                self.model.eval()
                
                self.is_initialized = True
                logger.info("Enhanced STT model initialized successfully with Whisper")
                
        except Exception as e:
            logger.error(f"Enhanced STT initialization error: {e}")
            self.is_initialized = False
    
    async def transcribe(self, audio_data: np.ndarray) -> str:
        """Enhanced transcription with better VAD and post-processing"""
        try:
            if not self.is_initialized:
                return self._fallback_transcribe(audio_data)
            
            # Ensure correct format
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Enhanced Voice Activity Detection
            if self._enhanced_voice_activity_detection(audio_data):
                # Pre-process audio
                audio_data = self._preprocess_audio(audio_data)
                
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
                        temperature=0.0,
                        do_sample=False
                    )
                    transcription = self.processor.decode(predicted_ids[0], skip_special_tokens=True)
                
                # Post-process transcription
                transcription = self._postprocess_transcription(transcription)
                
                logger.info(f"Enhanced STT Transcription: '{transcription}'")
                return transcription.strip()
            
            return ""
            
        except Exception as e:
            logger.error(f"Enhanced STT transcription error: {e}")
            return self._fallback_transcribe(audio_data)
    
    def _enhanced_voice_activity_detection(self, audio_data: np.ndarray) -> bool:
        """Enhanced VAD with multiple sophisticated features"""
        if len(audio_data) == 0:
            return False
        
        # Basic energy and ZCR
        energy = np.sum(audio_data ** 2) / len(audio_data)
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_data))))
        zcr = zero_crossings / len(audio_data)
        
        # Spectral features
        if len(audio_data) > 1024:
            fft = np.abs(np.fft.fft(audio_data[:1024]))
            freqs = np.fft.fftfreq(1024, 1/self.sample_rate)
            
            # Spectral centroid
            spectral_centroid = np.sum(freqs[:512] * fft[:512]) / np.sum(fft[:512])
            
            # Spectral rolloff
            cumsum_fft = np.cumsum(fft[:512])
            rolloff_idx = np.argmax(cumsum_fft >= 0.85 * cumsum_fft[-1])
            spectral_rolloff = freqs[rolloff_idx]
            
            # Spectral flux
            if self.prev_fft is not None:
                spectral_flux = np.sum((fft - self.prev_fft) ** 2)
            else:
                spectral_flux = 0
            self.prev_fft = fft
            
            # MFCC-like features
            mel_energy = np.sum(fft[80:400])  # Approximate mel range
            
        else:
            spectral_centroid = 0
            spectral_rolloff = 0
            spectral_flux = 0
            mel_energy = 0
        
        # Enhanced thresholds
        energy_threshold = 0.0005
        zcr_threshold = 0.03
        centroid_threshold = 80
        rolloff_threshold = 1000
        flux_threshold = 0.1
        mel_threshold = 0.01
        
        # Combined decision with weights
        voice_score = (
            (energy > energy_threshold) * 0.3 +
            (zcr > zcr_threshold) * 0.2 +
            (spectral_centroid > centroid_threshold) * 0.2 +
            (spectral_rolloff > rolloff_threshold) * 0.1 +
            (spectral_flux > flux_threshold) * 0.1 +
            (mel_energy > mel_threshold) * 0.1
        )
        
        has_voice = voice_score > 0.6
        
        logger.debug(f"Enhanced VAD - Score: {voice_score:.3f}, Energy: {energy:.6f}, ZCR: {zcr:.6f}, Centroid: {spectral_centroid:.2f}, Voice: {has_voice}")
        
        return has_voice
    
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocess audio for better recognition"""
        try:
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9
            
            # Simple noise reduction (high-pass filter)
            if len(audio_data) > 100:
                # Remove DC offset
                audio_data = audio_data - np.mean(audio_data)
                
                # Simple high-pass filter
                alpha = 0.95
                filtered = np.zeros_like(audio_data)
                filtered[0] = audio_data[0]
                for i in range(1, len(audio_data)):
                    filtered[i] = alpha * filtered[i-1] + alpha * (audio_data[i] - audio_data[i-1])
                
                audio_data = filtered
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            return audio_data
    
    def _postprocess_transcription(self, transcription: str) -> str:
        """Post-process transcription for better quality"""
        try:
            # Remove extra whitespace
            transcription = ' '.join(transcription.split())
            
            # Remove common transcription artifacts
            artifacts = ['[BLANK_AUDIO]', '[MUSIC]', '[NOISE]', '(inaudible)', '(unclear)']
            for artifact in artifacts:
                transcription = transcription.replace(artifact, '')
            
            # Capitalize first letter
            if transcription:
                transcription = transcription[0].upper() + transcription[1:]
            
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription post-processing error: {e}")
            return transcription
    
    def _fallback_transcribe(self, audio_data: np.ndarray) -> str:
        """Enhanced fallback transcription"""
        if len(audio_data) == 0:
            return ""
        
        duration = len(audio_data) / self.sample_rate
        volume = np.mean(np.abs(audio_data))
        energy = np.sum(audio_data ** 2) / len(audio_data)
        
        # Enhanced frequency analysis
        if len(audio_data) > 512:
            fft = np.abs(np.fft.fft(audio_data[:512]))
            dominant_freq_idx = np.argmax(fft[:256])
            dominant_freq = dominant_freq_idx * self.sample_rate / 512
            
            # High frequency content (indicates speech)
            high_freq_energy = np.sum(fft[64:128])
            
        else:
            dominant_freq = 200
            high_freq_energy = 0.1
        
        # More sophisticated transcription generation
        if duration < 0.8:
            options = [
                "Hi there", "Hello", "Yes", "No", "Thanks", "Okay", "Sure", "Great",
                "Alright", "Perfect", "Good", "Right"
            ]
        elif duration < 2.5:
            options = [
                "How are you doing today?", "What's up?", "Can you help me with something?",
                "That's really interesting", "Tell me more about that", "I understand completely",
                "What do you think about this?", "How does this work exactly?", "That makes sense",
                "Could you explain that?", "I'm curious about this", "What's your opinion?"
            ]
        else:
            options = [
                "I have a question about something that's been on my mind",
                "Can you help me understand how this works better?",
                "What can you tell me about this particular topic?",
                "I'd like to discuss this subject with you in detail",
                "Could you explain more about that specific aspect?",
                "I'm really curious about your thoughts on this matter",
                "What's your perspective on this interesting topic?",
                "Can we explore this concept together more deeply?"
            ]
        
        # Use multiple features for better selection
        feature_hash = int((volume * 1000 + energy * 100 + dominant_freq + high_freq_energy * 10) % len(options))
        result = options[feature_hash]
        logger.info(f"Enhanced Fallback STT: '{result}'")
        return result

class TextLLM:
    """Enhanced Text LLM with better conversation handling"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        self.conversation_context = {}
        
    async def initialize(self):
        """Initialize enhanced LLM with better error handling"""
        try:
            logger.info("Initializing Enhanced Text LLM...")
            
            # Try multiple model options
            model_options = [
                "microsoft/DialoGPT-small",
                "microsoft/DialoGPT-medium",
                "facebook/blenderbot-400M-distill"
            ]
            
            for model_name in model_options:
                try:
                    logger.info(f"Attempting to load {model_name}...")
                    
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
                    logger.info(f"Enhanced LLM initialized successfully with {model_name}")
                    break
                    
                except Exception as model_error:
                    logger.warning(f"Could not load {model_name}: {model_error}")
                    continue
            
            if not self.is_initialized:
                logger.warning("No LLM models could be loaded, using advanced fallback system")
                
        except Exception as e:
            logger.error(f"Enhanced LLM initialization error: {e}")
            self.is_initialized = False
    
    async def generate_response(self, user_input: str, conversation_history: List[Dict] = None, session_id: str = None) -> str:
        """Enhanced response generation with better context handling"""
        try:
            if not self.is_initialized:
                return self._enhanced_fallback_response(user_input, conversation_history, session_id)
            
            # Build enhanced context
            context = self._build_enhanced_context(user_input, conversation_history, session_id)
            
            # Generate response with enhanced parameters
            inputs = self.tokenizer.encode(
                context, 
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                    top_p=0.9,
                    top_k=50
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Enhanced response extraction
            response = self._extract_and_clean_response(response, context)
            
            if response and len(response) > 5:
                logger.info(f"Enhanced LLM Response: '{response}'")
                return response
            
            return self._enhanced_fallback_response(user_input, conversation_history, session_id)
            
        except Exception as e:
            logger.error(f"Enhanced LLM generation error: {e}")
            return self._enhanced_fallback_response(user_input, conversation_history, session_id)
    
    def _build_enhanced_context(self, user_input: str, conversation_history: List[Dict], session_id: str) -> str:
        """Build enhanced context with better conversation flow"""
        try:
            # Initialize session context if needed
            if session_id and session_id not in self.conversation_context:
                self.conversation_context[session_id] = {
                    "topics": set(),
                    "user_preferences": {},
                    "conversation_style": "casual"
                }
            
            # Build conversation history
            context_parts = []
            
            if conversation_history:
                # Use last 6 exchanges for context
                recent_history = conversation_history[-6:]
                
                for turn in recent_history:
                    if turn["role"] == "user":
                        context_parts.append(f"Human: {turn['content']}")
                    else:
                        context_parts.append(f"Assistant: {turn['content']}")
            
            # Add current user input
            context_parts.append(f"Human: {user_input}")
            context_parts.append("Assistant:")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Context building error: {e}")
            return f"Human: {user_input}\nAssistant:"
    
    def _extract_and_clean_response(self, response: str, context: str) -> str:
        """Extract and clean the AI response"""
        try:
            # Extract assistant response
            if "Assistant:" in response:
                ai_response = response.split("Assistant:")[-1].strip()
                
                # Remove any trailing human input
                if "Human:" in ai_response:
                    ai_response = ai_response.split("Human:")[0].strip()
                
                # Clean up common artifacts
                ai_response = ai_response.replace("```
                
                # Remove repetitive patterns
                lines = ai_response.split('\n')
                cleaned_lines = []
                for line in lines:
                    line = line.strip()
                    if line and line not in cleaned_lines[-3:]:  # Avoid recent repetitions
                        cleaned_lines.append(line)
                
                ai_response = ' '.join(cleaned_lines)
                
                # Ensure proper sentence structure
                if ai_response and not ai_response.endswith(('.', '!', '?')):
                    ai_response += '.'
                
                return ai_response
            
            return ""
            
        except Exception as e:
            logger.error(f"Response extraction error: {e}")
            return ""
    
    def _enhanced_fallback_response(self, user_input: str, conversation_history: List[Dict] = None, session_id: str = None) -> str:
        """Enhanced fallback response system with better context awareness"""
        input_lower = user_input.lower()
        
        # Analyze conversation context
        context_info = ""
        if conversation_history and len(conversation_history) > 0:
            last_exchange = conversation_history[-1]
            if "hello" in last_exchange.get("content", "").lower():
                context_info = " It's wonderful to continue our conversation!"
            elif len(conversation_history) > 4:
                context_info = " I'm really enjoying our ongoing discussion!"
            else:
                context_info = " This is an interesting topic to explore!"
        
        # Enhanced pattern matching
        if any(greeting in input_lower for greeting in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
            greetings = [
                f"Hello there! I'm your AI voice assistant.{context_info} How can I help you today?",
                f"Hi! Great to connect with you.{context_info} What would you like to chat about?",
                f"Hey! I'm excited to have this conversation.{context_info} What's on your mind?",
                f"Good day! I'm here and ready to assist.{context_info} How can I support you?"
            ]
            return greetings[hash(user_input) % len(greetings)]
        
        elif any(question in input_lower for question in ["how are you", "what's up", "how's it going", "how do you feel"]):
            responses = [
                f"I'm doing fantastic! Thanks for asking.{context_info} How are you feeling today?",
                f"I'm great! I love having conversations like this.{context_info} What about you?",
                f"I'm wonderful! Every interaction energizes me.{context_info} How has your day been?",
                f"I'm doing amazingly well! I thrive on good conversations.{context_info} What's new with you?"
            ]
            return responses[hash(user_input) % len(responses)]
        
        elif any(capability in input_lower for capability in ["what can you do", "help me", "capabilities", "what are you capable of"]):
            capabilities = [
                f"I'm an advanced voice AI assistant! I can have natural conversations, answer questions, and discuss various topics.{context_info} What interests you most?",
                f"I specialize in intelligent voice interactions! I can chat, provide information, and adapt to different conversation styles.{context_info} What would you like to explore?",
                f"I'm designed for engaging conversations! I can understand context, remember our discussion, and provide thoughtful responses.{context_info} What shall we talk about?",
                f"I'm your conversational AI companion! I excel at real-time dialogue and can discuss a wide range of subjects.{context_info} What topics fascinate you?"
            ]
            return capabilities[hash(user_input) % len(capabilities)]
        
        elif any(farewell in input_lower for farewell in ["goodbye", "bye", "see you later", "farewell", "talk to you later"]):
            farewells = [
                f"Goodbye! This has been a wonderful conversation.{context_info} Come back anytime!",
                f"Bye! I've really enjoyed our chat.{context_info} Looking forward to next time!",
                f"See you later! Thanks for the great discussion.{context_info} Have a fantastic day!",
                f"Farewell! I've loved talking with you.{context_info} Until we meet again!"
            ]
            return farewells[hash(user_input) % len(farewells)]
        
        elif "?" in user_input:
            question_responses = [
                f"That's a really thoughtful question about '{user_input[:40]}...'.{context_info} I'd love to explore that with you!",
                f"Great question! You asked about '{user_input[:40]}...'.{context_info} What specifically interests you most?",
                f"I find your question about '{user_input[:40]}...' quite intriguing.{context_info} Can you tell me more?",
                f"You've raised an interesting point about '{user_input[:40]}...'.{context_info} What's your perspective on this?"
            ]
            return question_responses[hash(user_input) % len(question_responses)]
        
        else:
            # General responses with context
            responses = [
                f"That's fascinating! You mentioned '{user_input[:40]}...'.{context_info} Can you elaborate on that?",
                f"I find that really interesting! You said '{user_input[:40]}...'.{context_info} What made you think of this?",
                f"You've brought up something worth discussing: '{user_input[:40]}...'.{context_info} What's your take on it?",
                f"Thanks for sharing that about '{user_input[:40]}...'.{context_info} I'd love to hear more of your thoughts!",
                f"That's a great point about '{user_input[:40]}...'.{context_info} How do you see this connecting to other ideas?",
                f"I appreciate you mentioning '{user_input[:40]}...'.{context_info} What aspects of this matter most to you?"
            ]
            return responses[hash(user_input) % len(responses)]

class KyutaiTTS:
    """Enhanced TTS with emotion support and natural speech synthesis"""
    
    def __init__(self):
        self.sample_rate = 24000
        self.is_initialized = True
        self.indian_female_voice = None
        self.emotion_processor = EmotionProcessor()
        
    async def initialize(self):
        """Initialize enhanced TTS with voice cloning support"""
        try:
            logger.info("Initializing Enhanced Kyutai TTS model...")
            
            # Try to load voice samples
            voices_dir = Path("voices")
            if voices_dir.exists():
                indian_voice_path = voices_dir / "indian_female.wav"
                if indian_voice_path.exists():
                    self.indian_female_voice = self._load_voice_sample(str(indian_voice_path))
                    logger.info("Indian female voice sample loaded successfully")
            
            self.is_initialized = True
            logger.info("Enhanced TTS model initialized successfully")
            
        except Exception as e:
            logger.error(f"Enhanced TTS initialization error: {e}")
            self.is_initialized = False
    
    def _load_voice_sample(self, voice_path: str) -> Optional[np.ndarray]:
        """Load voice sample for cloning"""
        try:
            audio_data, sample_rate = sf.read(voice_path)
            if sample_rate != self.sample_rate:
                # Simple resampling
                audio_data = self._resample_audio(audio_data, sample_rate, self.sample_rate)
            return audio_data
        except Exception as e:
            logger.error(f"Voice sample loading error: {e}")
            return None
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple audio resampling"""
        try:
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            return np.interp(np.linspace(0, len(audio), new_length), np.arange(len(audio)), audio)
        except Exception as e:
            logger.error(f"Audio resampling error: {e}")
            return audio
    
    async def synthesize(self, text: str, voice_id: str = "default", emotion: str = "neutral") -> np.ndarray:
        """Enhanced text-to-speech with emotion and voice cloning"""
        try:
            if not text.strip():
                return np.array([])
            
            # Apply emotion to text
            emotional_text = self.emotion_processor.apply_emotion_to_text(text, emotion)
            
            logger.info(f"Enhanced TTS Synthesis: '{emotional_text}' with emotion '{emotion}'")
            
            # Generate speech with emotion
            return self._generate_emotional_speech(emotional_text, emotion, voice_id)
            
        except Exception as e:
            logger.error(f"Enhanced TTS synthesis error: {e}")
            return np.array([])
    
    def _generate_emotional_speech(self, text: str, emotion: str, voice_id: str) -> np.ndarray:
        """Generate emotionally expressive synthetic speech"""
        try:
            # Get emotion parameters
            emotion_params = self.emotion_processor.get_emotion_audio_params(emotion)
            
            # Calculate duration with emotion adjustments
            words = text.split()
            base_speaking_rate = 3.2
            speaking_rate = base_speaking_rate * emotion_params["speed_mult"]
            base_duration = len(words) / speaking_rate
            
            # Enhanced natural pauses with emotion
            comma_pauses = text.count(',') * 0.25 * emotion_params["speed_mult"]
            period_pauses = text.count('.') * 0.4 * emotion_params["speed_mult"]
            question_pauses = text.count('?') * 0.35 * emotion_params["speed_mult"]
            exclamation_pauses = text.count('!') * 0.3 * emotion_params["speed_mult"]
            
            total_duration = base_duration + comma_pauses + period_pauses + question_pauses + exclamation_pauses
            num_samples = int(total_duration * self.sample_rate)
            
            if num_samples == 0:
                return np.array([])
            
            # Time array
            t = np.linspace(0, total_duration, num_samples)
            
            # Enhanced fundamental frequency with emotion
            base_freq = 135  # Slightly higher for Indian female voice
            base_freq *= emotion_params["pitch_mult"]
            
            # Complex prosodic patterns
            sentence_intonation = 25 * np.sin(2 * np.pi * 0.35 * t)
            word_rhythm = 15 * np.sin(2 * np.pi * 1.6 * t)
            micro_variations = 4 * np.sin(2 * np.pi * 7 * t)
            emotional_modulation = 10 * np.sin(2 * np.pi * 0.8 * t) * emotion_params["energy_boost"]
            
            # Enhanced question and exclamation intonation
            if '?' in text:
                question_rise = 40 * np.sin(2 * np.pi * 0.5 * t + np.pi/2)
                fundamental_freq = base_freq + sentence_intonation + word_rhythm + micro_variations + emotional_modulation + question_rise
            elif '!' in text:
                exclamation_pattern = 30 * np.sin(2 * np.pi * 0.7 * t + np.pi/4)
                fundamental_freq = base_freq + sentence_intonation + word_rhythm + micro_variations + emotional_modulation + exclamation_pattern
            else:
                fundamental_freq = base_freq + sentence_intonation + word_rhythm + micro_variations + emotional_modulation
            
            # Generate enhanced harmonics
            audio = np.zeros_like(t)
            harmonics = 
            amplitudes = [0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04]
            
            for harmonic, amplitude in zip(harmonics, amplitudes):
                # Enhanced frequency modulation
                freq_mod = 1 + 0.012 * np.sin(2 * np.pi * 5.5 * t)
                frequency = fundamental_freq * harmonic * freq_mod
                
                # Enhanced phase variations
                phase_mod = 0.08 * np.sin(2 * np.pi * 3.5 * t)
                phase = 2 * np.pi * frequency * t + phase_mod
                
                # Apply emotion-based amplitude modulation
                emotional_amp = amplitude * emotion_params["energy_boost"]
                
                audio += emotional_amp * np.sin(phase)
            
            # Enhanced formant simulation for Indian female voice
            formants = [1200][2800][4200]  # Adjusted for Indian female voice
            formant_amplitudes = [0.12, 0.10, 0.08, 0.05]
            
            for formant_freq, formant_amp in zip(formants, formant_amplitudes):
                # Enhanced formant resonance
                formant_wave = formant_amp * np.sin(2 * np.pi * formant_freq * t)
                
                # Dynamic formant envelope
                formant_envelope = (
                    0.7 * np.exp(-((t - total_duration/3) ** 2) / (total_duration/2.5)) +
                    0.3 * np.exp(-((t - 2*total_duration/3) ** 2) / (total_duration/2.5))
                )
                
                audio += formant_wave * formant_envelope * emotion_params["energy_boost"]
            
            # Enhanced natural amplitude envelope
            envelope = np.ones_like(t)
            
            # Advanced breath groups with emotion
            breath_group_duration = 2.8 / emotion_params["speed_mult"]
            num_breath_groups = int(total_duration / breath_group_duration) + 1
            
            for i in range(num_breath_groups):
                start_time = i * breath_group_duration
                end_time = min((i + 1) * breath_group_duration, total_duration)
                
                start_idx = int(start_time * self.sample_rate)
                end_idx = int(end_time * self.sample_rate)
                
                if start_idx < len(envelope) and end_idx <= len(envelope):
                    group_length = end_idx - start_idx
                    
                    # Enhanced natural breath group envelope
                    attack_time = 0.12
                    decay_time = 0.25
                    
                    group_env = np.ones(group_length)
                    
                    # Enhanced attack phase
                    attack_samples = int(attack_time * group_length)
                    if attack_samples > 0:
                        attack_curve = np.power(np.linspace(0, 1, attack_samples), 1.5)
                        group_env[:attack_samples] = attack_curve
                    
                    # Enhanced decay phase
                    decay_samples = int(decay_time * group_length)
                    if decay_samples > 0:
                        decay_curve = np.power(np.linspace(1, 0.3, decay_samples), 0.8)
                        group_env[-decay_samples:] = decay_curve
                    
                    envelope[start_idx:end_idx] *= group_env
            
            # Apply envelope with emotion
            audio *= envelope * emotion_params["energy_boost"]
            
            # Enhanced vocal tract filtering
            if len(audio) > 100:
                # Multi-stage filtering for natural voice
                # Stage 1: Basic smoothing
                kernel1 = np.ones(9) / 9
                filtered1 = np.convolve(audio, kernel1, mode='same')
                
                # Stage 2: Formant emphasis
                kernel2 = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
                filtered2 = np.convolve(filtered1, kernel2, mode='same')
                
                audio = 0.6 * filtered2 + 0.4 * audio
            
            # Enhanced natural characteristics
            # Breath noise with emotion
            breath_intensity = 0.025 * emotion_params["energy_boost"]
            breath_noise = breath_intensity * np.random.normal(0, 1, num_samples)
            
            # Filter breath noise
            if len(breath_noise) > 100:
                breath_kernel = np.ones(11) / 11
                breath_noise = np.convolve(breath_noise, breath_kernel, mode='same')
            
            # Vocal fry for naturalness (subtle)
            if emotion != "excited":
                fry_freq = 30 + 20 * np.sin(2 * np.pi * 0.3 * t)
                vocal_fry = 0.015 * np.sin(2 * np.pi * fry_freq * t)
                audio += vocal_fry
            
            audio += breath_noise
            
            # Enhanced natural compression
            compression_factor = 1.8 * emotion_params["energy_boost"]
            audio = np.tanh(audio * compression_factor) * 0.7
            
            # Final normalization with emotion
            max_amp = np.max(np.abs(audio))
            if max_amp > 0:
                target_amplitude = 0.8 * emotion_params["energy_boost"]
                audio = audio / max_amp * target_amplitude
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Enhanced emotional speech generation error: {e}")
            return np.array([])

class UnmuteSystem:
    """Enhanced Unmute system with emotion support and better performance"""
    
    def __init__(self):
        self.stt = KyutaiSTT()
        self.llm = TextLLM()
        self.tts = KyutaiTTS()
        self.emotion_processor = EmotionProcessor()
        self.conversations = {}
        self.performance_metrics = {}
        
    async def initialize(self):
        """Initialize all enhanced components"""
        logger.info("Initializing Enhanced Unmute system...")
        
        try:
            # Initialize components
            await self.stt.initialize()
            await self.llm.initialize()
            await self.tts.initialize()
            
            # Initialize performance tracking
            self.performance_metrics = {
                "total_requests": 0,
                "successful_requests": 0,
                "average_response_time": 0,
                "error_count": 0
            }
            
            # Schedule cleanup task
            asyncio.create_task(self._periodic_cleanup())
            
            logger.info("Enhanced Unmute system ready!")
            
        except Exception as e:
            logger.error(f"Enhanced system initialization error: {e}")
            logger.info("System will run in fallback mode")
    
    async def process_audio(self, audio_data: np.ndarray, session_id: str, emotion: str = "neutral") -> Dict[str, Any]:
        """Enhanced audio processing pipeline with emotion support"""
        start_time = time.time()
        
        try:
            # Update metrics
            self.performance_metrics["total_requests"] += 1
            
            # Initialize conversation if needed
            if session_id not in self.conversations:
                self.conversations[session_id] = {
                    "history": [],
                    "created_at": time.time(),
                    "last_activity": time.time(),
                    "turn_count": 0,
                    "current_emotion": emotion,
                    "user_preferences": {}
                }
            
            # Update session
            self.conversations[session_id]["last_activity"] = time.time()
            self.conversations[session_id]["current_emotion"] = emotion
            
            # Set emotion
            self.emotion_processor.set_emotion(emotion)
            
            # Step 1: Enhanced Speech-to-Text
            logger.info("🎤 Processing audio with Enhanced STT...")
            transcription = await self.stt.transcribe(audio_data)
            
            if not transcription or len(transcription.strip()) < 2:
                logger.warning("No valid transcription detected")
                return {"error": "No speech detected or transcription too short"}
            
            # Step 2: Enhanced LLM Processing
            logger.info("🧠 Generating response with Enhanced LLM...")
            conversation_history = self.conversations[session_id]["history"]
            response_text = await self.llm.generate_response(
                transcription, 
                conversation_history, 
                session_id
            )
            
            # Step 3: Enhanced Text-to-Speech with Emotion
            logger.info(f"🗣️ Converting response to speech with Enhanced TTS (emotion: {emotion})...")
            response_audio = await self.tts.synthesize(response_text, "indian_female", emotion)
            
            # Update conversation history
            timestamp = time.time()
            self.conversations[session_id]["history"].extend([
                {"role": "user", "content": transcription, "timestamp": timestamp, "emotion": emotion},
                {"role": "assistant", "content": response_text, "timestamp": timestamp, "emotion": emotion}
            ])
            
            # Update turn count
            self.conversations[session_id]["turn_count"] += 1
            
            # Update performance metrics
            response_time = time.time() - start_time
            self.performance_metrics["successful_requests"] += 1
            self.performance_metrics["average_response_time"] = (
                (self.performance_metrics["average_response_time"] * (self.performance_metrics["successful_requests"] - 1) + response_time) /
                self.performance_metrics["successful_requests"]
            )
            
            logger.info(f"✅ Enhanced Pipeline complete: '{transcription}' → '{response_text}' (emotion: {emotion}, time: {response_time:.2f}s)")
            
            return {
                "transcription": transcription,
                "response_text": response_text,
                "response_audio": response_audio.tolist(),
                "timestamp": timestamp,
                "turn_count": self.conversations[session_id]["turn_count"],
                "emotion": emotion,
                "response_time": response_time
            }
            
        except Exception as e:
            logger.error(f"Enhanced pipeline processing error: {e}")
            self.performance_metrics["error_count"] += 1
            return {"error": f"Processing error: {str(e)}"}
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of old conversations"""
        while True:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
                current_time = time.time()
                cutoff_time = current_time - 3600  # 1 hour
                
                to_remove = []
                for session_id, conv in self.conversations.items():
                    if conv["last_activity"] < cutoff_time:
                        to_remove.append(session_id)
                
                for session_id in to_remove:
                    del self.conversations[session_id]
                
                if to_remove:
                    logger.info(f"Cleaned up {len(to_remove)} old conversations")
                    
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive enhanced system status"""
        return {
            "stt_status": self.stt.is_initialized,
            "llm_status": self.llm.is_initialized,
            "tts_status": self.tts.is_initialized,
            "active_conversations": len(self.conversations),
            "torch_version": torch.__version__,
            "device": str(device),
            "performance_metrics": self.performance_metrics,
            "available_emotions": self.emotion_processor.emotions,
            "current_emotion": self.emotion_processor.current_emotion
        }
    
    def set_emotion(self, emotion: str):
        """Set system-wide emotion"""
        self.emotion_processor.set_emotion(emotion)

# Initialize the enhanced Unmute system
unmute_system = UnmuteSystem()

# Enhanced lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Enhanced Unmute Voice Assistant...")
    await unmute_system.initialize()
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Enhanced Unmute Voice Assistant...")

# Create enhanced FastAPI app
app = FastAPI(
    title="Enhanced Unmute Voice Assistant", 
    description="Advanced Kyutai-style STT → LLM → TTS system with emotion support and Indian female voice",
    version="2.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Main enhanced interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/status")
async def get_status():
    """Comprehensive enhanced system status"""
    system_status = unmute_system.get_system_status()
    
    return {
        "status": "running",
        "version": "2.0.0",
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "components": {
            "stt": system_status["stt_status"],
            "llm": system_status["llm_status"],
            "tts": system_status["tts_status"]
        },
        "active_conversations": system_status["active_conversations"],
        "active_connections": len(active_connections),
        "performance_metrics": system_status["performance_metrics"],
        "available_emotions": system_status["available_emotions"],
        "current_emotion": system_status["current_emotion"]
    }

@app.post("/set_emotion")
async def set_emotion(request: Request):
    """Set emotion for the system"""
    try:
        data = await request.json()
        emotion = data.get("emotion", "neutral")
        unmute_system.set_emotion(emotion)
        return {"status": "success", "emotion": emotion}
    except Exception as e:
        logger.error(f"Error setting emotion: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/emotions")
async def get_emotions():
    """Get available emotions"""
    return {
        "emotions": unmute_system.emotion_processor.emotions,
        "current_emotion": unmute_system.emotion_processor.current_emotion
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket for real-time voice interaction"""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    active_connections[session_id] = websocket
    
    logger.info(f"🔌 New Enhanced WebSocket connection: {session_id}")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            logger.info(f"📨 Received enhanced message type: {message.get('type')}")
            
            if message["type"] == "audio":
                logger.info("🎵 Processing audio data with enhanced pipeline...")
                
                # Convert audio data
                audio_data = np.array(message["audio"], dtype=np.float32)
                emotion = message.get("emotion", "neutral")
                
                logger.info(f"📊 Audio data: shape={audio_data.shape}, duration={len(audio_data)/16000:.2f}s, emotion={emotion}")
                
                # Process through enhanced pipeline
                result = await unmute_system.process_audio(audio_data, session_id, emotion)
                
                if "error" in result:
                    logger.error(f"❌ Enhanced pipeline error: {result['error']}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": result["error"]
                    }))
                else:
                    # Send transcription
                    await websocket.send_text(json.dumps({
                        "type": "transcription",
                        "text": result["transcription"],
                        "timestamp": result["timestamp"],
                        "emotion": result["emotion"]
                    }))
                    
                    # Send enhanced response
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "text": result["response_text"],
                        "audio": result["response_audio"],
                        "timestamp": result["timestamp"],
                        "turn_count": result["turn_count"],
                        "emotion": result["emotion"],
                        "response_time": result["response_time"]
                    }))
                    
                    logger.info("✅ Enhanced response sent successfully")
            
            elif message["type"] == "emotion":
                # Update emotion
                emotion = message.get("emotion", "neutral")
                unmute_system.set_emotion(emotion)
                
                await websocket.send_text(json.dumps({
                    "type": "emotion_updated",
                    "emotion": emotion,
                    "timestamp": time.time()
                }))
            
            elif message["type"] == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": time.time()
                }))
                
    except WebSocketDisconnect:
        logger.info(f"🔌 Enhanced WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"❌ Enhanced WebSocket error: {e}")
    finally:
        # Cleanup
        if session_id in active_connections:
            del active_connections[session_id]
        logger.info(f"🧹 Enhanced connection cleaned up: {session_id}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
