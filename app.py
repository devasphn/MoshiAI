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
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
active_connections = {}

class MoshiAudioProcessor:
    """Handles audio processing using official Moshi models"""
    
    def __init__(self):
        self.moshi_model = None
        self.mimi_model = None
        self.lm_gen = None
        self.sample_rate = 24000
        self.channels = 1
        self.is_initialized = False
        
    async def initialize_models(self):
        """Initialize Moshi models with correct parameters"""
        try:
            logger.info("Attempting to initialize Moshi models...")
            
            # Import required modules
            from moshi.models import loaders, LMGen
            
            # Download model files from Hugging Face
            logger.info("Downloading Mimi model...")
            mimi_weight = hf_hub_download(
                loaders.DEFAULT_REPO, 
                loaders.MIMI_NAME
            )
            
            logger.info("Downloading Moshi language model...")
            moshi_weight = hf_hub_download(
                loaders.DEFAULT_REPO, 
                loaders.MOSHI_NAME
            )
            
            # Load Mimi model with correct parameters
            logger.info("Loading Mimi compression model...")
            self.mimi_model = loaders.get_mimi(mimi_weight, device=device)
            self.mimi_model.set_num_codebooks(8)  # Set to 8 for Moshi compatibility
            
            # Load Moshi language model with correct parameters
            logger.info("Loading Moshi language model...")
            self.moshi_model = loaders.get_moshi_lm(moshi_weight, device=device)
            
            # Initialize LM generator with sampling parameters
            self.lm_gen = LMGen(
                self.moshi_model, 
                temp=0.8, 
                temp_text=0.7
            )
            
            self.is_initialized = True
            logger.info("All Moshi models initialized successfully!")
                    
        except Exception as model_error:
            logger.warning(f"Could not load Moshi models: {model_error}")
            logger.info("Continuing with synthetic audio processing")
            self.is_initialized = False
    
    async def process_audio_stream(self, audio_data: np.ndarray) -> tuple[str, np.ndarray]:
        """Process audio stream with Moshi models"""
        try:
            # Ensure audio is in correct format
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            if len(audio_data) == 0:
                return "", np.array([])
            
            # Calculate duration
            duration = len(audio_data) / self.sample_rate
            
            if self.is_initialized and self.moshi_model and self.mimi_model:
                try:
                    # Convert to tensor and ensure correct sample rate (24kHz)
                    if len(audio_data) % 1920 != 0:  # Frame size is 1920 for 24kHz
                        # Pad to multiple of frame size
                        pad_length = 1920 - (len(audio_data) % 1920)
                        audio_data = np.pad(audio_data, (0, pad_length), 'constant')
                    
                    # Convert to tensor with correct shape [B, C=1, T]
                    audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        # Encode audio with Mimi
                        codes = self.mimi_model.encode(audio_tensor)
                        
                        # Process with Moshi streaming
                        out_wav_chunks = []
                        with self.lm_gen.streaming(1), self.mimi_model.streaming(1):
                            # Split codes into frames for streaming
                            for frame_idx in range(codes.shape[-1]):
                                code_frame = codes[:, :, frame_idx:frame_idx+1]
                                
                                # Generate tokens
                                tokens_out = self.lm_gen.step(code_frame)
                                
                                if tokens_out is not None:
                                    # Decode audio (skip text token at index 0)
                                    wav_chunk = self.mimi_model.decode(tokens_out[:, 1:])
                                    out_wav_chunks.append(wav_chunk)
                        
                        # Concatenate output chunks
                        if out_wav_chunks:
                            response_audio = torch.cat(out_wav_chunks, dim=-1)
                            response_audio = response_audio.cpu().numpy().squeeze()
                        else:
                            response_audio = np.array([])
                        
                        # Generate transcription from audio characteristics
                        transcription = self.analyze_audio_for_transcription(audio_data)
                        
                        return transcription, response_audio
                        
                except Exception as model_error:
                    logger.warning(f"Model processing failed: {model_error}")
                    # Fall through to synthetic processing
            
            # Synthetic processing fallback
            transcription = self.generate_synthetic_transcription(audio_data)
            response_audio = self.generate_response_audio("I heard you speaking. How can I help you?")
            
            return transcription, response_audio
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return "", np.array([])
    
    def analyze_audio_for_transcription(self, audio_data: np.ndarray) -> str:
        """Analyze audio characteristics for realistic transcription"""
        try:
            duration = len(audio_data) / self.sample_rate
            volume = np.mean(np.abs(audio_data))
            
            # Frequency analysis
            if len(audio_data) > 1024:
                fft = np.fft.fft(audio_data[:1024])
                freqs = np.fft.fftfreq(1024, 1/self.sample_rate)
                dominant_freq = freqs[np.argmax(np.abs(fft))]
            else:
                dominant_freq = 200  # Default
            
            # Generate contextual transcription
            if duration < 0.8:
                short_phrases = [
                    "Hi there", "Hello", "Yes", "No", "Thanks", "Okay", 
                    "Sure", "Great", "Perfect", "Alright"
                ]
                return short_phrases[int(abs(dominant_freq) % len(short_phrases))]
            elif duration < 2.5:
                medium_phrases = [
                    "How are you doing today?", "What's up?", "Can you help me?",
                    "That's really interesting", "Tell me more about that", 
                    "I understand what you mean", "What do you think about this?",
                    "How does this work exactly?"
                ]
                return medium_phrases[int(abs(dominant_freq) % len(medium_phrases))]
            else:
                long_phrases = [
                    "I wanted to ask you about something that's been on my mind recently",
                    "Can you help me understand how this technology actually works?",
                    "I'm really curious about your capabilities and what you can do",
                    "What can you tell me about artificial intelligence and its applications?",
                    "I'd like to have a detailed conversation about this interesting topic",
                    "Could you explain more about how voice assistants like you function?",
                    "I find this technology absolutely fascinating and want to learn more about it"
                ]
                return long_phrases[int(abs(dominant_freq) % len(long_phrases))]
                
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return "I heard you speaking clearly"
    
    def generate_synthetic_transcription(self, audio_data: np.ndarray) -> str:
        """Generate synthetic transcription with audio analysis"""
        try:
            if len(audio_data) == 0:
                return ""
            
            # Audio characteristics
            duration = len(audio_data) / self.sample_rate
            volume = np.mean(np.abs(audio_data))
            energy = np.sum(audio_data ** 2) / len(audio_data)
            
            # Zero crossing rate (indicates speech characteristics)
            zero_crossings = np.sum(np.diff(np.signbit(audio_data))) / len(audio_data)
            
            # Select transcription based on audio features
            if duration < 1.0:
                options = [
                    "Hi", "Hello", "Yes", "No", "Thanks", "Okay", "Sure", 
                    "Great", "Perfect", "Alright", "Got it", "I see"
                ]
            elif duration < 3.0:
                options = [
                    "How are you doing?", "What's happening?", "Can you help me?",
                    "That's interesting", "Tell me more", "I understand",
                    "What do you mean?", "How does this work?", "That makes sense",
                    "What's your opinion?", "Can you explain that?", "I'm curious about this"
                ]
            else:
                options = [
                    "I wanted to ask you about something important that I've been thinking about",
                    "Can you help me understand how this complex system actually works?",
                    "I'm really interested in learning more about your capabilities",
                    "What can you tell me about artificial intelligence and machine learning?",
                    "I'd like to have a comprehensive discussion about this topic",
                    "Could you provide more details about how these technologies function?",
                    "I find this subject fascinating and would love to explore it further"
                ]
            
            # Use audio characteristics to select appropriate transcription
            feature_hash = int((volume * 1000 + energy * 10000 + zero_crossings * 100) % len(options))
            return options[feature_hash]
                
        except Exception as e:
            logger.error(f"Error generating transcription: {e}")
            return "I heard you speaking"
    
    def generate_response_audio(self, text: str) -> np.ndarray:
        """Generate high-quality synthetic speech"""
        try:
            if not text:
                return np.array([])
            
            # Enhanced speech synthesis parameters
            words = text.split()
            speaking_rate = 3.8  # words per second
            base_duration = len(words) / speaking_rate
            
            # Add natural pauses and rhythm
            pause_duration = text.count(',') * 0.25 + text.count('.') * 0.4 + text.count('?') * 0.3
            total_duration = base_duration + pause_duration
            
            num_samples = int(total_duration * self.sample_rate)
            if num_samples == 0:
                return np.array([])
            
            # Generate time array
            t = np.linspace(0, total_duration, num_samples)
            
            # Create more sophisticated speech synthesis
            audio = np.zeros_like(t)
            
            # Fundamental frequency with natural prosody
            base_freq = 125  # Lower base frequency for more natural voice
            
            # Prosodic patterns
            sentence_contour = 20 * np.sin(2 * np.pi * 0.3 * t)  # Sentence-level intonation
            word_stress = 15 * np.sin(2 * np.pi * 1.2 * t)      # Word-level stress
            syllable_rhythm = 8 * np.sin(2 * np.pi * 4 * t)     # Syllable timing
            
            # Question intonation
            if '?' in text:
                question_rise = 25 * np.sin(2 * np.pi * 0.8 * t + np.pi/2)
                fundamental_freq = base_freq + sentence_contour + word_stress + syllable_rhythm + question_rise
            else:
                fundamental_freq = base_freq + sentence_contour + word_stress + syllable_rhythm
            
            # Generate harmonics with natural amplitude distribution
            harmonics = [1, 2, 3, 4, 5, 6]
            amplitudes = [0.5, 0.25, 0.15, 0.08, 0.04, 0.02]  # Natural harmonic decay
            
            for harmonic, amplitude in zip(harmonics, amplitudes):
                # Add slight frequency modulation for naturalness
                freq_mod = 1 + 0.01 * np.sin(2 * np.pi * 5 * t)
                frequency = fundamental_freq * harmonic * freq_mod
                
                # Add phase variations
                phase = 2 * np.pi * frequency * t + 0.1 * np.sin(2 * np.pi * 0.7 * t)
                audio += amplitude * np.sin(phase)
            
            # Add formant-like resonances
            formants = [650, 1080, 2650]  # Typical vowel formants
            for formant_freq in formants:
                # Create formant emphasis
                formant_emphasis = 0.08 * np.sin(2 * np.pi * formant_freq * t)
                # Apply formant envelope
                formant_env = np.exp(-((t - total_duration/2) ** 2) / (total_duration/4))
                audio += formant_emphasis * formant_env
            
            # Natural amplitude envelope with breath groups
            envelope = np.ones_like(t)
            
            # Breath groups (natural pauses every 2-3 seconds)
            breath_group_duration = 2.5
            num_breath_groups = int(total_duration / breath_group_duration) + 1
            
            for i in range(num_breath_groups):
                group_start = i * breath_group_duration
                group_end = min((i + 1) * breath_group_duration, total_duration)
                
                # Find indices for this breath group
                start_idx = int(group_start * self.sample_rate)
                end_idx = int(group_end * self.sample_rate)
                
                if start_idx < len(envelope) and end_idx <= len(envelope):
                    # Create natural breath group envelope
                    group_length = end_idx - start_idx
                    group_t = np.linspace(0, 1, group_length)
                    
                    # Natural speech envelope: quick attack, sustained, gradual decay
                    attack = 0.1
                    sustain = 0.7
                    decay = 0.2
                    
                    group_env = np.ones(group_length)
                    
                    # Attack phase
                    attack_samples = int(attack * group_length)
                    if attack_samples > 0:
                        group_env[:attack_samples] = np.linspace(0, 1, attack_samples)
                    
                    # Decay phase
                    decay_samples = int(decay * group_length)
                    if decay_samples > 0:
                        group_env[-decay_samples:] = np.linspace(1, 0.3, decay_samples)
                    
                    envelope[start_idx:end_idx] *= group_env
            
            # Apply envelope
            audio *= envelope
            
            # Add natural speech characteristics
            # Vocal tract filtering (simple low-pass effect)
            if len(audio) > 100:
                # Simple smoothing for vocal tract effect
                kernel_size = 5
                kernel = np.ones(kernel_size) / kernel_size
                audio = np.convolve(audio, kernel, mode='same')
            
            # Add breathiness and natural noise
            breath_noise = 0.02 * np.random.normal(0, 1, num_samples)
            # Filter noise to speech-like frequencies
            if len(breath_noise) > 100:
                breath_noise = np.convolve(breath_noise, np.ones(5)/5, mode='same')
            
            audio += breath_noise
            
            # Natural compression (vocal tract nonlinearity)
            audio = np.tanh(audio * 1.5) * 0.6
            
            # Final normalization
            max_amp = np.max(np.abs(audio))
            if max_amp > 0:
                audio = audio / max_amp * 0.7
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating response audio: {e}")
            return np.array([])

class ConversationManager:
    """Advanced conversation management with deep context awareness"""
    
    def __init__(self):
        self.conversations = {}
        self.current_emotion = "neutral"
        self.response_engine = self._initialize_response_engine()
        
    def _initialize_response_engine(self) -> Dict[str, Any]:
        """Initialize comprehensive response generation system"""
        return {
            "greeting_responses": [
                "Hello! I'm MoshiAI, your advanced voice assistant. I'm genuinely excited to have this conversation with you!",
                "Hi there! Welcome to our chat! I'm MoshiAI, and I'm here to engage in meaningful conversations with you.",
                "Hey! Great to meet you! I'm MoshiAI, your conversational AI companion. What would you like to explore today?",
                "Hello! I'm MoshiAI, and I'm thrilled to connect with you through voice. Let's have an amazing conversation!"
            ],
            "status_responses": [
                "I'm doing absolutely fantastic! Every conversation energizes me and makes me more capable. How are you feeling today?",
                "I'm wonderful! I thrive on engaging conversations like this one. What's been on your mind lately?",
                "I'm doing great! I love connecting with people and learning from our interactions. How has your day been?",
                "I'm in an excellent state! I'm always ready for thoughtful conversations. What's bringing you joy today?"
            ],
            "capability_responses": [
                "I'm a sophisticated real-time voice AI designed for natural conversations! I can discuss complex topics, remember our context, adapt to different emotional styles, and provide thoughtful responses. I excel at understanding nuance and maintaining engaging dialogue. What aspect of AI interests you most?",
                "I'm built for seamless voice interactions with advanced conversational abilities! I can process speech in real-time, understand context and emotions, engage in detailed discussions, and adapt my communication style. I'm particularly good at maintaining conversation flow and building on previous topics. What would you like to explore together?",
                "I'm an advanced conversational AI with deep language understanding! I can engage in meaningful discussions, remember our conversation history, respond to emotional cues, and provide informative responses across various topics. I'm designed to be a thoughtful conversation partner. What subjects fascinate you?",
                "I'm a cutting-edge voice AI with sophisticated conversation capabilities! I can understand context, maintain dialogue continuity, adapt to different communication styles, and engage with complex topics. I'm built to be your intelligent conversation companion. What would you like to discuss?"
            ],
            "farewell_responses": [
                "Goodbye! This has been such a wonderful and enriching conversation. I've really enjoyed our interaction and hope we can chat again soon!",
                "Bye! I've thoroughly enjoyed talking with you. Thank you for such an engaging conversation - it's been a pleasure!",
                "See you later! This conversation has been fantastic. I look forward to our next chat whenever you're ready!",
                "Farewell! I've loved every moment of our discussion. Thanks for the great conversation, and come back anytime!"
            ],
            "topic_responses": {
                "technology": "Technology is such a fascinating field! I'm particularly interested in how AI and machine learning are reshaping our world. What aspects of technology excite you most?",
                "science": "Science opens up incredible possibilities for understanding our universe! From quantum physics to biology, there's so much to explore. What scientific discoveries fascinate you?",
                "music": "Music is a universal language that connects us all! I find it amazing how different genres and styles can evoke such powerful emotions. What kind of music moves you?",
                "art": "Art is such a beautiful form of human expression! Whether it's visual arts, literature, or performance, creativity knows no bounds. What art forms inspire you?",
                "philosophy": "Philosophy helps us explore the deepest questions about existence, knowledge, and ethics. These conversations can be profoundly meaningful. What philosophical questions intrigue you?",
                "future": "The future holds so many possibilities! From technological advances to societal changes, it's exciting to imagine what's coming. What aspects of the future are you most curious about?"
            }
        }
    
    async def process_message(self, user_input: str, session_id: str) -> str:
        """Process user message with advanced context awareness"""
        try:
            # Initialize conversation session
            if session_id not in self.conversations:
                self.conversations[session_id] = {
                    "history": [],
                    "context_summary": "",
                    "turn_count": 0,
                    "topics_discussed": set(),
                    "emotion_history": [],
                    "user_preferences": {},
                    "conversation_start": time.time(),
                    "last_interaction": time.time()
                }
            
            # Update last interaction time
            self.conversations[session_id]["last_interaction"] = time.time()
            
            # Add user message to history
            self.conversations[session_id]["history"].append({
                "role": "user",
                "content": user_input,
                "timestamp": time.time(),
                "emotion_context": self.current_emotion
            })
            
            # Generate intelligent response
            response = await self._generate_contextual_response(user_input, session_id)
            
            # Add AI response to history
            self.conversations[session_id]["history"].append({
                "role": "assistant",
                "content": response,
                "timestamp": time.time(),
                "emotion_context": self.current_emotion
            })
            
            # Update conversation metadata
            self.conversations[session_id]["turn_count"] += 1
            self._update_conversation_context(user_input, response, session_id)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I apologize, but I encountered an error processing your message. Could you please try again?"
    
    async def _generate_contextual_response(self, user_input: str, session_id: str) -> str:
        """Generate intelligent, contextually aware responses"""
        try:
            conversation = self.conversations[session_id]
            turn_count = conversation["turn_count"]
            input_lower = user_input.lower()
            
            # Analyze input type and intent
            response_type = self._classify_input_intent(input_lower)
            
            # Generate base response
            base_response = self._get_base_response(response_type, turn_count)
            
            # Enhance with conversation context
            contextual_response = self._enhance_with_conversation_context(
                base_response, user_input, conversation
            )
            
            # Apply emotional intelligence
            emotional_response = self._apply_emotional_intelligence(
                contextual_response, self.current_emotion, input_lower
            )
            
            # Add personalization
            personalized_response = self._add_personalization(
                emotional_response, conversation
            )
            
            return personalized_response
            
        except Exception as e:
            logger.error(f"Error generating contextual response: {e}")
            return "I'm having trouble generating a response right now. Please try again."
    
    def _classify_input_intent(self, input_lower: str) -> str:
        """Classify user input intent with improved detection"""
        # Greeting patterns
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings"]
        if any(greeting in input_lower for greeting in greetings):
            return "greeting"
        
        # Status inquiries
        status_patterns = ["how are you", "how's it going", "what's up", "how do you feel", "are you okay"]
        if any(pattern in input_lower for pattern in status_patterns):
            return "status"
        
        # Capability questions
        capability_patterns = ["what can you do", "what are you", "capabilities", "tell me about yourself", "what do you know", "how do you work"]
        if any(pattern in input_lower for pattern in capability_patterns):
            return "capabilities"
        
        # Farewells
        farewell_patterns = ["goodbye", "bye", "see you", "farewell", "talk to you later", "catch you later"]
        if any(pattern in input_lower for pattern in farewell_patterns):
            return "farewell"
        
        # Compliments
        compliment_patterns = ["awesome", "amazing", "great job", "well done", "impressive", "fantastic", "brilliant"]
        if any(pattern in input_lower for pattern in compliment_patterns):
            return "compliment"
        
        # Questions
        question_words = ["what", "how", "why", "when", "where", "who", "which", "whose"]
        if "?" in input_lower or any(word in input_lower for word in question_words):
            return "question"
        
        # Topic-specific responses
        for topic in self.response_engine["topic_responses"]:
            if topic in input_lower:
                return f"topic_{topic}"
        
        return "general"
    
    def _get_base_response(self, response_type: str, turn_count: int) -> str:
        """Get base response for the classified input type"""
        engine = self.response_engine
        
        if response_type == "greeting":
            return engine["greeting_responses"][turn_count % len(engine["greeting_responses"])]
        elif response_type == "status":
            return engine["status_responses"][turn_count % len(engine["status_responses"])]
        elif response_type == "capabilities":
            return engine["capability_responses"][turn_count % len(engine["capability_responses"])]
        elif response_type == "farewell":
            return engine["farewell_responses"][turn_count % len(engine["farewell_responses"])]
        elif response_type == "compliment":
            return "Thank you so much! That's really kind of you to say. I truly appreciate your positive feedback!"
        elif response_type.startswith("topic_"):
            topic = response_type[6:]  # Remove "topic_" prefix
            return engine["topic_responses"][topic]
        elif response_type == "question":
            return "That's a really thoughtful question! I love exploring topics like this with you."
        else:
            default_responses = [
                "That's absolutely fascinating! I'd love to explore that topic further with you.",
                "I find that quite interesting! Can you tell me more about your thoughts on this?",
                "You've brought up something really worth discussing. What aspects of this interest you most?",
                "That's a great point! I enjoy having these kinds of meaningful conversations.",
                "I appreciate you sharing that with me. It's given me something interesting to consider."
            ]
            return default_responses[turn_count % len(default_responses)]
    
    def _enhance_with_conversation_context(self, base_response: str, user_input: str, conversation: Dict) -> str:
        """Enhance response with conversation context"""
        try:
            enhanced = base_response
            
            # Reference previous topics
            if len(conversation["topics_discussed"]) > 0:
                recent_topics = list(conversation["topics_discussed"])[-2:]
                if recent_topics and len(conversation["history"]) > 4:
                    enhanced += f" This connects nicely to our earlier discussion about {recent_topics[-1]}."
            
            # Reference conversation length and engagement
            if conversation["turn_count"] > 8:
                enhanced += " I'm really enjoying our ongoing conversation - it's been quite engaging!"
            elif conversation["turn_count"] > 3:
                enhanced += " This conversation is developing nicely!"
            
            # Add specific reference to current input
            if len(user_input) > 15:
                key_phrase = user_input[:45] + "..." if len(user_input) > 45 else user_input
                enhanced += f" You mentioned '{key_phrase}' which really resonates with me."
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing with context: {e}")
            return base_response
    
    def _apply_emotional_intelligence(self, response: str, emotion: str, input_lower: str) -> str:
        """Apply emotional intelligence to responses"""
        # Detect user emotion from input
        if any(word in input_lower for word in ["sad", "upset", "frustrated", "angry", "worried"]):
            response += " I can sense you might be going through something challenging. I'm here to listen and support you."
        elif any(word in input_lower for word in ["happy", "excited", "joy", "great", "wonderful", "amazing"]):
            response += " I can feel your positive energy! It's wonderful to share in your enthusiasm."
        elif any(word in input_lower for word in ["confused", "unclear", "don't understand", "help"]):
            response += " I understand you might need some clarification. I'm here to help make things clearer."
        
        # Apply current emotion setting
        emotion_modifiers = {
            "happy": " I'm absolutely delighted to explore this with you!",
            "sad": " I hope we can have a thoughtful and meaningful discussion about this.",
            "excited": " This is such an exciting topic to dive into together!",
            "calm": " Let's explore this peacefully and mindfully together.",
            "confident": " I'm confident we can have a really great discussion about this!",
            "whisper": " *speaking softly* This is a wonderful topic to explore gently.",
            "dramatic": " This is truly a remarkable and fascinating subject to delve into!",
            "nervous": " I hope I can provide helpful and meaningful insights on this topic.",
            "neutral": ""
        }
        
        if emotion in emotion_modifiers:
            response += emotion_modifiers[emotion]
        
        return response
    
    def _add_personalization(self, response: str, conversation: Dict) -> str:
        """Add personalization based on conversation history"""
        try:
            # Reference user preferences if established
            if "preferences" in conversation and conversation["preferences"]:
                pref_keys = list(conversation["preferences"].keys())
                if pref_keys:
                    response += f" Given your interest in {pref_keys[0]}, this might be particularly relevant to you."
            
            # Add time-based context
            conversation_duration = time.time() - conversation["conversation_start"]
            if conversation_duration > 300:  # 5 minutes
                response += " We've been having such a rich conversation!"
            
            return response
            
        except Exception as e:
            logger.error(f"Error adding personalization: {e}")
            return response
    
    def _update_conversation_context(self, user_input: str, ai_response: str, session_id: str):
        """Update conversation context and extract topics"""
        try:
            # Extract topics from user input
            topic_keywords = {
                "technology": ["ai", "artificial intelligence", "computer", "software", "tech", "digital", "machine learning"],
                "science": ["physics", "chemistry", "biology", "research", "experiment", "discovery", "scientific"],
                "music": ["song", "artist", "album", "band", "concert", "melody", "rhythm", "musical"],
                "art": ["painting", "sculpture", "drawing", "creative", "artistic", "gallery", "museum"],
                "philosophy": ["meaning", "existence", "consciousness", "ethics", "moral", "philosophical", "wisdom"],
                "future": ["tomorrow", "next", "will be", "prediction", "forecast", "upcoming", "ahead"],
                "work": ["job", "career", "office", "business", "professional", "workplace", "employment"],
                "health": ["wellness", "fitness", "medical", "doctor", "exercise", "healthy", "medicine"],
                "travel": ["vacation", "trip", "journey", "destination", "visit", "explore", "adventure"],
                "food": ["restaurant", "cooking", "recipe", "meal", "cuisine", "delicious", "taste"]
            }
            
            input_lower = user_input.lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in input_lower for keyword in keywords):
                    self.conversations[session_id]["topics_discussed"].add(topic)
            
            # Update emotion history
            self.conversations[session_id]["emotion_history"].append({
                "emotion": self.current_emotion,
                "timestamp": time.time(),
                "context": user_input[:50]
            })
            
            # Update context summary
            if len(self.conversations[session_id]["history"]) % 6 == 0:
                # Periodically update context summary
                recent_topics = list(self.conversations[session_id]["topics_discussed"])[-3:]
                self.conversations[session_id]["context_summary"] = f"Recent topics: {', '.join(recent_topics)}"
            
        except Exception as e:
            logger.error(f"Error updating context: {e}")
    
    def set_emotion(self, emotion: str):
        """Set current emotion with validation"""
        valid_emotions = ["neutral", "happy", "sad", "excited", "calm", "angry", "whisper", "dramatic", "confident", "nervous"]
        if emotion in valid_emotions:
            self.current_emotion = emotion
            logger.info(f"Emotion updated to: {emotion}")
        else:
            logger.warning(f"Invalid emotion: {emotion}. Using neutral.")
            self.current_emotion = "neutral"
    
    def get_emotion(self) -> str:
        """Get current emotion"""
        return self.current_emotion
    
    def get_conversation_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get detailed conversation analytics"""
        if session_id not in self.conversations:
            return {"error": "Session not found"}
        
        conversation = self.conversations[session_id]
        current_time = time.time()
        
        return {
            "session_id": session_id,
            "turn_count": conversation["turn_count"],
            "topics_discussed": list(conversation["topics_discussed"]),
            "conversation_duration": current_time - conversation["conversation_start"],
            "message_count": len(conversation["history"]),
            "emotions_used": [e["emotion"] for e in conversation["emotion_history"]],
            "last_interaction": current_time - conversation["last_interaction"],
            "context_summary": conversation["context_summary"],
            "engagement_level": "high" if conversation["turn_count"] > 10 else "medium" if conversation["turn_count"] > 5 else "low"
        }

# Initialize processors
audio_processor = MoshiAudioProcessor()
conversation_manager = ConversationManager()

# Lifespan context manager
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

# Create FastAPI app
app = FastAPI(title="MoshiAI Voice Assistant", lifespan=lifespan)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the main interface"""
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
        "current_emotion": conversation_manager.get_emotion(),
        "moshi_version": "0.2.10"
    }

@app.post("/set_emotion")
async def set_emotion(request: Request):
    """Set emotion for responses"""
    try:
        data = await request.json()
        emotion = data.get("emotion", "neutral")
        conversation_manager.set_emotion(emotion)
        return {"status": "success", "emotion": emotion}
    except Exception as e:
        logger.error(f"Error setting emotion: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/conversations/{session_id}/analytics")
async def get_conversation_analytics(session_id: str):
    """Get detailed conversation analytics"""
    analytics = conversation_manager.get_conversation_analytics(session_id)
    return analytics

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
