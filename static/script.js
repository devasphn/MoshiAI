class UnmuteVoiceAssistant {
    constructor() {
        this.ws = null;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioContext = null;
        this.analyzer = null;
        this.audioChunks = [];
        
        this.init();
    }
    
    init() {
        this.setupWebSocket();
        this.setupEventListeners();
        this.checkStatus();
    }
    
    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        console.log('🔌 Connecting to WebSocket:', wsUrl);
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('✅ WebSocket connected');
            this.updateStatus('Connected', 'connected');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('📨 Received message:', data.type);
            this.handleMessage(data);
        };
        
        this.ws.onclose = () => {
            console.log('🔌 WebSocket disconnected');
            this.updateStatus('Disconnected', 'disconnected');
            setTimeout(() => this.setupWebSocket(), 3000);
        };
        
        this.ws.onerror = (error) => {
            console.error('❌ WebSocket error:', error);
            this.updateStatus('Error', 'error');
        };
    }
    
    setupEventListeners() {
        document.getElementById('start-btn').addEventListener('click', () => {
            this.startRecording();
        });
        
        document.getElementById('stop-btn').addEventListener('click', () => {
            this.stopRecording();
        });
    }
    
    async startRecording() {
        try {
            console.log('🎤 Starting recording...');
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
            
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                console.log('📊 Audio data available:', event.data.size, 'bytes');
                this.audioChunks.push(event.data);
            };
            
            this.mediaRecorder.onstop = () => {
                console.log('🛑 Recording stopped, processing audio...');
                this.processAudio();
            };
            
            this.mediaRecorder.start();
            this.isRecording = true;
            
            // Update UI
            document.getElementById('start-btn').disabled = true;
            document.getElementById('stop-btn').disabled = false;
            
            console.log('✅ Recording started successfully');
            
        } catch (error) {
            console.error('❌ Recording error:', error);
            this.showError('Could not access microphone');
        }
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            console.log('⏹️ Stopping recording...');
            
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            // Update UI
            document.getElementById('start-btn').disabled = false;
            document.getElementById('stop-btn').disabled = true;
            
            // Stop audio tracks
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    }
    
    async processAudio() {
        try {
            console.log('🔄 Processing audio chunks...');
            
            if (this.audioChunks.length === 0) {
                console.warn('⚠️ No audio chunks to process');
                return;
            }
            
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
            console.log('📦 Audio blob size:', audioBlob.size, 'bytes');
            
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioData = await this.audioContext.decodeAudioData(arrayBuffer);
            
            // Convert to float32 array
            const audioArray = audioData.getChannelData(0);
            console.log('📊 Audio array length:', audioArray.length, 'samples');
            
            // Send to server
            this.sendAudioData(Array.from(audioArray));
            
        } catch (error) {
            console.error('❌ Audio processing error:', error);
            this.showError('Failed to process audio');
        }
    }
    
    sendAudioData(audioArray) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            console.log('📤 Sending audio data to server...');
            
            const message = {
                type: 'audio',
                audio: audioArray
            };
            
            this.ws.send(JSON.stringify(message));
            console.log('✅ Audio data sent successfully');
        } else {
            console.error('❌ WebSocket not ready');
        }
    }
    
    handleMessage(data) {
        console.log('📨 Handling message:', data.type);
        
        switch (data.type) {
            case 'transcription':
                console.log('🎤 Transcription:', data.text);
                this.addMessage(data.text, 'user');
                break;
                
            case 'response':
                console.log('🤖 AI Response:', data.text);
                this.addMessage(data.text, 'assistant');
                if (data.audio && data.audio.length > 0) {
                    this.playAudio(data.audio);
                }
                break;
                
            case 'error':
                console.error('❌ Server error:', data.message);
                this.showError(data.message);
                break;
                
            case 'pong':
                console.log('🏓 Pong received');
                break;
                
            default:
                console.warn('⚠️ Unknown message type:', data.type);
        }
    }
    
    addMessage(text, sender) {
        const conversation = document.getElementById('conversation');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = text;
        
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = new Date().toLocaleTimeString();
        
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timeDiv);
        conversation.appendChild(messageDiv);
        
        // Scroll to bottom
        conversation.scrollTop = conversation.scrollHeight;
        
        console.log('💬 Message added:', sender, text);
    }
    
    async playAudio(audioArray) {
        try {
            console.log('🔊 Playing audio response...');
            
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            
            const audioBuffer = this.audioContext.createBuffer(1, audioArray.length, 24000);
            const channelData = audioBuffer.getChannelData(0);
            
            for (let i = 0; i < audioArray.length; i++) {
                channelData[i] = audioArray[i];
            }
            
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            source.start();
            
            console.log('✅ Audio playback started');
            
        } catch (error) {
            console.error('❌ Audio playback error:', error);
        }
    }
    
    updateStatus(status, className) {
        const statusElement = document.getElementById('connection-status');
        statusElement.textContent = status;
        statusElement.className = `status-value ${className}`;
        
        console.log('📊 Status updated:', status);
    }
    
    async checkStatus() {
        try {
            const response = await fetch('/status');
            const data = await response.json();
            
            console.log('📊 System status:', data);
            
            if (data.status === 'running') {
                this.updateStatus('Ready', 'connected');
            }
        } catch (error) {
            console.error('❌ Status check error:', error);
            this.updateStatus('Error', 'error');
        }
    }
    
    showError(message) {
        console.error('❌ Error:', message);
        this.addMessage(`Error: ${message}`, 'system');
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing Unmute Voice Assistant...');
    new UnmuteVoiceAssistant();
});
