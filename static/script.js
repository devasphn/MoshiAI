class MoshiAIClient {
    constructor() {
        this.ws = null;
        this.audioContext = null;
        this.mediaStream = null;
        this.scriptProcessor = null;
        this.isRecording = false;
        this.isConnected = false;
        
        this.initializeElements();
        this.connectWebSocket();
    }
    
    initializeElements() {
        this.startBtn = document.getElementById('start-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.clearBtn = document.getElementById('clear-btn');
        this.statusElement = document.getElementById('status');
        this.conversationDiv = document.getElementById('conversation');
        this.audioLevelElement = document.getElementById('audio-level');
        this.responseTimeElement = document.getElementById('response-time');
        
        // Event listeners
        this.startBtn.addEventListener('click', () => this.startRecording());
        this.stopBtn.addEventListener('click', () => this.stopRecording());
        this.clearBtn.addEventListener('click', () => this.clearConversation());
    }
    
    updateStatus(message, type = 'info') {
        this.statusElement.textContent = message;
        this.statusElement.className = `status ${type}`;
        console.log(`[${type.toUpperCase()}] ${message}`);
    }
    
    addMessage(text, sender, timing = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = text;
        messageDiv.appendChild(contentDiv);
        
        if (timing) {
            const timingDiv = document.createElement('div');
            timingDiv.className = 'message-timing';
            timingDiv.textContent = `${timing.total.toFixed(2)}s total`;
            messageDiv.appendChild(timingDiv);
        }
        
        this.conversationDiv.appendChild(messageDiv);
        this.conversationDiv.scrollTop = this.conversationDiv.scrollHeight;
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.updateStatus('Connecting...', 'connecting');
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            this.isConnected = true;
            this.updateStatus('Connected', 'connected');
            this.startBtn.disabled = false;
            this.addMessage('Connected to MoshiAI. You can start talking!', 'system');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
        
        this.ws.onclose = () => {
            this.isConnected = false;
            this.updateStatus('Disconnected', 'error');
            this.startBtn.disabled = true;
            this.stopBtn.disabled = true;
            
            // Reconnect after delay
            setTimeout(() => this.connectWebSocket(), 3000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('Connection error', 'error');
        };
    }
    
    handleMessage(data) {
        if (data.error) {
            this.addMessage(`Error: ${data.error}`, 'system');
            return;
        }
        
        if (data.transcription) {
            this.addMessage(data.transcription, 'user');
        }
        
        if (data.response_text) {
            this.addMessage(data.response_text, 'assistant', data.timing);
            
            if (data.timing) {
                this.responseTimeElement.textContent = `${data.timing.total.toFixed(2)}s`;
            }
        }
        
        if (data.response_audio && data.response_audio.length > 0) {
            this.playAudio(data.response_audio);
        }
    }
    
    async playAudio(audioArray) {
        try {
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }
            
            const audioBuffer = this.audioContext.createBuffer(1, audioArray.length, 24000);
            audioBuffer.getChannelData(0).set(audioArray);
            
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            source.start();
            
        } catch (error) {
            console.error('Audio playback error:', error);
        }
    }
    
    async startRecording() {
        if (this.isRecording || !this.isConnected) return;
        
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000
            });
            
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            this.scriptProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);
            
            this.scriptProcessor.onaudioprocess = (event) => {
                if (!this.isRecording) return;
                
                const inputData = event.inputBuffer.getChannelData(0);
                
                // Calculate audio level
                const level = Math.sqrt(
                    inputData.reduce((sum, val) => sum + val * val, 0) / inputData.length
                );
                this.audioLevelElement.textContent = `${Math.round(level * 100)}%`;
                
                // Send audio data
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({
                        type: 'audio',
                        audio: Array.from(inputData)
                    }));
                }
            };
            
            source.connect(this.scriptProcessor);
            this.scriptProcessor.connect(this.audioContext.destination);
            
            this.isRecording = true;
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.updateStatus('Recording...', 'recording');
            
        } catch (error) {
            console.error('Recording error:', error);
            this.addMessage(`Microphone error: ${error.message}`, 'system');
            this.updateStatus('Microphone error', 'error');
        }
    }
    
    stopRecording() {
        if (!this.isRecording) return;
        
        this.isRecording = false;
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
        }
        
        if (this.scriptProcessor) {
            this.scriptProcessor.disconnect();
        }
        
        if (this.audioContext) {
            this.audioContext.close();
        }
        
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.updateStatus('Connected', 'connected');
        this.audioLevelElement.textContent = '0%';
    }
    
    clearConversation() {
        this.conversationDiv.innerHTML = '';
        this.responseTimeElement.textContent = '-';
        this.addMessage('Conversation cleared', 'system');
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new MoshiAIClient();
});
