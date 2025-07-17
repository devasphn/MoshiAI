class MoshiAIStreamingClient {
    constructor() {
        this.ws = null;
        this.audioContext = null;
        this.mediaStream = null;
        this.processor = null;
        this.isRecording = false;
        this.isConnected = false;
        this.sessionId = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.audioBuffer = [];
        this.bufferSize = 4096;
        
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
        
        // Add reconnection button
        this.reconnectBtn = document.createElement('button');
        this.reconnectBtn.textContent = '🔄 Reconnect';
        this.reconnectBtn.className = 'btn btn-info';
        this.reconnectBtn.style.display = 'none';
        this.reconnectBtn.addEventListener('click', () => this.connectWebSocket());
        this.clearBtn.parentNode.appendChild(this.reconnectBtn);
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
            timingDiv.textContent = `${timing.total.toFixed(2)}s total (STT: ${timing.stt.toFixed(2)}s)`;
            messageDiv.appendChild(timingDiv);
        }
        
        this.conversationDiv.appendChild(messageDiv);
        this.conversationDiv.scrollTop = this.conversationDiv.scrollHeight;
    }
    
    connectWebSocket() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            return;
        }
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.updateStatus('Connecting...', 'connecting');
        this.reconnectBtn.style.display = 'none';
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.updateStatus('Connected', 'connected');
            this.startBtn.disabled = false;
            this.addMessage('Connected to MoshiAI Streaming. You can start talking!', 'system');
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
            this.reconnectBtn.style.display = 'inline-block';
            
            // Auto-reconnect with exponential backoff
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
                this.reconnectAttempts++;
                setTimeout(() => this.connectWebSocket(), delay);
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('Connection error', 'error');
        };
        
        // Setup ping/pong for keepalive
        setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 25000);
    }
    
    handleMessage(data) {
        if (data.type === 'connection') {
            this.sessionId = data.session_id;
            console.log('Session ID:', this.sessionId);
            return;
        }
        
        if (data.type === 'keepalive' || data.type === 'pong') {
            return;
        }
        
        if (data.error) {
            if (data.error !== 'Could not transcribe audio' && data.error !== 'Duplicate transcription') {
                this.addMessage(`Error: ${data.error}`, 'system');
            }
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
            
            // Use AudioWorklet for better performance
            try {
                await this.audioContext.audioWorklet.addModule('/static/audio-worklet.js');
                this.processor = new AudioWorkletNode(this.audioContext, 'audio-processor');
                
                this.processor.port.onmessage = (event) => {
                    const { audioData, level } = event.data;
                    this.handleAudioData(audioData, level);
                };
                
                source.connect(this.processor);
                
            } catch (workletError) {
                console.warn('AudioWorklet not supported, falling back to ScriptProcessor');
                
                // Fallback to ScriptProcessor
                this.processor = this.audioContext.createScriptProcessor(this.bufferSize, 1, 1);
                
                this.processor.onaudioprocess = (event) => {
                    if (!this.isRecording) return;
                    
                    const inputData = event.inputBuffer.getChannelData(0);
                    const level = Math.sqrt(
                        inputData.reduce((sum, val) => sum + val * val, 0) / inputData.length
                    );
                    
                    this.handleAudioData(Array.from(inputData), level);
                };
                
                source.connect(this.processor);
                this.processor.connect(this.audioContext.destination);
            }
            
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
    
    handleAudioData(audioData, level) {
        if (!this.isRecording) return;
        
        // Update audio level display
        this.audioLevelElement.textContent = `${Math.round(level * 100)}%`;
        
        // Buffer audio data
        this.audioBuffer.push(...audioData);
        
        // Send buffered audio when we have enough data
        if (this.audioBuffer.length >= this.bufferSize) {
            this.sendAudioData(this.audioBuffer);
            this.audioBuffer = [];
        }
    }
    
    sendAudioData(audioData) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'audio',
                audio: audioData
            }));
        }
    }
    
    stopRecording() {
        if (!this.isRecording) return;
        
        this.isRecording = false;
        
        // Send remaining buffered audio
        if (this.audioBuffer.length > 0) {
            this.sendAudioData(this.audioBuffer);
            this.audioBuffer = [];
        }
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
        }
        
        if (this.processor) {
            this.processor.disconnect();
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
        
        // Reset server-side session
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'reset' }));
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new MoshiAIStreamingClient();
});
