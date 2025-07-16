class MoshiVoiceAssistant {
    constructor() {
        this.ws = null;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioContext = null;
        this.analyzer = null;
        this.canvas = null;
        this.canvasCtx = null;
        this.audioChunks = [];
        this.currentEmotion = 'neutral';
        
        this.init();
    }
    
    init() {
        this.setupWebSocket();
        this.setupEventListeners();
        this.setupAudioVisualization();
        this.checkStatus();
    }
    
    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus('Connected', 'connected');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus('Disconnected', 'disconnected');
            // Attempt to reconnect after 3 seconds
            setTimeout(() => this.setupWebSocket(), 3000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus('Error', 'error');
        };
    }
    
    setupEventListeners() {
        // Voice control buttons
        document.getElementById('start-btn').addEventListener('click', () => {
            this.startRecording();
        });
        
        document.getElementById('stop-btn').addEventListener('click', () => {
            this.stopRecording();
        });
        
        // Emotion selector
        document.getElementById('emotion-select').addEventListener('change', (e) => {
            this.setEmotion(e.target.value);
        });
    }
    
    setupAudioVisualization() {
        this.canvas = document.getElementById('audio-canvas');
        this.canvasCtx = this.canvas.getContext('2d');
        
        // Set canvas size
        this.canvas.width = 800;
        this.canvas.height = 100;
        
        this.drawVisualization();
    }
    
    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
            
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.analyzer = this.audioContext.createAnalyser();
            this.analyzer.fftSize = 2048;
            
            const source = this.audioContext.createMediaStreamSource(stream);
            source.connect(this.analyzer);
            
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };
            
            this.mediaRecorder.onstop = () => {
                this.processAudioChunks();
            };
            
            this.mediaRecorder.start();
            this.isRecording = true;
            
            // Update UI
            document.getElementById('start-btn').disabled = true;
            document.getElementById('stop-btn').disabled = false;
            document.getElementById('start-btn').classList.add('recording');
            
            this.startVisualization();
            
        } catch (error) {
            console.error('Error starting recording:', error);
            this.showError('Could not access microphone. Please check permissions.');
        }
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            // Update UI
            document.getElementById('start-btn').disabled = false;
            document.getElementById('stop-btn').disabled = true;
            document.getElementById('start-btn').classList.remove('recording');
            
            // Stop audio tracks
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
            
            this.stopVisualization();
        }
    }
    
    async processAudioChunks() {
        if (this.audioChunks.length === 0) return;
        
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioData = await this.audioContext.decodeAudioData(arrayBuffer);
        
        // Convert to float32 array
        const audioArray = audioData.getChannelData(0);
        
        // Send to server
        this.sendAudioData(Array.from(audioArray));
    }
    
    sendAudioData(audioArray) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            const message = {
                type: 'audio',
                audio: audioArray
            };
            
            this.ws.send(JSON.stringify(message));
        }
    }
    
    setEmotion(emotion) {
        this.currentEmotion = emotion;
        
        // Update UI
        document.getElementById('current-emotion').textContent = emotion;
        
        // Send to server
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            const message = {
                type: 'emotion',
                emotion: emotion
            };
            
            this.ws.send(JSON.stringify(message));
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'transcription':
                this.addMessage(data.text, 'user');
                break;
                
            case 'response':
                this.addMessage(data.text, 'assistant');
                if (data.audio && data.audio.length > 0) {
                    this.playAudio(data.audio);
                }
                break;
                
            case 'emotion_updated':
                document.getElementById('current-emotion').textContent = data.emotion;
                break;
                
            default:
                console.log('Unknown message type:', data.type);
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
    }
    
    async playAudio(audioArray) {
        try {
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            
            const audioBuffer = this.audioContext.createBuffer(1, audioArray.length, 16000);
            const channelData = audioBuffer.getChannelData(0);
            
            for (let i = 0; i < audioArray.length; i++) {
                channelData[i] = audioArray[i];
            }
            
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            source.start();
            
        } catch (error) {
            console.error('Error playing audio:', error);
        }
    }
    
    startVisualization() {
        this.isVisualizing = true;
        this.visualize();
    }
    
    stopVisualization() {
        this.isVisualizing = false;
        this.drawVisualization();
    }
    
    visualize() {
        if (!this.isVisualizing) return;
        
        requestAnimationFrame(() => this.visualize());
        
        if (this.analyzer) {
            const bufferLength = this.analyzer.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            this.analyzer.getByteFrequencyData(dataArray);
            
            this.drawVisualization(dataArray);
        }
    }
    
    drawVisualization(dataArray = null) {
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        this.canvasCtx.clearRect(0, 0, width, height);
        
        if (dataArray) {
            // Draw frequency bars
            this.canvasCtx.fillStyle = '#007bff';
            
            const barWidth = width / dataArray.length * 2;
            let x = 0;
            
            for (let i = 0; i < dataArray.length; i++) {
                const barHeight = (dataArray[i] / 255) * height;
                
                this.canvasCtx.fillRect(x, height - barHeight, barWidth, barHeight);
                x += barWidth + 1;
            }
        } else {
            // Draw placeholder
            this.canvasCtx.fillStyle = '#e9ecef';
            this.canvasCtx.fillRect(0, height / 2 - 1, width, 2);
            
            this.canvasCtx.fillStyle = '#6c757d';
            this.canvasCtx.font = '16px Arial';
            this.canvasCtx.textAlign = 'center';
            this.canvasCtx.fillText('Audio Visualization', width / 2, height / 2 - 10);
        }
    }
    
    updateConnectionStatus(status, className) {
        const statusElement = document.getElementById('connection-status');
        statusElement.textContent = status;
        statusElement.className = `status-value ${className}`;
    }
    
    async checkStatus() {
        try {
            const response = await fetch('/status');
            const data = await response.json();
            
            if (data.status === 'running') {
                this.updateConnectionStatus('Ready', 'connected');
            }
        } catch (error) {
            console.error('Error checking status:', error);
            this.updateConnectionStatus('Error', 'error');
        }
    }
    
    showError(message) {
        this.addMessage(`Error: ${message}`, 'system');
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MoshiVoiceAssistant();
});
