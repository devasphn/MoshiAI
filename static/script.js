class EnhancedUnmuteVoiceAssistant {
    constructor() {
        this.ws = null;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioContext = null;
        this.analyzer = null;
        this.audioChunks = [];
        this.currentEmotion = 'neutral';
        this.performanceMetrics = {
            totalRequests: 0,
            successfulRequests: 0,
            turnCount: 0
        };
        this.isVisualizing = false;
        this.microphonePermission = false;
        
        this.init();
    }
    
    init() {
        this.setupWebSocket();
        this.setupEventListeners();
        this.setupAudioVisualization();
        this.requestMicrophonePermission();
        this.checkStatus();
    }
    
    async requestMicrophonePermission() {
        try {
            console.log('🎤 Requesting microphone permission...');
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            stream.getTracks().forEach(track => track.stop());
            
            this.microphonePermission = true;
            console.log('✅ Microphone permission granted');
            this.updateStatus('Ready', 'connected');
            
        } catch (error) {
            console.error('❌ Microphone permission denied:', error);
            this.microphonePermission = false;
            this.updateStatus('Microphone Access Denied', 'error');
            this.showError('Microphone access is required. Please allow microphone access and refresh the page.');
        }
    }
    
    setupWebSocket() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
        
        console.log('🔌 Connecting to Enhanced WebSocket:', wsUrl);
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('✅ Enhanced WebSocket connected');
            if (this.microphonePermission) {
                this.updateStatus('Connected', 'connected');
            }
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('📨 Received enhanced message:', data.type);
            this.handleMessage(data);
        };
        
        this.ws.onclose = () => {
            console.log('🔌 Enhanced WebSocket disconnected');
            this.updateStatus('Disconnected', 'disconnected');
            setTimeout(() => this.setupWebSocket(), 3000);
        };
        
        this.ws.onerror = (error) => {
            console.error('❌ Enhanced WebSocket error:', error);
            this.updateStatus('Connection Error', 'error');
        };
    }
    
    setupEventListeners() {
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');
        
        startBtn.addEventListener('click', () => {
            if (!this.microphonePermission) {
                this.showError('Microphone permission required. Please refresh and allow access.');
                return;
            }
            this.startRecording();
        });
        
        stopBtn.addEventListener('click', () => {
            this.stopRecording();
        });
        
        document.getElementById('emotion-select').addEventListener('change', (e) => {
            this.setEmotion(e.target.value);
        });
        
        const clearBtn = document.getElementById('clear-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.clearChat();
            });
        }
        
        const statsBtn = document.getElementById('stats-btn');
        if (statsBtn) {
            statsBtn.addEventListener('click', () => {
                this.showStatistics();
            });
        }
    }
    
    setupAudioVisualization() {
        this.canvas = document.getElementById('audio-canvas');
        if (!this.canvas) {
            console.error('❌ Audio canvas not found');
            return;
        }
        
        this.canvasCtx = this.canvas.getContext('2d');
        this.canvas.width = 800;
        this.canvas.height = 100;
        
        this.drawVisualization();
    }
    
    async startRecording() {
        if (!this.microphonePermission) {
            this.showError('Microphone permission required');
            return;
        }
        
        try {
            console.log('🎤 Starting enhanced recording...');
            
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 16000
                });
            }
            
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            this.analyzer = this.audioContext.createAnalyser();
            this.analyzer.fftSize = 2048;
            this.analyzer.minDecibels = -90;
            this.analyzer.maxDecibels = -10;
            this.analyzer.smoothingTimeConstant = 0.85;
            
            const source = this.audioContext.createMediaStreamSource(stream);
            source.connect(this.analyzer);
            
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    console.log('📊 Enhanced audio data available:', event.data.size, 'bytes');
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                console.log('🛑 Enhanced recording stopped, processing audio...');
                this.processAudio();
            };
            
            this.mediaRecorder.start(1000);
            this.isRecording = true;
            
            document.getElementById('start-btn').disabled = true;
            document.getElementById('stop-btn').disabled = false;
            document.getElementById('start-btn').classList.add('recording');
            
            this.startVisualization();
            
            console.log('✅ Enhanced recording started successfully');
            
        } catch (error) {
            console.error('❌ Enhanced recording error:', error);
            this.showError('Could not start recording: ' + error.message);
        }
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            console.log('⏹️ Stopping enhanced recording...');
            
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            document.getElementById('start-btn').disabled = false;
            document.getElementById('stop-btn').disabled = true;
            document.getElementById('start-btn').classList.remove('recording');
            
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
            
            this.stopVisualization();
        }
    }
    
    async processAudio() {
        try {
            console.log('🔄 Processing audio chunks...');
            
            if (this.audioChunks.length === 0) {
                console.warn('⚠️ No audio chunks to process');
                return;
            }
            
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm;codecs=opus' });
            console.log('📦 Audio blob size:', audioBlob.size, 'bytes');
            
            if (audioBlob.size === 0) {
                console.warn('⚠️ Empty audio blob');
                return;
            }
            
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioData = await this.audioContext.decodeAudioData(arrayBuffer);
            const audioArray = audioData.getChannelData(0);
            
            console.log('📊 Audio array length:', audioArray.length, 'samples');
            
            this.sendAudioData(Array.from(audioArray));
            
        } catch (error) {
            console.error('❌ Audio processing error:', error);
            this.showError('Failed to process audio: ' + error.message);
        }
    }
    
    sendAudioData(audioArray) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            console.log('📤 Sending audio data to server...');
            
            this.performanceMetrics.totalRequests++;
            this.updatePerformanceMetrics();
            
            const message = {
                type: 'audio',
                audio: audioArray,
                emotion: this.currentEmotion
            };
            
            try {
                this.ws.send(JSON.stringify(message));
                console.log('✅ Audio data sent successfully');
            } catch (error) {
                console.error('❌ Failed to send audio data:', error);
                this.showError('Failed to send audio data');
            }
        } else {
            console.error('❌ WebSocket not ready, state:', this.ws ? this.ws.readyState : 'null');
            this.showError('WebSocket connection not ready');
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
                this.performanceMetrics.successfulRequests++;
                this.performanceMetrics.turnCount++;
                this.updatePerformanceMetrics();
                
                if (data.response_time) {
                    this.updateResponseTime(data.response_time);
                }
                
                if (data.audio && data.audio.length > 0) {
                    this.playAudio(data.audio);
                }
                break;
                
            case 'error':
                console.error('❌ Server error:', data.message);
                this.showError('Server error: ' + data.message);
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
        if (!conversation) {
            console.error('❌ Conversation element not found');
            return;
        }
        
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
        
        conversation.scrollTop = conversation.scrollHeight;
        
        console.log('💬 Message added:', sender, text);
    }
    
    async playAudio(audioArray) {
        try {
            console.log('🔊 Playing audio response...');
            
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
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
            this.showError('Audio playback failed: ' + error.message);
        }
    }
    
    setEmotion(emotion) {
        this.currentEmotion = emotion;
        const emotionElement = document.getElementById('current-emotion');
        if (emotionElement) {
            emotionElement.textContent = emotion;
        }
        
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'emotion',
                emotion: emotion
            }));
        }
        
        console.log('😊 Emotion set to:', emotion);
    }
    
    clearChat() {
        const conversation = document.getElementById('conversation');
        if (conversation) {
            conversation.innerHTML = '';
        }
        
        this.performanceMetrics = {
            totalRequests: 0,
            successfulRequests: 0,
            turnCount: 0
        };
        this.updatePerformanceMetrics();
        
        console.log('🗑️ Chat cleared');
    }
    
    showStatistics() {
        const stats = {
            totalRequests: this.performanceMetrics.totalRequests,
            successfulRequests: this.performanceMetrics.successfulRequests,
            turnCount: this.performanceMetrics.turnCount,
            successRate: this.performanceMetrics.totalRequests > 0 ? 
                (this.performanceMetrics.successfulRequests / this.performanceMetrics.totalRequests * 100).toFixed(1) : 0
        };
        
        alert(`📊 Statistics:\n\nTotal Requests: ${stats.totalRequests}\nSuccessful Requests: ${stats.successfulRequests}\nTurn Count: ${stats.turnCount}\nSuccess Rate: ${stats.successRate}%`);
    }
    
    updatePerformanceMetrics() {
        const elements = {
            'turn-count': this.performanceMetrics.turnCount,
            'total-requests': this.performanceMetrics.totalRequests,
            'success-rate': this.performanceMetrics.totalRequests > 0 ? 
                (this.performanceMetrics.successfulRequests / this.performanceMetrics.totalRequests * 100).toFixed(1) + '%' : '100%'
        };
        
        for (const [id, value] of Object.entries(elements)) {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        }
    }
    
    updateResponseTime(responseTime) {
        const responseTimeElement = document.getElementById('response-time');
        if (responseTimeElement) {
            responseTimeElement.textContent = `${responseTime.toFixed(2)}s`;
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
            
            const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
            const level = Math.round((average / 255) * 100);
            const audioLevelElement = document.getElementById('audio-level');
            if (audioLevelElement) {  // ✅ FIXED: Was "audiogLevelElement"
                audioLevelElement.textContent = `Level: ${level}%`;
            }
        }
    }
    
    drawVisualization(dataArray = null) {
        if (!this.canvas || !this.canvasCtx) return;
        
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        this.canvasCtx.clearRect(0, 0, width, height);
        
        if (dataArray && dataArray.length > 0) {
            this.canvasCtx.fillStyle = '#007bff';
            
            const barWidth = width / (dataArray.length / 4);
            let x = 0;
            
            for (let i = 0; i < dataArray.length / 4; i++) {
                const barHeight = (dataArray[i] / 255) * height;
                
                this.canvasCtx.fillRect(x, height - barHeight, barWidth, barHeight);
                x += barWidth + 1;
            }
        } else {
            this.canvasCtx.fillStyle = '#e9ecef';
            this.canvasCtx.fillRect(0, height / 2 - 1, width, 2);
            
            this.canvasCtx.fillStyle = '#6c757d';
            this.canvasCtx.font = '16px Arial';
            this.canvasCtx.textAlign = 'center';
            this.canvasCtx.fillText('Audio Visualization', width / 2, height / 2 - 10);
        }
    }
    
    updateStatus(status, className) {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            statusElement.textContent = status;
            statusElement.className = `status-value ${className}`;
        }
        
        console.log('📊 Status updated:', status);
    }
    
    async checkStatus() {
        try {
            const response = await fetch('/status');
            const data = await response.json();
            
            console.log('📊 System status:', data);
            
            if (data.status === 'running' && this.microphonePermission) {
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
        
        alert(`❌ Error: ${message}\n\nPlease try:\n1. Refresh the page\n2. Allow microphone access\n3. Check your internet connection`);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing Enhanced Unmute Voice Assistant...');
    
    if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost') {
        alert('⚠️ HTTPS is required for microphone access. Please use HTTPS.');
        return;
    }
    
    if (!window.WebSocket) {
        alert('❌ WebSocket not supported in this browser.');
        return;
    }
    
    if (!window.MediaRecorder) {
        alert('❌ MediaRecorder not supported in this browser.');
        return;
    }
    
    new EnhancedUnmuteVoiceAssistant();
});
