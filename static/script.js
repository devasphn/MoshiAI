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
        this.permissionRequested = false;
        
        this.init();
    }
    
    init() {
        this.setupWebSocket();
        this.setupEventListeners();
        this.setupAudioVisualization();
        this.checkStatus();
        // DON'T request microphone permission here - it must be user-triggered
    }
    
    setupWebSocket() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
        
        console.log('🔌 Connecting to Enhanced WebSocket:', wsUrl);
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('✅ Enhanced WebSocket connected');
            this.updateStatus('Connected', 'connected');
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
        
        if (startBtn) {
            startBtn.addEventListener('click', async () => {
                console.log('🎤 Start button clicked');
                
                // Request microphone permission when user clicks start
                if (!this.permissionRequested || !this.microphonePermission) {
                    await this.requestMicrophonePermission();
                }
                
                if (this.microphonePermission) {
                    this.startRecording();
                } else {
                    this.showError('Microphone permission is required to use voice features.');
                }
            });
        }
        
        if (stopBtn) {
            stopBtn.addEventListener('click', () => {
                console.log('🛑 Stop button clicked');
                this.stopRecording();
            });
        }
        
        const emotionSelect = document.getElementById('emotion-select');
        if (emotionSelect) {
            emotionSelect.addEventListener('change', (e) => {
                this.setEmotion(e.target.value);
            });
        }
        
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
    
    async requestMicrophonePermission() {
        this.permissionRequested = true;
        
        try {
            console.log('🎤 Requesting microphone permission...');
            
            // Check if we're in a secure context
            if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost') {
                throw new Error('HTTPS is required for microphone access');
            }
            
            // Check if getUserMedia is supported
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('getUserMedia is not supported in this browser');
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
            
            // Stop the test stream immediately
            stream.getTracks().forEach(track => track.stop());
            
            this.microphonePermission = true;
            console.log('✅ Microphone permission granted');
            this.updateStatus('Microphone Ready', 'connected');
            
            return true;
            
        } catch (error) {
            console.error('❌ Microphone permission denied:', error);
            this.microphonePermission = false;
            this.updateStatus('Microphone Access Denied', 'error');
            
            let errorMessage = 'Microphone access denied. ';
            if (error.name === 'NotAllowedError') {
                errorMessage += 'Please allow microphone access and try again.';
            } else if (error.name === 'NotFoundError') {
                errorMessage += 'No microphone found. Please check your audio devices.';
            } else if (error.name === 'NotSupportedError') {
                errorMessage += 'Microphone not supported in this browser.';
            } else {
                errorMessage += error.message;
            }
            
            this.showError(errorMessage);
            return false;
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
            console.error('❌ No microphone permission');
            this.showError('Microphone permission required');
            return;
        }
        
        try {
            console.log('🎤 Starting enhanced recording...');
            
            // Create AudioContext if needed
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 16000
                });
            }
            
            // Resume AudioContext if suspended
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
                console.log('🔊 AudioContext resumed');
            }
            
            // Get media stream
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            console.log('📡 Media stream obtained');
            
            // Set up audio analyzer
            this.analyzer = this.audioContext.createAnalyser();
            this.analyzer.fftSize = 2048;
            this.analyzer.minDecibels = -90;
            this.analyzer.maxDecibels = -10;
            this.analyzer.smoothingTimeConstant = 0.85;
            
            const source = this.audioContext.createMediaStreamSource(stream);
            source.connect(this.analyzer);
            
            console.log('🎵 Audio analyzer connected');
            
            // Set up media recorder
            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 
                'audio/webm;codecs=opus' : 'audio/webm';
            
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: mimeType
            });
            
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    console.log('📊 Audio chunk received:', event.data.size, 'bytes');
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                console.log('🛑 MediaRecorder stopped, processing audio...');
                this.processAudio();
            };
            
            this.mediaRecorder.onerror = (event) => {
                console.error('❌ MediaRecorder error:', event.error);
                this.showError('Recording error: ' + event.error);
            };
            
            // Start recording
            this.mediaRecorder.start(1000); // Record in 1-second chunks
            this.isRecording = true;
            
            // Update UI
            const startBtn = document.getElementById('start-btn');
            const stopBtn = document.getElementById('stop-btn');
            
            if (startBtn) {
                startBtn.disabled = true;
                startBtn.classList.add('recording');
                startBtn.textContent = '🎤 Recording...';
            }
            
            if (stopBtn) {
                stopBtn.disabled = false;
            }
            
            // Start visualization
            this.startVisualization();
            
            console.log('✅ Recording started successfully');
            
        } catch (error) {
            console.error('❌ Recording start error:', error);
            this.showError('Could not start recording: ' + error.message);
            this.resetRecordingUI();
        }
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            console.log('⏹️ Stopping recording...');
            
            try {
                this.mediaRecorder.stop();
                this.isRecording = false;
                
                // Stop all tracks
                if (this.mediaRecorder.stream) {
                    this.mediaRecorder.stream.getTracks().forEach(track => {
                        track.stop();
                        console.log('🔇 Audio track stopped');
                    });
                }
                
                this.resetRecordingUI();
                this.stopVisualization();
                
                console.log('✅ Recording stopped successfully');
                
            } catch (error) {
                console.error('❌ Error stopping recording:', error);
                this.showError('Error stopping recording: ' + error.message);
            }
        }
    }
    
    resetRecordingUI() {
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');
        
        if (startBtn) {
            startBtn.disabled = false;
            startBtn.classList.remove('recording');
            startBtn.textContent = '🎤 Start Talking';
        }
        
        if (stopBtn) {
            stopBtn.disabled = true;
        }
    }
    
    async processAudio() {
        try {
            console.log('🔄 Processing audio chunks...');
            
            if (this.audioChunks.length === 0) {
                console.warn('⚠️ No audio chunks to process');
                return;
            }
            
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
            console.log('📦 Audio blob created:', audioBlob.size, 'bytes');
            
            if (audioBlob.size === 0) {
                console.warn('⚠️ Empty audio blob');
                return;
            }
            
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioData = await this.audioContext.decodeAudioData(arrayBuffer);
            const audioArray = audioData.getChannelData(0);
            
            console.log('📊 Audio processed:', audioArray.length, 'samples');
            
            // Check if audio has sufficient energy
            const energy = audioArray.reduce((sum, sample) => sum + sample * sample, 0) / audioArray.length;
            console.log('🔊 Audio energy level:', energy);
            
            if (energy < 0.00001) {
                console.warn('⚠️ Audio energy too low, might be silence');
                this.showError('No audio detected. Please speak louder.');
                return;
            }
            
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
                this.showError('Failed to send audio data: ' + error.message);
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
            if (audioLevelElement) {
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
            this.canvasCtx.fillText('Click "Start Talking" to begin', width / 2, height / 2 - 10);
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
            
            if (data.status === 'running') {
                this.updateStatus('Ready - Click Start', 'connected');
            }
        } catch (error) {
            console.error('❌ Status check error:', error);
            this.updateStatus('Error', 'error');
        }
    }
    
    showError(message) {
        console.error('❌ Error:', message);
        this.addMessage(`Error: ${message}`, 'system');
        
        // Show user-friendly error popup
        const errorDialog = document.createElement('div');
        errorDialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 1000;
            max-width: 400px;
            text-align: center;
        `;
        
        errorDialog.innerHTML = `
            <h3>❌ Error</h3>
            <p>${message}</p>
            <button onclick="this.parentElement.remove()" style="margin-top: 10px; padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer;">OK</button>
        `;
        
        document.body.appendChild(errorDialog);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (errorDialog && errorDialog.parentElement) {
                errorDialog.remove();
            }
        }, 5000);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing Enhanced Unmute Voice Assistant...');
    
    // Check browser compatibility
    if (!window.WebSocket) {
        alert('❌ WebSocket not supported in this browser.');
        return;
    }
    
    if (!window.MediaRecorder) {
        alert('❌ MediaRecorder not supported in this browser.');
        return;
    }
    
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('❌ getUserMedia not supported. Please use a modern browser with HTTPS.');
        return;
    }
    
    // Initialize the voice assistant
    new EnhancedUnmuteVoiceAssistant();
});
