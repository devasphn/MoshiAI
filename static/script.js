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
        
        document.getElementById('emotion-select').addEventListener('change', (e) => {
            this.setEmotion(e.target.value);
        });
        
        document.getElementById('clear-btn').addEventListener('click', () => {
            this.clearChat();
        });
        
        document.getElementById('stats-btn').addEventListener('click', () => {
            this.showStatistics();
        });
    }
    
    setupAudioVisualization() {
        this.canvas = document.getElementById('audio-canvas');
        this.canvasCtx = this.canvas.getContext('2d');
        
        this.canvas.width = 800;
        this.canvas.height = 100;
        
        this.drawVisualization();
    }
    
    async startRecording() {
        try {
            console.log('🎤 Starting enhanced recording...');
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
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
                console.log('📊 Enhanced audio data available:', event.data.size, 'bytes');
                this.audioChunks.push(event.data);
            };
            
            this.mediaRecorder.onstop = () => {
                console.log('🛑 Enhanced recording stopped, processing audio...');
                this.processAudio();
            };
            
            this.mediaRecorder.start();
            this.isRecording = true;
            
            // Update UI
            document.getElementById('start-btn').disabled = true;
            document.getElementById('stop-btn').disabled = false;
            document.getElementById('start-btn').classList.add('recording');
            
            this.startVisualization();
            
            console.log('✅ Enhanced recording started successfully');
            
        } catch (error) {
            console.error('❌ Enhanced recording error:', error);
            this.showError('Could not access microphone. Please check permissions.');
        }
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            console.log('⏹️ Stopping enhanced recording...');
            
            this.mediaRecorder.stop();
            this.isRec
