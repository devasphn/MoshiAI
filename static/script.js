class VoiceAssistant {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.isMuted = false;
        this.mediaRecorder = null;
        this.audioContext = null;
        this.currentEmotion = 'happy';
        
        this.initializeElements();
        this.setupEventListeners();
    }
    
    initializeElements() {
        this.startBtn = document.getElementById('startCall');
        this.endBtn = document.getElementById('endCall');
        this.muteBtn = document.getElementById('muteBtn');
        this.statusEl = document.getElementById('connectionStatus');
        this.currentEmotionEl = document.getElementById('currentEmotion');
        this.userTranscriptEl = document.getElementById('userTranscript');
        this.aiTranscriptEl = document.getElementById('aiTranscript');
        this.emotionBtns = document.querySelectorAll('.emotion-btn');
    }
    
    setupEventListeners() {
        this.startBtn.addEventListener('click', () => this.startCall());
        this.endBtn.addEventListener('click', () => this.endCall());
        this.muteBtn.addEventListener('click', () => this.toggleMute());
        
        this.emotionBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const emotion = btn.dataset.emotion;
                this.setEmotion(emotion);
            });
        });
    }
    
    async startCall() {
        try {
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Setup WebSocket connection
            this.ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            this.ws.onopen = () => {
                this.isConnected = true;
                this.updateStatus('Connected');
                this.startBtn.disabled = true;
                this.endBtn.disabled = false;
                this.muteBtn.disabled = false;
            };
            
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleResponse(data);
            };
            
            this.ws.onclose = () => {
                this.isConnected = false;
                this.updateStatus('Disconnected');
                this.startBtn.disabled = false;
                this.endBtn.disabled = true;
                this.muteBtn.disabled = true;
            };
            
            // Setup media recorder
            this.mediaRecorder = new MediaRecorder(stream);
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0 && this.isConnected && !this.isMuted) {
                    this.sendAudioData(event.data);
                }
            };
            
            this.mediaRecorder.start(100); // Send data every 100ms
            
        } catch (error) {
            console.error('Error starting call:', error);
            alert('Could not access microphone. Please check permissions.');
        }
    }
    
    endCall() {
        if (this.ws) {
            this.ws.close();
        }
        
        if (this.mediaRecorder) {
            this.mediaRecorder.stop();
        }
        
        this.isConnected = false;
        this.updateStatus('Disconnected');
        this.startBtn.disabled = false;
        this.endBtn.disabled = true;
        this.muteBtn.disabled = true;
    }
    
    toggleMute() {
        this.isMuted = !this.isMuted;
        this.muteBtn.textContent = this.isMuted ? 'Unmute' : 'Mute';
        this.muteBtn.style.background = this.isMuted ? '#ff9800' : '#2196F3';
    }
    
    async sendAudioData(audioBlob) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            const arrayBuffer = await audioBlob.arrayBuffer();
            this.ws.send(arrayBuffer);
        }
    }
    
    handleResponse(data) {
        if (data.type === 'response') {
            const { user_text, ai_text, emotion } = data.data;
            
            // Update transcripts
            if (user_text) {
                this.userTranscriptEl.querySelector('.text').textContent = user_text;
            }
            
            if (ai_text) {
                this.aiTranscriptEl.querySelector('.text').textContent = ai_text;
            }
            
            // Update current emotion
            if (emotion) {
                this.currentEmotion = emotion;
                this.currentEmotionEl.textContent = emotion;
                this.updateEmotionButtons();
            }
            
            // Play AI response audio
            if (data.data.audio) {
                this.playAudioResponse(data.data.audio);
            }
        }
    }
    
    playAudioResponse(audioData) {
        const audio = new Audio();
        const blob = new Blob([audioData], { type: 'audio/wav' });
        audio.src = URL.createObjectURL(blob);
        audio.play();
    }
    
    async setEmotion(emotion) {
        try {
            const response = await fetch('/set_emotion', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ emotion })
            });
            
            if (response.ok) {
                this.currentEmotion = emotion;
                this.currentEmotionEl.textContent = emotion;
                this.updateEmotionButtons();
            }
        } catch (error) {
            console.error('Error setting emotion:', error);
        }
    }
    
    updateEmotionButtons() {
        this.emotionBtns.forEach(btn => {
