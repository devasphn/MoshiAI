document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing Enhanced Unmute Voice Assistant...');

    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const statusElement = document.getElementById('connection-status');
    const conversationDiv = document.getElementById('conversation');
    const responseTimeElement = document.getElementById('response-time');
    const audioLevelElement = document.getElementById('audio-level');

    let ws;
    let audioContext;
    let scriptProcessor;
    let mediaStream;
    let isRecording = false;

    function updateStatus(status, className) {
        statusElement.textContent = status;
        statusElement.className = `status-value ${className}`;
        console.log(`📊 Status updated: ${status}`);
    }

    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = text;
        messageDiv.appendChild(contentDiv);
        conversationDiv.appendChild(messageDiv);
        conversationDiv.scrollTop = conversationDiv.scrollHeight;
    }

    function connectWebSocket() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
        
        updateStatus('Connecting...', 'disconnected');
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            updateStatus('Connected', 'connected');
            startBtn.disabled = false;
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        };

        ws.onclose = () => {
            updateStatus('Disconnected', 'disconnected');
            isRecording = false;
            startBtn.disabled = true;
            stopBtn.disabled = true;
            setTimeout(connectWebSocket, 3000); // Attempt to reconnect
        };

        ws.onerror = (error) => {
            console.error('❌ WebSocket Error:', error);
            updateStatus('Connection Error', 'error');
        };
    }

    function handleWebSocketMessage(data) {
        console.log('📨 Received message:', data.type);
        switch (data.type) {
            case 'transcription':
                addMessage(data.text, 'user');
                break;
            case 'response':
                addMessage(data.text, 'assistant');
                if (data.response_time) {
                    responseTimeElement.textContent = `${data.response_time.toFixed(2)}s`;
                }
                if (data.audio && data.audio.length > 0) {
                    playAudio(data.audio);
                }
                break;
            case 'error':
                addMessage(`Server Error: ${data.message}`, 'system');
                break;
            default:
                console.warn('⚠️ Unknown message type:', data.type);
        }
    }
    
    async function playAudio(audioArray) {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
        }
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }
        
        const audioBuffer = audioContext.createBuffer(1, audioArray.length, 24000);
        audioBuffer.getChannelData(0).set(audioArray);

        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        source.start();
    }

    async function startRecording() {
        if (isRecording) return;
        try {
            updateStatus('Recording...', 'connected');
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });

            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });

            const source = audioContext.createMediaStreamSource(mediaStream);
            scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);

            scriptProcessor.onaudioprocess = (event) => {
                if (!isRecording) return;
                const inputData = event.inputBuffer.getChannelData(0);
                
                // Calculate audio level
                const level = Math.sqrt(inputData.reduce((sum, val) => sum + val * val, 0) / inputData.length);
                audioLevelElement.textContent = `Level: ${Math.round(level * 100)}%`;

                // Send audio data via WebSocket
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'audio', audio: Array.from(inputData) }));
                }
            };
            
            source.connect(scriptProcessor);
            scriptProcessor.connect(audioContext.destination);

            isRecording = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            startBtn.classList.add('recording');

        } catch (error) {
            console.error('❌ Could not start recording:', error);
            addMessage(`Error: ${error.message}. Please allow microphone access.`, 'system');
            updateStatus('Mic Error', 'error');
        }
    }

    function stopRecording() {
        if (!isRecording) return;
        
        isRecording = false;
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
        }
        if (scriptProcessor) {
            scriptProcessor.disconnect();
            scriptProcessor = null;
        }
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
        
        startBtn.disabled = false;
        stopBtn.disabled = true;
        startBtn.classList.remove('recording');
        updateStatus('Connected', 'connected');
        audioLevelElement.textContent = 'Level: 0%';
    }

    // Event Listeners
    startBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);
    document.getElementById('clear-btn').addEventListener('click', () => {
        conversationDiv.innerHTML = '';
        responseTimeElement.textContent = '-';
    });

    // Initial setup
    connectWebSocket();
});
