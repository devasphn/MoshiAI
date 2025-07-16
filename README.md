# MoshiAI Voice Assistant

This is a real-time conversational AI voice agent built with Kyutai's Moshi AI. It supports ultra-low latency speech-to-speech interactions, 70+ emotional speaking styles (e.g., happy, sad, whispering, giggling), voice cloning (using indian_female.wav), live transcription, and a simple web UI with start/end call, mute, and emotion selection buttons.

## Features
- Real-time voice chat via WebSockets (no TURN server needed for basic use).
- Emotional expressions: Applied naturally (e.g., "*giggles* That's funny!" or "*sighs* I'm sad.") without overusing in every sentence.
- Voice cloning: Uses indian_female.wav for an Indian female voice.
- UI: Start/End call, mute, emotion picker, and live user/AI transcription.
- Runs on GPU (e.g., RTX 4090 on Runpod) for low latency (~200ms).
- Open-source: Based on kyutai-labs/moshi and kyutai-labs/unmute.

## Setup Instructions
1. Clone this repo: `git clone https://github.com/devasphn/MoshiAI.git`
2. Download indian_female.wav: `wget https://huggingface.co/datasets/ai4bharat/indic-tts/resolve/main/samples/indictts_tamil_female.wav -O voices/indian_female.wav`
3. Install dependencies (see below).
4. Run: `python app.py`
5. Access the UI at http://localhost:8000 (or your Runpod IP:8000).

## Dependencies
See requirements.txt. Install via `pip install -r requirements.txt`.

## Run on Runpod
- Select a pod with RTX 4090 (24GB VRAM min), 24GB container memory, 50GB volume.
- Expose port 8000 (HTTP).
- Follow the commands in your query history for venv, apt installs, and running app.py.

## Usage
- Click "Start Call" to begin (grants mic access).
- Speak naturally; AI responds with emotions.
- Change emotions via buttons (e.g., say "speak with happy" or click).
- Transcripts show user/AI speech in real-time.
- Mute/End as needed.

## License
MIT License (compatible with Moshi's CC-BY 4.0 for models).

For issues, check logs in app.py or Moshi docs: https://github.com/kyutai-labs/moshi.
