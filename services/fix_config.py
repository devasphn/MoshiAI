import json
from pathlib import Path

def fix_model_configs():
    models_dir = Path("./models")
    
    # Fix STT config
    stt_config = models_dir / "stt" / "models--kyutai--stt-1b-en_fr" / "snapshots"
    for path in stt_config.rglob("config.json"):
        with open(path, 'r') as f:
            config = json.load(f)
        
        if "model_type" not in config:
            config["model_type"] = "kyutai_speech_to_text"
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Fixed STT config: {path}")
    
    # Fix TTS config
    tts_config = models_dir / "tts" / "models--kyutai--tts-1.6b-en_fr" / "snapshots"
    for path in tts_config.rglob("config.json"):
        with open(path, 'r') as f:
            config = json.load(f)
        
        if "model_type" not in config:
            config["model_type"] = "kyutai_tts"
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Fixed TTS config: {path}")

if __name__ == "__main__":
    fix_model_configs()
