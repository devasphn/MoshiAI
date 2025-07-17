import json
from pathlib import Path

def fix_model_configs():
    """Fix Kyutai model configurations by adding missing model_type fields"""
    models_dir = Path("./models")
    
    print("🔧 Fixing Kyutai model configurations...")
    
    # Fix STT config
    stt_config = models_dir / "stt" / "models--kyutai--stt-1b-en_fr" / "snapshots"
    if stt_config.exists():
        for path in stt_config.rglob("config.json"):
            try:
                with open(path, 'r') as f:
                    config = json.load(f)
                
                if "model_type" not in config:
                    config["model_type"] = "kyutai_speech_to_text"
                    with open(path, 'w') as f:
                        json.dump(config, f, indent=2)
                    print(f"✅ Fixed STT config: {path}")
                else:
                    print(f"✅ STT config already has model_type: {path}")
            except Exception as e:
                print(f"❌ Error fixing STT config {path}: {e}")
    else:
        print("⚠️  STT model directory not found")
    
    # Fix TTS config
    tts_config = models_dir / "tts" / "models--kyutai--tts-1.6b-en_fr" / "snapshots"
    if tts_config.exists():
        for path in tts_config.rglob("config.json"):
            try:
                with open(path, 'r') as f:
                    config = json.load(f)
                
                if "model_type" not in config:
                    config["model_type"] = "kyutai_tts"
                    with open(path, 'w') as f:
                        json.dump(config, f, indent=2)
                    print(f"✅ Fixed TTS config: {path}")
                else:
                    print(f"✅ TTS config already has model_type: {path}")
            except Exception as e:
                print(f"❌ Error fixing TTS config {path}: {e}")
    else:
        print("⚠️  TTS model directory not found")
    
    # Fix LLM config
    llm_config = models_dir / "llm" / "models--kyutai--moshika-pytorch-bf16" / "snapshots"
    if llm_config.exists():
        for path in llm_config.rglob("config.json"):
            try:
                with open(path, 'r') as f:
                    config = json.load(f)
                
                if "model_type" not in config:
                    config["model_type"] = "moshi"
                    with open(path, 'w') as f:
                        json.dump(config, f, indent=2)
                    print(f"✅ Fixed LLM config: {path}")
                else:
                    print(f"✅ LLM config already has model_type: {path}")
            except Exception as e:
                print(f"❌ Error fixing LLM config {path}: {e}")
    else:
        print("⚠️  LLM model directory not found")
    
    print("🎉 Configuration fixing complete!")

if __name__ == "__main__":
    fix_model_configs()
