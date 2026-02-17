"""
Model download script — runs during Docker build (no GPU available).
Downloads and caches weights to disk in bfloat16 format.
device_map and attn_implementation are omitted here (GPU not present at build time).
"""
import sys
import torch
from qwen_tts import Qwen3TTSModel
import whisper_timestamped as whisper

def download(name, model_id):
    print(f"--- ⬇️ Downloading {name} ---")
    try:
        Qwen3TTSModel.from_pretrained(model_id, dtype=torch.bfloat16, torch_dtype=torch.bfloat16)
        print(f"✅ {name} cached.")
    except Exception as e:
        print(f"❌ Failed to download {name}: {e}")
        sys.exit(1)

download("Qwen3-TTS VoiceDesign", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
download("Qwen3-TTS CustomVoice", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
download("Qwen3-TTS Base (voice clone)", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

print("--- ⬇️ Downloading Whisper base ---")
try:
    whisper.load_model("base", device="cpu")
    print("✅ Whisper cached.")
except Exception as e:
    print(f"❌ Failed to download Whisper: {e}")
    sys.exit(1)

print("✅ All models downloaded and cached!")
