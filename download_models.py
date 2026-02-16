import torch
from qwen_tts import Qwen3TTSModel
import whisper_timestamped as whisper

# ✅ FIX: Use `dtype` instead of `torch_dtype`.
# The Qwen3 TTS library deprecated `torch_dtype` and silently ignores it,
# meaning models were being cached as float32. Using `dtype=torch.float16`
# ensures the cached weights match what the handler loads at runtime.

print("--- ⬇️ Downloading Qwen VoiceDesign ---")
Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    dtype=torch.float16,
)

print("--- ⬇️ Downloading Qwen CustomVoice ---")
Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    dtype=torch.float16,
)

print("--- ⬇️ Downloading Qwen Base (Clone) ---")
Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    dtype=torch.float16,
)

print("--- ⬇️ Downloading Whisper ---")
whisper.load_model("base", device="cpu")

print("✅ All models downloaded and cached!")
