import torch
from qwen_tts import Qwen3TTSModel
import whisper_timestamped as whisper

# ✅ FIXED: Cache models in float16 on CPU.
# Previously models were saved in float32 (default). The handler loads them as
# float16 at runtime, which forced an extra dtype conversion on every cold start.
# Caching in float16 here ensures the cached weights match exactly what the
# handler expects, removing that conversion overhead.

print("--- ⬇️ Downloading Qwen VoiceDesign ---")
Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    torch_dtype=torch.float16,
)

print("--- ⬇️ Downloading Qwen CustomVoice ---")
Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    torch_dtype=torch.float16,
)

print("--- ⬇️ Downloading Qwen Base (Clone) ---")
Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    torch_dtype=torch.float16,
)

print("--- ⬇️ Downloading Whisper ---")
# Whisper is always loaded at runtime with device="cuda", so CPU download is fine.
whisper.load_model("base", device="cpu")

print("✅ All models downloaded and cached!")
