from qwen_tts import Qwen3TTSModel
import whisper_timestamped as whisper

# Download and cache model weights to disk.
# Note: we do NOT convert dtype here — that happens at runtime via .half().cuda()
# in handler.py, which reliably converts ALL sub-models regardless of kwarg support.

print("--- ⬇️ Downloading Qwen VoiceDesign ---")
Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")

print("--- ⬇️ Downloading Qwen CustomVoice ---")
Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")

print("--- ⬇️ Downloading Qwen Base (Clone) ---")
Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

print("--- ⬇️ Downloading Whisper ---")
whisper.load_model("base", device="cpu")

print("✅ All models downloaded and cached!")
