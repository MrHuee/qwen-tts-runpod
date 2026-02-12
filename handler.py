import runpod
import torch
import scipy.io.wavfile
import io
import base64
import gc
import os
import tempfile
import json
# Import TTS
from qwen_tts import Qwen3TTSModel
# Import Whisper
import whisper_timestamped as whisper

# --- Configuration ---
MODEL_IDS = {
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_clone":  "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
}

# --- Global State ---
CURRENT_MODEL = None
CURRENT_MODE = None
WHISPER_MODEL = None # We keep Whisper loaded separately if needed, or load on demand

def load_tts_model(target_mode):
    global CURRENT_MODEL, CURRENT_MODE
    
    # Check if we need to swap TTS models
    if CURRENT_MODE == target_mode and CURRENT_MODEL is not None:
        return CURRENT_MODEL

    # Clean up previous TTS model
    if CURRENT_MODEL is not None:
        del CURRENT_MODEL
        CURRENT_MODEL = None
        gc.collect()
        torch.cuda.empty_cache()

    print(f"--- 🚀 Loading TTS: {target_mode} ---")
    model_id = MODEL_IDS.get(target_mode)
    try:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = Qwen3TTSModel.from_pretrained(
            model_id, device_map="cuda", dtype=dtype, attn_implementation="flash_attention_2"
        )
    except:
        model = Qwen3TTSModel.from_pretrained(model_id, device_map="cuda", dtype=torch.float16)
        
    CURRENT_MODEL = model
    CURRENT_MODE = target_mode
    return model

def get_whisper_model():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        print("--- 🎙️ Loading Whisper Model (Base) ---")
        # 'base' is fast and good. Use 'medium' for higher accuracy.
        WHISPER_MODEL = whisper.load_model("base", device="cuda")
    return WHISPER_MODEL

def handler(job):
    job_input = job["input"]
    mode = job_input.get("mode", "voice_design").lower()
    
    # --- MODE: TRANSCRIPTION (Get Timestamps) ---
    if mode == "transcribe":
        try:
            audio_base64 = job_input.get("audio_base64")
            if not audio_base64: return {"error": "No audio_base64 provided"}

            # 1. Decode Audio to Temp File
            # Whisper expects a file path usually
            audio_bytes = base64.b64decode(audio_base64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name

            # 2. Run Whisper Timestamped
            model = get_whisper_model()
            audio = whisper.load_audio(temp_audio_path)
            result = whisper.transcribe(model, audio, language="en")

            # 3. Cleanup
            os.remove(temp_audio_path)

            return {
                "status": "success",
                "transcription": result # This contains the word-level JSON
            }
        except Exception as e:
            return {"error": str(e)}

    # --- MODE: TTS GENERATION ---
    text = job_input.get("text")
    language = job_input.get("language", "English")

    if not text: return {"error": "No text provided."}

    try:
        model = load_tts_model(mode)
        wavs, sr = None, None

        if mode == "voice_design":
            instruct = job_input.get("instruct", "Clear voice.")
            wavs, sr = model.generate_voice_design(text=text, language=language, instruct=instruct)
        elif mode == "custom_voice":
            speaker = job_input.get("speaker", "Anna")
            wavs, sr = model.generate_custom_voice(text=text, language=language, speaker=speaker)
        elif mode == "voice_clone":
            ref_audio = job_input.get("ref_audio")
            ref_text = job_input.get("ref_text")
            wavs, sr = model.generate_voice_clone(text=text, language=language, ref_audio=ref_audio, ref_text=ref_text)

        # Convert to Base64
        audio_array = wavs[0]
        byte_io = io.BytesIO()
        scipy.io.wavfile.write(byte_io, sr, audio_array)
        wav_bytes = byte_io.getvalue()
        audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
        
        return {"status": "success", "mode": mode, "audio_base64": audio_b64}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})