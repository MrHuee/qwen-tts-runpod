import runpod
import torch
import scipy.io.wavfile
import io
import base64
import gc
import os
import tempfile
import numpy as np  # <--- Explicit import fixes "Numpy not available"
from qwen_tts import Qwen3TTSModel
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
WHISPER_MODEL = None 

def load_target_model(target_mode):
    """
    Loads ONLY the requested TTS model. Unloads previous ones to save VRAM.
    """
    global CURRENT_MODEL, CURRENT_MODE

    # 1. Check if the requested model is already loaded
    if CURRENT_MODE == target_mode and CURRENT_MODEL is not None:
        print(f"--- ⚡ Model '{target_mode}' is already loaded. Skipping load. ---")
        return CURRENT_MODEL

    # 2. Unload existing model to free VRAM (Clean Slate)
    if CURRENT_MODEL is not None:
        print("--- 🧹 Unloading previous model... ---")
        del CURRENT_MODEL
        CURRENT_MODEL = None
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Load the new model
    model_id = MODEL_IDS.get(target_mode)
    if not model_id:
        raise ValueError(f"Invalid mode '{target_mode}'. Options: {list(MODEL_IDS.keys())}")

    print(f"--- 🚀 Loading {target_mode} ({model_id}) with Flash Attention 2... ---")
    
    try:
        # Determine best dtype (bfloat16 is best for A100/A10/L4/3090)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        model = Qwen3TTSModel.from_pretrained(
            model_id, 
            device_map="cuda", 
            dtype=dtype,
            attn_implementation="flash_attention_2"
        )
        
        CURRENT_MODEL = model
        CURRENT_MODE = target_mode
        print(f"✅ {target_mode} loaded successfully.")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model with Flash Attention 2. Trying fallback... Error: {e}")
        model = Qwen3TTSModel.from_pretrained(
            model_id, 
            device_map="cuda", 
            dtype=torch.float16
        )
        CURRENT_MODEL = model
        CURRENT_MODE = target_mode
        return model

def get_whisper_model():
    """Lazy loads the Whisper model only when needed."""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        print("--- 🎙️ Loading Whisper Model (Base) ---")
        # 'base' is fast. Change to 'medium' or 'large-v2' for better accuracy.
        WHISPER_MODEL = whisper.load_model("base", device="cuda")
    return WHISPER_MODEL

def handler(job):
    job_input = job["input"]
    
    # Determine Mode
    mode = job_input.get("mode", "voice_design").lower()

    # --- MODE 1: TRANSCRIPTION (Whisper) ---
    if mode == "transcribe":
        try:
            audio_base64 = job_input.get("audio_base64")
            if not audio_base64: return {"error": "No audio_base64 provided"}

            # Decode Audio to Temp File
            try:
                audio_bytes = base64.b64decode(audio_base64)
            except Exception as e:
                return {"error": f"Base64 decode failed: {str(e)}"}

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name

            # Run Whisper
            model = get_whisper_model()
            audio = whisper.load_audio(temp_audio_path)
            result = whisper.transcribe(model, audio, language="en")

            # Cleanup
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

            return {
                "status": "success",
                "transcription": result
            }
        except Exception as e:
            if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return {"error": f"Transcription failed: {str(e)}"}

    # --- MODE 2: TTS GENERATION (Qwen) ---
    text = job_input.get("text")
    language = job_input.get("language", "English")

    if not text:
        return {"error": "No 'text' provided in input."}
    
    try:
        # Load ONLY the specific model requested
        model = load_target_model(mode)

        print(f"Generating audio in mode: {mode}")
        wavs = None
        sr = None

        if mode == "voice_design":
            instruct = job_input.get("instruct", "Clear, neutral voice.")
            wavs, sr = model.generate_voice_design(
                text=text, language=language, instruct=instruct
            )

        elif mode == "custom_voice":
            speaker = job_input.get("speaker", "Anna")
            wavs, sr = model.generate_custom_voice(
                text=text, language=language, speaker=speaker
            )

        elif mode == "voice_clone":
            ref_audio = job_input.get("ref_audio")
            ref_text = job_input.get("ref_text")
            
            if not ref_audio or not ref_text:
                return {"error": "Mode 'voice_clone' requires 'ref_audio' and 'ref_text'."}
            
            wavs, sr = model.generate_voice_clone(
                text=text, language=language, ref_audio=ref_audio, ref_text=ref_text
            )

        # --- THE FIX IS HERE ---
        # 1. Get the tensor from the GPU
        audio_tensor = wavs[0]
        
        # 2. Force move to CPU and convert to Numpy
        # This prevents "Numpy is not available" errors on GPU tensors
        if isinstance(audio_tensor, torch.Tensor):
            audio_array = audio_tensor.to(torch.float32).cpu().numpy()
        else:
            audio_array = audio_array # It was already numpy
            
        # 3. Write to IO buffer
        byte_io = io.BytesIO()
        scipy.io.wavfile.write(byte_io, sr, audio_array)
        wav_bytes = byte_io.getvalue()
        audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
        
        return {
            "status": "success",
            "mode": mode,
            "audio_base64": audio_b64
        }

    except Exception as e:
        print(f"Generation Error: {e}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})