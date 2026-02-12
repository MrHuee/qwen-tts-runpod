import runpod
import torch
import scipy.io.wavfile
import io
import base64
import gc
from qwen_tts import Qwen3TTSModel

# --- Configuration ---
MODEL_IDS = {
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_clone":  "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
}

# --- Global State ---
# We keep track of what is currently loaded so we don't reload if 
# by chance you send two requests in a row before the server shuts down.
CURRENT_MODEL = None
CURRENT_MODE = None

def load_target_model(target_mode):
    """
    Loads ONLY the requested model. Unloads previous ones if necessary.
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

    # 3. Load the new model with Flash Attention 2
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
            attn_implementation="flash_attention_2" # <--- Optimization Enabled
        )
        
        CURRENT_MODEL = model
        CURRENT_MODE = target_mode
        print(f"✅ {target_mode} loaded successfully.")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model with Flash Attention 2. Trying fallback... Error: {e}")
        # Fallback for older GPUs (T4) that might struggle with FA2
        model = Qwen3TTSModel.from_pretrained(
            model_id, 
            device_map="cuda", 
            dtype=torch.float16
        )
        CURRENT_MODEL = model
        CURRENT_MODE = target_mode
        return model

def handler(job):
    job_input = job["input"]
    
    # 1. Determine Mode from Input (Default to voice_design)
    mode = job_input.get("mode", "voice_design").lower()
    text = job_input.get("text")
    language = job_input.get("language", "English")

    if not text:
        return {"error": "No 'text' provided in input."}
    
    try:
        # 2. Load ONLY the specific model requested
        model = load_target_model(mode)

        print(f"Generating audio in mode: {mode}")
        wavs = None
        sr = None

        # 3. Mode-Specific Generation
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

        # 4. Process Output
        audio_array = wavs[0]
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