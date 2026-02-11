import runpod
import torch
import scipy.io.wavfile
import io
import base64
import gc
from qwen_tts import Qwen3TTSModel

# --- Configuration ---
# Models to load. Comment out any you don't need to save VRAM.
MODEL_IDS = {
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_clone":  "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
}

MODELS = {}

def load_models():
    """Loads all configured models into GPU memory at startup."""
    print("--- 🚀 Starting Model Loading Sequence... ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for mode_name, model_id in MODEL_IDS.items():
        print(f"--- Loading {mode_name} ({model_id})... ---")
        try:
            # We use float16 to save VRAM. Use bfloat16 if on Ampere (3090/A10/A100)
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            
            MODELS[mode_name] = Qwen3TTSModel.from_pretrained(
                model_id, 
                device_map=device, 
                dtype=dtype
            )
            print(f"✅ {mode_name} loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load {mode_name}: {e}")
            # We continue loading others even if one fails
            
    print("--- 🏁 All Models Loaded ---")

# --- Initialize Models Immediately ---
if not MODELS:
    load_models()

def handler(job):
    job_input = job["input"]
    
    # 1. Determine Mode
    mode = job_input.get("mode", "voice_design").lower()
    text = job_input.get("text")
    language = job_input.get("language", "English")

    if not text:
        return {"error": "No 'text' provided in input."}
    
    # 2. Select Model
    model = MODELS.get(mode)
    if not model:
        return {"error": f"Model for mode '{mode}' is not loaded or invalid mode."}

    try:
        print(f"Generating audio in mode: {mode}")
        wavs = None
        sr = None

        # 3. Mode-Specific Generation
        if mode == "voice_design":
            # Requires 'instruct'
            instruct = job_input.get("instruct", "Clear, neutral voice.")
            wavs, sr = model.generate_voice_design(
                text=text, 
                language=language, 
                instruct=instruct
            )

        elif mode == "custom_voice":
            # Requires 'speaker' (Preset list: Vivian, Ryan, etc.)
            speaker = job_input.get("speaker", "Anna")
            wavs, sr = model.generate_custom_voice(
                text=text, 
                language=language, 
                speaker=speaker
            )

        elif mode == "voice_clone":
            # Requires 'ref_audio' (URL/Base64) and 'ref_text' (transcript of that audio)
            ref_audio = job_input.get("ref_audio")
            ref_text = job_input.get("ref_text")
            
            if not ref_audio or not ref_text:
                return {"error": "Mode 'voice_clone' requires 'ref_audio' and 'ref_text'."}
            
            # Note: The Qwen library handles URLs automatically
            wavs, sr = model.generate_voice_clone(
                text=text, 
                language=language, 
                ref_audio=ref_audio, 
                ref_text=ref_text
            )

        # 4. Process Output
        # Helper to convert numpy audio to base64
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