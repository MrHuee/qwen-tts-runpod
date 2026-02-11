import runpod
import torch
import scipy.io.wavfile
import io
import base64
import numpy as np
from qwen_tts import Qwen3TTSModel

# --- Configuration ---
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# --- Load Model Once (Global) ---
print("--- Loading Model... ---")
try:
    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID, 
        device_map="cuda", 
        dtype=torch.float16
    )
    print("--- Model Loaded ---")
except Exception as e:
    print(f"Error loading model: {e}")
    # We don't raise error here so the container stays alive for debugging logs
    model = None

def handler(job):
    if model is None:
        return {"error": "Model failed to load during startup. Check logs."}

    job_input = job["input"]
    text = job_input.get("text")
    
    if not text:
        return {"error": "No text provided."}

    try:
        # Generate Audio
        # Note: 'voice_design' usually needs a reference audio. 
        # If just text, we use standard generation or a default voice.
        # This checks if user provided a reference voice URL or base64.
        
        # Simple generation for now (adjust parameters as needed)
        wavs, sr = model.generate_custom_voice(
            language="English", # or detect from text
            text=text,
            speaker="Announcer" # Qwen3 generic speaker if no reference provided
        )
        
        # Convert to Base64
        audio_array = wavs[0]
        byte_io = io.BytesIO()
        scipy.io.wavfile.write(byte_io, sr, audio_array)
        wav_bytes = byte_io.getvalue()
        audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
        
        return {"audio_base64": audio_b64}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})