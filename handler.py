import runpod
import torch
import torchaudio
import io
import base64
import gc
import os
import tempfile
import numpy as np 
from qwen_tts import Qwen3TTSModel
import whisper_timestamped as whisper

# --- Configuration ---
MODEL_IDS = {
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_clone":  "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
}

CURRENT_MODEL = None
CURRENT_MODE = None
WHISPER_MODEL = None 

def load_target_model(target_mode):
    global CURRENT_MODEL, CURRENT_MODE
# ... (keep existing cleanup code) ...

    model_id = MODEL_IDS.get(target_mode)
    print(f"--- 🚀 Loading {target_mode}... ---")
    
    try:
        # 1. Force bfloat16 if available (A40 supports it)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # 2. explicit device map and flash attention
        model = Qwen3TTSModel.from_pretrained(
            model_id, 
            device_map="cuda", 
            dtype=dtype, 
            attn_implementation="flash_attention_2"
        )
        
        # 3. CRITICAL: Sanity check to ensure model is actually on GPU
        # If the wrapper exposes the internal model, force it (safeguard)
        if hasattr(model, 'model'):
            print(f"Model loaded on: {model.model.device}")
            
    except Exception as e:
        print(f"Warning: Flash Attention load failed, falling back. Error: {e}")
        model = Qwen3TTSModel.from_pretrained(model_id, device_map="cuda", dtype=torch.float16)

    CURRENT_MODEL = model
    CURRENT_MODE = target_mode
    return model

def get_whisper_model():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        print("--- 🎙️ Loading Whisper ---")
        WHISPER_MODEL = whisper.load_model("base", device="cuda")
    return WHISPER_MODEL

def handler(job):
    job_input = job["input"]
    mode = job_input.get("mode", "voice_design").lower()

    # --- TRANSCRIPTION ---
    if mode == "transcribe":
        try:
            audio_b64 = job_input.get("audio_base64")
            if not audio_b64: return {"error": "No audio_base64 provided"}
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(base64.b64decode(audio_b64))
                tmp_path = tmp.name

            model = get_whisper_model()
            audio = whisper.load_audio(tmp_path)
            result = whisper.transcribe(model, audio, language="en")
            
            os.remove(tmp_path)
            return {"status": "success", "transcription": result}
        except Exception as e:
            return {"error": str(e)}

    # --- TTS GENERATION ---
    text = job_input.get("text")
    language = job_input.get("language", "English")
    if not text: return {"error": "No text provided."}

    try:
        model = load_target_model(mode)
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

        # --- ROBUST AUDIO SAVING (FIXED) ---
        raw_audio = wavs[0]

        # Check if it's a NumPy array (which caused the error) or a Tensor
        if isinstance(raw_audio, np.ndarray):
            # Convert NumPy array to PyTorch Tensor
            audio_tensor = torch.from_numpy(raw_audio).float()
        elif torch.is_tensor(raw_audio):
            # If it is a Tensor, ensure it's on CPU
            audio_tensor = raw_audio.detach().cpu().float()
        else:
            # Fallback if type is unknown (though unlikely)
            raise ValueError(f"Unexpected audio data type: {type(raw_audio)}")
        
        # Ensure it is 2D (Channels, Time) for torchaudio
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        byte_io = io.BytesIO()
        torchaudio.save(byte_io, audio_tensor, sr, format="wav")
        
        return {
            "status": "success",
            "audio_base64": base64.b64encode(byte_io.getvalue()).decode('utf-8')
        }

    except Exception as e:
        return {"error": f"Detailed Error: {str(e)}"}

runpod.serverless.start({"handler": handler})