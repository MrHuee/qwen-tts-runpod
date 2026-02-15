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

# --- ❌ REMOVED PERFORMANCE FIX 1: THREAD LIMITING ---
# Limiting threads to 4 is fatal if the model falls back to CPU.
# We want the CPU to run as fast as possible if it is needed.
# os.environ["OMP_NUM_THREADS"] = "4" <--- DELETED
# os.environ["MKL_NUM_THREADS"] = "4" <--- DELETED
# torch.set_num_threads(4)            <--- DELETED

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
    if CURRENT_MODE == target_mode and CURRENT_MODEL is not None:
        return CURRENT_MODEL

    if CURRENT_MODEL is not None:
        del CURRENT_MODEL
        CURRENT_MODEL = None
        gc.collect()
        torch.cuda.empty_cache()

    model_id = MODEL_IDS.get(target_mode)
    if not model_id: raise ValueError(f"Invalid mode: {target_mode}")

    print(f"--- 🚀 Loading {target_mode} on GPU... ---")
    try:
        # Load the model
        model = Qwen3TTSModel.from_pretrained(
            model_id, 
            device_map="cuda",  # Attempt to map to GPU
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2"
        )
        # --- ⚡ FIX 2: FORCE GPU ---
        # Explicitly move the model to CUDA. This helps catch sub-modules 
        # that device_map might miss.
        model = model.cuda()
        print("✅ Model loaded and forced to CUDA.")
        
    except Exception as e:
        print(f"⚠️ Flash Attention load failed, falling back. Error: {e}")
        model = Qwen3TTSModel.from_pretrained(
            model_id, 
            device_map="cuda", 
            torch_dtype=torch.float16
        )
        model = model.cuda()

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

        print(f"--- 🗣️ Generating audio... ---")
        
        # NOTE: If the library re-initializes components during generation, 
        # they might still default to CPU. Removing thread limits ensures 
        # this fallback is at least fast (seconds instead of minutes).
        
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

        # --- AUDIO SAVING ---
        raw_audio = wavs[0]

        if isinstance(raw_audio, np.ndarray):
            audio_tensor = torch.from_numpy(raw_audio).float()
        elif torch.is_tensor(raw_audio):
            audio_tensor = raw_audio.detach().cpu().float()
        else:
            raise ValueError(f"Unexpected audio data type: {type(raw_audio)}")
        
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