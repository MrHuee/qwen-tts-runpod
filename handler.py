import runpod
import torch
import torchaudio
import io
import base64
import gc
import os
import re
import tempfile
import numpy as np
import time
import warnings
import requests
import subprocess
import soundfile as sf  # Required for manual loading

# ──────────────────────────────────────────────────────────────────────────────
# ⚡ THREAD / ENV TUNING
# ──────────────────────────────────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_IDS = {
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "voice_clone":  "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

CURRENT_MODEL = None
CURRENT_MODE = None
WHISPER_MODEL = None

# ──────────────────────────────────────────────────────────────────────────────
# 🛠️ HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    if seed is not None:
        try:
            seed = int(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            print(f"🌱 Seed set to: {seed}")
        except Exception as e:
            print(f"⚠️ Failed to set seed: {e}")

def _load_tts(model_id, use_flash=True):
    from qwen_tts import Qwen3TTSModel
    print(f"--- 🛠️ Loading {model_id} (Flash={use_flash}) ---")
    kwargs = dict(
        device_map="cuda:0",
        dtype=torch.bfloat16,
        torch_dtype=torch.bfloat16, 
    )
    if use_flash:
        kwargs["attn_implementation"] = "flash_attention_2"

    t_start = time.time()
    model = Qwen3TTSModel.from_pretrained(model_id, **kwargs)
    print(f"   ⏱️ Model load took {time.time() - t_start:.2f}s")
    
    if hasattr(model, "model") and hasattr(model.model, "config"):
        model.model.config.torch_dtype = torch.bfloat16

    return model

def load_target_model(target_mode):
    global CURRENT_MODEL, CURRENT_MODE
    if CURRENT_MODE == target_mode and CURRENT_MODEL is not None:
        return CURRENT_MODEL
    
    if CURRENT_MODEL is not None:
        print("♻️ Unloading previous model...")
        del CURRENT_MODEL
        CURRENT_MODEL = None
        gc.collect()
        torch.cuda.empty_cache()

    model_id = MODEL_IDS.get(target_mode)
    try:
        model = _load_tts(model_id, use_flash=True)
        print("✅ Loaded: bfloat16 + Flash Attention 2")
    except Exception as e:
        print(f"⚠️ FA2 failed ({e}), retrying with standard...")
        model = _load_tts(model_id, use_flash=False)

    CURRENT_MODEL = model
    CURRENT_MODE = target_mode
    return model

def get_whisper_model():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        import whisper_timestamped as whisper
        print("--- 🎙️ Loading Whisper base on GPU ---")
        WHISPER_MODEL = whisper.load_model("base", device="cuda")
    return WHISPER_MODEL

# ──────────────────────────────────────────────────────────────────────────────
# 📥 DOWNLOAD HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _gdrive_download(file_id):
    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = session.get(url, timeout=60, allow_redirects=True)
    resp.raise_for_status()

    if b"</html>" in resp.content[:5000].lower() or b"confirm" in resp.content[:5000].lower():
        import re as _re
        confirm_match = _re.search(r'confirm=([0-9A-Za-z_-]+)', resp.text)
        if confirm_match:
            confirm_token = confirm_match.group(1)
            resp = session.get(url + f"&confirm={confirm_token}", timeout=60, allow_redirects=True)
            resp.raise_for_status()
        else:
            resp = session.get(url + "&confirm=t", timeout=60, allow_redirects=True)
            resp.raise_for_status()
    return resp

def download_ref_audio(ref_audio_input):
    """
    Downloads audio from URL/GDrive, converts to 16kHz WAV via FFmpeg,
    and returns path to the clean WAV file.
    """
    if not isinstance(ref_audio_input, str):
        return ref_audio_input, None 

    if not ref_audio_input.startswith(("http://", "https://")):
        # If it's a local path, ensure it exists
        if os.path.exists(ref_audio_input):
            return ref_audio_input, None
        return ref_audio_input, None 

    # Download logic
    url = ref_audio_input
    gdrive_match = re.search(r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)', url)
    
    if gdrive_match:
        file_id = gdrive_match.group(1)
        resp = _gdrive_download(file_id)
    else:
        resp = requests.get(url, timeout=60, allow_redirects=True)
        resp.raise_for_status()

    raw_bytes = resp.content
    
    # Handle Base64 text content
    if resp.headers.get("Content-Type", "").startswith("text/") or raw_bytes[:10].isascii():
        try:
            text_content = raw_bytes.strip()
            if len(text_content) > 100 and b" " not in text_content[0:50]: 
                raw_bytes = base64.b64decode(text_content)
                print("   🔓 Decoded base64 from URL")
        except:
            pass

    # Save raw download
    raw_tmp = tempfile.NamedTemporaryFile(suffix=".download", delete=False)
    raw_tmp.write(raw_bytes)
    raw_tmp.close()

    # Convert to standard 16kHz WAV
    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_tmp.close()

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_tmp.name, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", wav_tmp.name],
            capture_output=True, text=True, timeout=30, check=True
        )
    finally:
        if os.path.exists(raw_tmp.name):
            os.remove(raw_tmp.name)

    return wav_tmp.name, wav_tmp.name 


# ──────────────────────────────────────────────────────────────────────────────
# 🚀 MAIN HANDLER
# ──────────────────────────────────────────────────────────────────────────────
def handler(job):
    t_start = time.time()
    try:
        job_input = job.get("input", {})
        mode = job_input.get("mode", "voice_design").lower().strip()

        # === 1. TRANSCRIPTION LOGIC ===
        if mode == "transcribe":
            audio_b64 = job_input.get("audio_base64")
            if not audio_b64: return {"error": "No audio_base64 provided."}
            
            import whisper_timestamped as whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(base64.b64decode(audio_b64))
                tmp_path = tmp.name
            
            try:
                wmodel = get_whisper_model()
                audio = whisper.load_audio(tmp_path)
                with torch.inference_mode():
                    result = whisper.transcribe(wmodel, audio, language="en")
                return {"status": "success", "transcription": result}
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)

        # === 2. TTS LOGIC ===
        if mode not in MODEL_IDS:
            return {"error": f"Unknown mode '{mode}'."}

        text = job_input.get("text", "")
        if not text: return {"error": "No text provided."}

        user_max = job_input.get("max_new_tokens")
        max_tokens = int(user_max) if user_max else 1024
        
        # Load Model
        model = load_target_model(mode)
        
        # Set Seed
        seed = job_input.get("seed", 42)
        set_seed(seed)

        print(f"--- 🗣️ GEN: {mode} | Len:{len(text)} | Seed:{seed} ---")
        t_gen = time.time()
        
        with torch.inference_mode():
            gen_args = dict(
                text=text, 
                language=job_input.get("language", "English"),
                max_new_tokens=max_tokens,
                use_cache=True, 
            )

            # --- MODE A: VOICE DESIGN ---
            if mode == "voice_design":
                gen_args["instruct"] = job_input.get("instruct", "Clear voice.")
                wavs, sr = model.generate_voice_design(**gen_args)

            # --- MODE B: VOICE CLONE (The Fix!) ---
            elif mode == "voice_clone":
                raw_ref = job_input.get("ref_audio")
                ref_text = job_input.get("ref_text")

                if not raw_ref: return {"error": "ref_audio required"}
                if not ref_text: return {"error": "ref_text required (transcript of audio)"}

                # 1. Download & Convert Audio to file
                ref_path, tmp_path = download_ref_audio(raw_ref)
                
                try:
                    # 2. MANUALLY Load Audio into Memory (Numpy)
                    # This ensures the model receives valid data, not just a path
                    print(f"   🎤 Loading audio data from: {ref_path}")
                    ref_audio_data, ref_audio_sr = sf.read(ref_path)
                    
                    # 3. Explicitly Create the Prompt Object
                    # This step extracts the speaker embedding. If this fails, we catch it here.
                    print("   🧠 Extracting voice embedding...")
                    voice_clone_prompt = model.create_voice_clone_prompt(
                        ref_audio=(ref_audio_data, ref_audio_sr), # Pass Tuple: (numpy, int)
                        ref_text=ref_text
                    )
                    
                    # 4. Generate using the Prompt Object
                    print("   🗣️ Synthesizing speech...")
                    wavs, sr = model.generate_voice_clone(
                        text=text,
                        language=job_input.get("language", "English"),
                        voice_clone_prompt=voice_clone_prompt, # Pass the PROMPT, not the file
                        max_new_tokens=max_tokens,
                        use_cache=True
                    )
                except Exception as inner_e:
                    print(f"❌ Cloning Failed during extraction: {inner_e}")
                    raise inner_e # Re-raise to return error to n8n
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)

        torch.cuda.synchronize()
        dt_gen = time.time() - t_gen
        print(f"⏱️ Generation took {dt_gen:.2f}s")

        # Encode Output
        raw_audio = wavs[0]
        if isinstance(raw_audio, np.ndarray):
            audio_tensor = torch.from_numpy(raw_audio).float()
        elif torch.is_tensor(raw_audio):
             audio_tensor = raw_audio.detach().cpu().float()
        
        if audio_tensor.dim() == 1: audio_tensor = audio_tensor.unsqueeze(0)
        
        byte_io = io.BytesIO()
        torchaudio.save(byte_io, audio_tensor, int(sr), format="wav")
        audio_b64 = base64.b64encode(byte_io.getvalue()).decode("utf-8")

        dt_total = time.time() - t_start
        return {
            "status": "success", 
            "audio_base64": audio_b64,
            "sample_rate": int(sr),
            "stats": {"total_time": f"{dt_total:.2f}s", "generation_time": f"{dt_gen:.2f}s"}
        }

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"❌ Error: {err}")
        return {"error": str(e), "traceback": err}

runpod.serverless.start({"handler": handler})