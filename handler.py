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

# ──────────────────────────────────────────────────────────────────────────────
# ⚡ THREAD / ENV TUNING
# ──────────────────────────────────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Enable TF32 for potential speedup on Ampere+ GPUs (like A40)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_IDS = {
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    #"custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_clone":  "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

CURRENT_MODEL = None
CURRENT_MODE = None
WHISPER_MODEL = None

# ──────────────────────────────────────────────────────────────────────────────
# Model helpers
# ──────────────────────────────────────────────────────────────────────────────
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
    
    # 🩹 FIX: Explicitly set config dtype to silence warnings / ensure behavior
    if hasattr(model, "model") and hasattr(model.model, "config"):
        model.model.config.torch_dtype = torch.bfloat16
        print("   🔧 Enforced config.torch_dtype = bfloat16")

    try:
        p = next(model.model.parameters())
        print(f"   📊 Model: dtype={p.dtype}, device={p.device}")
    except:
        pass

    return model

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

    try:
        model = _load_tts(model_id, use_flash=True)
        print("✅ Loaded: bfloat16 + Flash Attention 2")
    except Exception as e:
        print(f"⚠️ FA2 failed ({e}), retrying with standard...")
        model = _load_tts(model_id, use_flash=False)
        print("✅ Loaded: bfloat16 + standard attention")

    CURRENT_MODEL = model
    CURRENT_MODE = target_mode
    return model

def set_seed(seed):
    """Sets the seed for reproducibility to keep voice tone consistent."""
    if seed is not None:
        try:
            seed = int(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            print(f"🌱 Seed set to: {seed}")
        except Exception as e:
            print(f"⚠️ Failed to set seed: {e}")

def get_whisper_model():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        import whisper_timestamped as whisper
        print("--- 🎙️ Loading Whisper base on GPU ---")
        t0 = time.time()
        WHISPER_MODEL = whisper.load_model("base", device="cuda")
        print(f"✅ Whisper ready ({time.time() - t0:.2f}s)")
    return WHISPER_MODEL

# ──────────────────────────────────────────────────────────────────────────────
# 🔥 STARTUP PRELOAD
# ──────────────────────────────────────────────────────────────────────────────
print("--- 🔥 Preloading models... ---")
try:
    load_target_model("voice_design")
    print("--- ⏱️ CUDA warmup ---")
    t0 = time.time()
    with torch.inference_mode():
        # Pass use_cache=True explicitly
        _ = CURRENT_MODEL.generate_voice_design(
            text="Hi.", language="English", instruct="Warmup.",
            max_new_tokens=50, use_cache=True
        )
    torch.cuda.synchronize()
    print(f"✅ Warmup done in {time.time() - t0:.2f}s")
except Exception as e:
    print(f"⚠️ Startup preload failed: {e}")

try:
    get_whisper_model()
except Exception as e:
    print(f"⚠️ Whisper preload failed: {e}")
print("--- ✅ Startup Complete ---")


# ──────────────────────────────────────────────────────────────────────────────
# Audio download helper (for voice cloning from URLs)
# ──────────────────────────────────────────────────────────────────────────────
import subprocess

def _gdrive_download(file_id):
    """Download from Google Drive, handling the virus-scan confirmation page."""
    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = session.get(url, timeout=60, allow_redirects=True)
    resp.raise_for_status()

    # Check if we got the confirmation page (large files / virus scan)
    # Google returns HTML with a confirm token we need to accept
    if b"</html>" in resp.content[:5000].lower() or b"confirm" in resp.content[:5000].lower():
        # Try to extract the confirm token
        import re as _re
        confirm_match = _re.search(r'confirm=([0-9A-Za-z_-]+)', resp.text)
        if confirm_match:
            confirm_token = confirm_match.group(1)
            print(f"   🔑 Got GDrive confirm token, retrying...")
            resp = session.get(url + f"&confirm={confirm_token}", timeout=60, allow_redirects=True)
            resp.raise_for_status()
        else:
            # Try with confirm=t (common workaround)
            print(f"   🔑 Trying GDrive confirm=t workaround...")
            resp = session.get(url + "&confirm=t", timeout=60, allow_redirects=True)
            resp.raise_for_status()

    return resp

def download_ref_audio(ref_audio_input):
    """
    If ref_audio_input is a URL, download it to a temp file and return the path.
    Handles Google Drive sharing links. Always converts to WAV via ffmpeg
    to guarantee compatibility with soundfile/librosa.
    If it's already a local path, return as-is.
    """
    if not isinstance(ref_audio_input, str):
        return ref_audio_input, None  # already bytes or file-like

    # Check if it's a URL
    if not ref_audio_input.startswith(("http://", "https://")):
        return ref_audio_input, None  # local path, return as-is

    url = ref_audio_input
    is_gdrive = False

    # Detect Google Drive links
    gdrive_match = re.search(r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)', url)
    if gdrive_match:
        file_id = gdrive_match.group(1)
        is_gdrive = True
        print(f"   🔗 Detected Google Drive file (ID: {file_id})")

    print(f"   ⬇️ Downloading ref audio...")
    t0 = time.time()

    if is_gdrive:
        resp = _gdrive_download(file_id)
    else:
        resp = requests.get(url, timeout=60, allow_redirects=True)
        resp.raise_for_status()

    raw_bytes = resp.content
    content_type = resp.headers.get("Content-Type", "")
    print(f"   ✅ Downloaded {len(raw_bytes)} bytes in {time.time()-t0:.2f}s")
    print(f"   📋 Content-Type: {content_type}")

    # Sanity check: if content is HTML, the download failed
    if raw_bytes[:50].strip().lower().startswith((b"<!doctype", b"<html")):
        raise ValueError(
            "Downloaded content is HTML, not audio. "
            "Make sure the Google Drive file is shared as 'Anyone with the link'."
        )

    # Check if the content is base64-encoded audio (e.g. a .txt file with base64)
    # Base64 audio starts with recognizable patterns like UklGR (RIFF/WAV header)
    text_content = raw_bytes.strip()
    is_base64 = False
    if content_type.startswith("text/") or text_content[:10].isascii():
        # Try to detect base64: check if it looks like pure base64 text
        try:
            sample = text_content[:100].decode("ascii", errors="strict")
            # Base64 only contains A-Z, a-z, 0-9, +, /, = and whitespace
            import string
            b64_chars = set(string.ascii_letters + string.digits + "+/=\n\r\t ")
            if all(c in b64_chars for c in sample):
                # Try decoding a small chunk to verify
                test_decode = base64.b64decode(text_content[:1000])
                if len(test_decode) > 0:
                    is_base64 = True
        except Exception:
            pass

    if is_base64:
        print(f"   🔓 Detected base64-encoded audio, decoding...")
        raw_bytes = base64.b64decode(text_content)
        print(f"   ✅ Decoded to {len(raw_bytes)} bytes of audio data")

    # Save raw download to temp file (use generic extension)
    raw_tmp = tempfile.NamedTemporaryFile(suffix=".download", delete=False)
    raw_tmp.write(raw_bytes)
    raw_tmp.close()

    # Convert to WAV using ffmpeg for guaranteed compatibility
    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_tmp.close()

    print(f"   🔄 Converting to WAV via ffmpeg...")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", raw_tmp.name, "-ar", "16000", "-ac", "1",
             "-sample_fmt", "s16", wav_tmp.name],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"   ❌ ffmpeg stderr: {result.stderr[-500:]}")
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[-200:]}")
        print(f"   ✅ Converted to WAV: {wav_tmp.name}")
    finally:
        # Clean up the raw download
        if os.path.exists(raw_tmp.name):
            os.remove(raw_tmp.name)

    return wav_tmp.name, wav_tmp.name  # return (path_for_model, path_to_cleanup)


# ──────────────────────────────────────────────────────────────────────────────
# Request handler
# ──────────────────────────────────────────────────────────────────────────────
def handler(job):
    t_start = time.time()
    try:
        job_input = job.get("input", {})
        mode = job_input.get("mode", "voice_design").lower()

        if mode == "transcribe":
            audio_b64 = job_input.get("audio_base64")
            if not audio_b64: return {"error": "No audio_base64 provided."}
            import whisper_timestamped as whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(base64.b64decode(audio_b64))
                tmp_path = tmp.name
            try:
                t_trans = time.time()
                wmodel = get_whisper_model()
                audio = whisper.load_audio(tmp_path)
                with torch.inference_mode():
                    result = whisper.transcribe(wmodel, audio, language="en")
                return {"status": "success", "transcription": result}
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)

        if mode not in MODEL_IDS: return {"error": f"Unknown mode {mode}"}

        text = job_input.get("text", "")
        if not text: return {"error": "No text."}

        # Dynamic max_new_tokens (default 1024 ~85s)
        # Explicitly enabling use_cache for speed
        user_max = job_input.get("max_new_tokens")
        max_tokens = int(user_max) if user_max else 1024

        model = load_target_model(mode)
        
        print(f"--- 🗣️ GEN: {mode} | Len:{len(text)} | Max:{max_tokens} | Cache:True ---")
        t_gen = time.time()
        
        with torch.inference_mode():
            # Common kwargs for speed
            gen_args = dict(
                text=text, 
                language=job_input.get("language", "English"),
                max_new_tokens=max_tokens,
                use_cache=True,  # 🚀 CRITICAL FOR SPEED
            )

            if mode == "voice_design":
                gen_args["instruct"] = job_input.get("instruct", "Clear voice.")
                wavs, sr = model.generate_voice_design(**gen_args)
            #elif mode == "custom_voice":
            #    gen_args["speaker"] = job_input.get("speaker", "Anna")
            #    wavs, sr = model.generate_custom_voice(**gen_args)
            elif mode == "voice_clone":
                raw_ref = job_input.get("ref_audio")
                if not raw_ref:
                    return {"error": "ref_audio is required for voice_clone mode."}
                ref_path, tmp_audio_path = download_ref_audio(raw_ref)
                gen_args["ref_audio"] = ref_path
                gen_args["ref_text"] = job_input.get("ref_text")
                try:
                    wavs, sr = model.generate_voice_clone(**gen_args)
                finally:
                    if tmp_audio_path and os.path.exists(tmp_audio_path):
                        os.remove(tmp_audio_path)

        torch.cuda.synchronize()
        dt_gen = time.time() - t_gen
        
        # Calculate Tokens/Sec heuristic (approx)
        # Using 12Hz as frame rate
        # 45s audio = 540 frames. 
        # But we don't know exact frames generated without inspecting output.
        # Just log time.
        print(f"⏱️ Generation took {dt_gen:.2f}s")

        # Encode
        raw_audio = wavs[0]
        if isinstance(raw_audio, np.ndarray):
            audio_tensor = torch.from_numpy(raw_audio).float()
        elif torch.is_tensor(raw_audio):
             audio_tensor = raw_audio.detach().cpu().float()
        
        if audio_tensor.dim() == 1: audio_tensor = audio_tensor.unsqueeze(0)
        
        byte_io = io.BytesIO()
        torchaudio.save(byte_io, audio_tensor, sr, format="wav")
        audio_b64 = base64.b64encode(byte_io.getvalue()).decode("utf-8")

        dt_total = time.time() - t_start
        print(f"✅ Complete in {dt_total:.2f}s (Gen: {dt_gen:.2f}s)")
        return {
            "status": "success", 
            "audio_base64": audio_b64,
            "stats": {"total_time": f"{dt_total:.2f}s", "generation_time": f"{dt_gen:.2f}s"}
        }

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"❌ Error: {err}")
        return {"error": str(e), "traceback": err}

runpod.serverless.start({"handler": handler})