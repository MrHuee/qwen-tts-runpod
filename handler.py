import runpod
import torch
import torchaudio
import io
import base64
import gc
import os
import re
import hashlib
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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_IDS = {
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "voice_clone":  "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

# ──────────────────────────────────────────────────────────────────────────────
# 🎙️ BUNDLED VOICE REGISTRY
# Maps friendly name → path inside the Docker image.
# Add entries here + COPY the files in your Dockerfile / build context.
# The "default" key is the fallback when no ref_audio is specified.
# ──────────────────────────────────────────────────────────────────────────────
VOICES_DIR = "/app/voices"

# ─────────────────────────────────────────────────────────────────────────────
# 🎙️ VOICE REGISTRY
# Maps friendly name → path inside the Docker image.
# Any format ffmpeg supports: .mp3 .wav .m4a .flac
#
# ⚡ ZERO-REBUILD voice switching via RunPod env vars:
#   Set  VOICE_DEFAULT_URL=https://...    to override "default" at runtime
#   Set  VOICE_NARRATOR_URL=https://...   to add/override "narrator" at runtime
#   Pattern: VOICE_<NAME_UPPERCASE>_URL=<url>
#   handler.py downloads & caches these ONCE at startup — no image rebuild needed.
# ─────────────────────────────────────────────────────────────────────────────
VOICE_REGISTRY: dict[str, str] = {
    # Bundled in the Docker image (baked in, always available):
    "default": os.path.join(VOICES_DIR, "default.mp3"),  # Theo Silk – British Deep Sleep
    # Add more bundled voices here (requires rebuild):
    # "narrator": os.path.join(VOICES_DIR, "narrator.mp3"),
}

# Auto-register env-var voices (no rebuild needed — set in RunPod dashboard)
# Pattern: VOICE_<NAME>_URL=https://cdn.example.com/voice.mp3
# These are downloaded at startup and cached; the key becomes the lowercase name.
_ENV_VOICE_URLS: dict[str, str] = {
    k.removeprefix("VOICE_").removesuffix("_URL").lower(): v
    for k, v in os.environ.items()
    if k.startswith("VOICE_") and k.endswith("_URL")
}
if _ENV_VOICE_URLS:
    print(f"   🌐 Found {len(_ENV_VOICE_URLS)} env-var voice(s): {list(_ENV_VOICE_URLS)}")

# In-memory audio cache: sha256(audio_bytes) → converted_wav_path
_AUDIO_CACHE: dict[str, str] = {}
# Paths downloaded from env-var URLs (populated at startup)
_ENV_VOICE_PATHS: dict[str, str] = {}

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

    if hasattr(model, "model") and hasattr(model.model, "config"):
        model.model.config.torch_dtype = torch.bfloat16

    try:
        p = next(model.model.parameters())
        print(f"   📊 Model: dtype={p.dtype}, device={p.device}")
    except Exception:
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
# 🎙️ AUDIO RESOLUTION — fast path for bundled voices, caching for everything else
# ──────────────────────────────────────────────────────────────────────────────
import subprocess


def _convert_to_wav(raw_bytes: bytes, label: str = "") -> str:
    """
    Convert arbitrary audio bytes → 16 kHz mono WAV via ffmpeg.
    Returns the path to a temp WAV file. Caller is responsible for cleanup
    only if they chose to; bundled voices are left on disk permanently.
    """
    key = hashlib.sha256(raw_bytes).hexdigest()
    if key in _AUDIO_CACHE:
        print(f"   ⚡ Cache hit for audio{label} — skipping conversion")
        return _AUDIO_CACHE[key]

    raw_tmp = tempfile.NamedTemporaryFile(suffix=".download", delete=False)
    raw_tmp.write(raw_bytes)
    raw_tmp.close()

    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_tmp.close()

    print(f"   🔄 Converting{label} to WAV via ffmpeg...")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", raw_tmp.name,
             "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", wav_tmp.name],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[-200:]}")
        print(f"   ✅ Converted to WAV: {wav_tmp.name}")
    finally:
        if os.path.exists(raw_tmp.name):
            os.remove(raw_tmp.name)

    _AUDIO_CACHE[key] = wav_tmp.name
    return wav_tmp.name


def _preload_bundled_voice(name: str, path: str) -> None:
    """Pre-convert a bundled WAV and warm the cache at startup."""
    if not os.path.exists(path):
        print(f"   ⚠️ Bundled voice '{name}' not found at {path}")
        return
    with open(path, "rb") as f:
        raw = f.read()
    # If it's already a clean WAV we trust it; still run through ffmpeg once
    # to normalize sample-rate/format, then cache the result.
    _convert_to_wav(raw, label=f" '{name}'")
    print(f"   ✅ Bundled voice '{name}' cached")


def resolve_ref_audio(ref_audio_input: str | None) -> tuple[str, bool]:
    """
    Resolve ref_audio to a ready-to-use WAV path.

    Priority:
      1. None / "default"    → bundled default voice (instant, pre-cached)
      2. Registry name       → other bundled voice (instant, pre-cached)
      3. Env-var voice name  → downloaded at startup, instant from cache
      4. "data:audio/..."    → inline base64 data URI
      5. Plain base64 str    → decode and convert
      6. http/https URL      → download then convert (slowest — also cached after)
      7. Local file path     → read and convert

    Returns (wav_path, is_temp).
      is_temp=True  → caller should delete after use
      is_temp=False → cached/bundled file, do NOT delete
    """
    normalized = ref_audio_input.strip().lower() if isinstance(ref_audio_input, str) else None

    # ── 1 & 2: Bundled registry ──────────────────────────────────────────────
    if normalized is None or normalized == "default":
        key = "default"
    elif normalized in VOICE_REGISTRY:
        key = normalized
    else:
        key = None

    if key is not None:
        path = VOICE_REGISTRY[key]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Bundled voice '{key}' not found at {path}. "
                "Did you COPY the voices/ folder into the Docker image?"
            )
        with open(path, "rb") as f:
            raw = f.read()
        h = hashlib.sha256(raw).hexdigest()
        cached = _AUDIO_CACHE.get(h, path)
        print(f"   ⚡ Using bundled voice '{key}': {cached}")
        return cached, False

    # ── 3: Env-var voice (downloaded at startup, instant) ────────────────────
    if normalized in _ENV_VOICE_PATHS:
        cached_path = _ENV_VOICE_PATHS[normalized]
        print(f"   ⚡ Using env-var voice '{normalized}': {cached_path}")
        return cached_path, False

    # ── 4: Data URI ──────────────────────────────────────────────────────────
    if isinstance(ref_audio_input, str) and ref_audio_input.startswith("data:"):
        _, b64_part = ref_audio_input.split(",", 1)
        raw_bytes = base64.b64decode(b64_part)
        print(f"   📦 Received data URI ({len(raw_bytes)} bytes)")
        return _convert_to_wav(raw_bytes, " (data URI)"), True

    # ── 5: Plain base64 ──────────────────────────────────────────────────────
    if isinstance(ref_audio_input, str) and not ref_audio_input.startswith(("http://", "https://", "/")):
        try:
            raw_bytes = base64.b64decode(ref_audio_input, validate=True)
            print(f"   📦 Received base64 audio ({len(raw_bytes)} bytes)")
            return _convert_to_wav(raw_bytes, " (base64)"), True
        except Exception:
            pass

    # ── 6: URL ───────────────────────────────────────────────────────────────
    if isinstance(ref_audio_input, str) and ref_audio_input.startswith(("http://", "https://")):
        raw_bytes = _download_audio(ref_audio_input)
        return _convert_to_wav(raw_bytes, " (URL)"), True

    # ── 7: Local path ────────────────────────────────────────────────────────
    if os.path.exists(ref_audio_input):
        with open(ref_audio_input, "rb") as f:
            raw_bytes = f.read()
        return _convert_to_wav(raw_bytes, " (local file)"), True

    raise ValueError(f"Cannot resolve ref_audio: {str(ref_audio_input)[:120]}")


# ──────────────────────────────────────────────────────────────────────────────
# URL / Google Drive download (kept as slow fallback)
# ──────────────────────────────────────────────────────────────────────────────
def _gdrive_download(file_id: str) -> bytes:
    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = session.get(url, timeout=60, allow_redirects=True)
    resp.raise_for_status()

    if b"</html>" in resp.content[:5000].lower() or b"confirm" in resp.content[:5000].lower():
        confirm_match = re.search(r'confirm=([0-9A-Za-z_-]+)', resp.text)
        if confirm_match:
            resp = session.get(url + f"&confirm={confirm_match.group(1)}", timeout=60)
        else:
            resp = session.get(url + "&confirm=t", timeout=60)
        resp.raise_for_status()

    return resp.content


def _download_audio(url: str) -> bytes:
    print(f"   ⬇️ Downloading ref audio from URL...")
    t0 = time.time()

    gdrive_match = re.search(r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)', url)
    if gdrive_match:
        raw = _gdrive_download(gdrive_match.group(1))
    else:
        resp = requests.get(url, timeout=60, allow_redirects=True)
        resp.raise_for_status()
        raw = resp.content

    if raw[:50].strip().lower().startswith((b"<!doctype", b"<html")):
        raise ValueError("Downloaded content is HTML, not audio.")

    # Detect base64-encoded content in a text response
    try:
        sample = raw.strip()[:100].decode("ascii", errors="strict")
        import string
        b64_chars = set(string.ascii_letters + string.digits + "+/=\n\r\t ")
        if all(c in b64_chars for c in sample):
            raw = base64.b64decode(raw.strip())
    except Exception:
        pass

    print(f"   ✅ Downloaded {len(raw)} bytes in {time.time()-t0:.2f}s")
    return raw


# ──────────────────────────────────────────────────────────────────────────────
# 🔥 STARTUP PRELOAD
# ──────────────────────────────────────────────────────────────────────────────
print("--- 🔥 Preloading models and voices... ---")

# Pre-cache all bundled voices (from Docker image) so first request is instant
for _name, _path in VOICE_REGISTRY.items():
    _preload_bundled_voice(_name, _path)

# Download & cache env-var voices (VOICE_<NAME>_URL) at startup
# These are instant from cache for every request after boot.
for _env_name, _env_url in _ENV_VOICE_URLS.items():
    try:
        print(f"   ⬇️ Downloading env-var voice '{_env_name}' from {_env_url[:60]}...")
        _raw = _download_audio(_env_url)
        _wav_path = _convert_to_wav(_raw, label=f" '{_env_name}' (env)")
        _ENV_VOICE_PATHS[_env_name] = _wav_path
        # If this is overriding the default, also update the registry path cache
        if _env_name == "default":
            # Re-hash the new bytes so resolve_ref_audio finds it via bundled path too
            _h = hashlib.sha256(_raw).hexdigest()
            _AUDIO_CACHE[_h] = _wav_path
            print(f"   ✅ Env-var DEFAULT voice ready (overrides bundled): {_wav_path}")
        else:
            print(f"   ✅ Env-var voice '{_env_name}' ready: {_wav_path}")
    except Exception as _e:
        print(f"   ⚠️ Failed to load env-var voice '{_env_name}': {_e}")

try:
    load_target_model("voice_design")
    print("--- ⏱️ CUDA warmup ---")
    t0 = time.time()
    with torch.inference_mode():
        _ = CURRENT_MODEL.generate_voice_design(
            text="Hi.", language="English", instruct="Warmup.",
            max_new_tokens=50, use_cache=True,
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
# Request handler
# ──────────────────────────────────────────────────────────────────────────────
def handler(job):
    t_start = time.time()
    try:
        job_input = job.get("input", {})
        mode = job_input.get("mode", "voice_design").lower()

        # ── Transcribe mode ──────────────────────────────────────────────────
        if mode == "transcribe":
            audio_b64 = job_input.get("audio_base64")
            if not audio_b64:
                return {"error": "No audio_base64 provided."}
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
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        if mode not in MODEL_IDS:
            return {"error": f"Unknown mode '{mode}'. Valid: {list(MODEL_IDS.keys())}"}

        text = job_input.get("text", "")
        if not text:
            return {"error": "No text provided."}

        user_max = job_input.get("max_new_tokens")
        max_tokens = int(user_max) if user_max else 1024

        model = load_target_model(mode)

        print(f"--- 🗣️ GEN: {mode} | Len:{len(text)} | Max:{max_tokens} | Cache:True ---")
        t_gen = time.time()

        with torch.inference_mode():
            gen_args = dict(
                text=text,
                language=job_input.get("language", "English"),
                max_new_tokens=max_tokens,
                use_cache=True,
            )

            if mode == "voice_design":
                gen_args["instruct"] = job_input.get("instruct", "Clear voice.")
                wavs, sr = model.generate_voice_design(**gen_args)

            elif mode == "voice_clone":
                # ── ref_audio resolution ─────────────────────────────────────
                # Accepts:
                #   "default"          → bundled default voice (fastest)
                #   "<voice_name>"     → other bundled voice (fastest)
                #   "<base64_string>"  → custom audio encoded as base64
                #   "data:audio/..."   → data URI
                #   "https://..."      → URL (slow fallback)
                raw_ref = job_input.get("ref_audio")  # None → uses default
                ref_path, is_temp = resolve_ref_audio(raw_ref)

                gen_args["ref_audio"] = ref_path
                gen_args["ref_text"] = job_input.get("ref_text")
                try:
                    wavs, sr = model.generate_voice_clone(**gen_args)
                finally:
                    # Only delete if it was a one-shot temp file, not a cached/bundled one
                    if is_temp and ref_path and os.path.exists(ref_path):
                        # Don't delete if it ended up in the cache (reused later)
                        if ref_path not in _AUDIO_CACHE.values():
                            os.remove(ref_path)

        torch.cuda.synchronize()
        dt_gen = time.time() - t_gen
        print(f"⏱️ Generation took {dt_gen:.2f}s")

        # ── Encode output ─────────────────────────────────────────────────────
        raw_audio = wavs[0]
        if isinstance(raw_audio, np.ndarray):
            audio_tensor = torch.from_numpy(raw_audio).float()
        elif torch.is_tensor(raw_audio):
            audio_tensor = raw_audio.detach().cpu().float()

        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        byte_io = io.BytesIO()
        torchaudio.save(byte_io, audio_tensor, sr, format="wav")
        audio_b64 = base64.b64encode(byte_io.getvalue()).decode("utf-8")

        dt_total = time.time() - t_start
        print(f"✅ Complete in {dt_total:.2f}s (Gen: {dt_gen:.2f}s)")
        return {
            "status": "success",
            "audio_base64": audio_b64,
            "stats": {
                "total_time": f"{dt_total:.2f}s",
                "generation_time": f"{dt_gen:.2f}s",
            },
        }

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"❌ Error: {err}")
        return {"error": str(e), "traceback": err}


runpod.serverless.start({"handler": handler})