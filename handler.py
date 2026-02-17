import runpod
import torch
import torchaudio
import io
import base64
import gc
import os
import tempfile
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# ⚡ THREAD / ENV TUNING  (GPU-bound workload → keep CPU threads low)
# ──────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
torch.set_num_threads(2)
torch.set_num_interop_threads(2)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
MODEL_IDS = {
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_clone":  "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

CURRENT_MODEL = None
CURRENT_MODE = None
WHISPER_MODEL = None


# ──────────────────────────────────────────────────────────────────────────────
# Model helpers
# ──────────────────────────────────────────────────────────────────────────────
def _load_tts(model_id, use_flash=True):
    """Load a Qwen3-TTS model with optional Flash Attention 2."""
    from qwen_tts import Qwen3TTSModel

    kwargs = dict(
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    if use_flash:
        kwargs["attn_implementation"] = "flash_attention_2"

    return Qwen3TTSModel.from_pretrained(model_id, **kwargs)


def load_target_model(target_mode):
    """Load the requested TTS model onto GPU. Caches across calls."""
    global CURRENT_MODEL, CURRENT_MODE

    # Already loaded → return immediately
    if CURRENT_MODE == target_mode and CURRENT_MODEL is not None:
        return CURRENT_MODEL

    # Unload previous model
    if CURRENT_MODEL is not None:
        del CURRENT_MODEL
        CURRENT_MODEL = None
        gc.collect()
        torch.cuda.empty_cache()

    model_id = MODEL_IDS.get(target_mode)
    if not model_id:
        raise ValueError(
            f"Invalid mode: '{target_mode}'. Must be one of: {list(MODEL_IDS.keys())}"
        )

    print(f"--- 🚀 Loading {target_mode} ({model_id}) ---")

    try:
        model = _load_tts(model_id, use_flash=True)
        print("✅ Loaded: bfloat16 + Flash Attention 2")
    except Exception as e:
        print(f"⚠️ FA2 failed ({e}), retrying with standard attention...")
        model = _load_tts(model_id, use_flash=False)
        print("✅ Loaded: bfloat16 + standard attention")

    model.eval()
    CURRENT_MODEL = model
    CURRENT_MODE = target_mode
    return model


def get_whisper_model():
    """Return the cached Whisper model, loading on first call."""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        import whisper_timestamped as whisper

        print("--- 🎙️ Loading Whisper base on GPU ---")
        WHISPER_MODEL = whisper.load_model("base", device="cuda")
        print("✅ Whisper ready")
    return WHISPER_MODEL


# ──────────────────────────────────────────────────────────────────────────────
# 🔥 STARTUP PRELOAD
#
# Preload the default TTS model + Whisper at import time so that the first
# real request hits a warm worker.  Wrapped in try/except so a failure here
# still lets the worker start (lazy fallback on first request).
# ──────────────────────────────────────────────────────────────────────────────
print("--- 🔥 Preloading models at startup... ---")

try:
    load_target_model("voice_design")
    # CUDA warmup — run a tiny forward pass to JIT-compile kernels
    print("--- ⏱️ CUDA warmup (voice_design) ---")
    with torch.inference_mode():
        _warm_wavs, _warm_sr = CURRENT_MODEL.generate_voice_design(
            text="Hi.",
            language="English",
            instruct="Clear voice.",
        )
    del _warm_wavs, _warm_sr
    torch.cuda.empty_cache()
    print("✅ Warmup done")
except Exception as e:
    print(f"⚠️ Startup preload/warmup failed ({e}); will load lazily.")

try:
    get_whisper_model()
except Exception as e:
    print(f"⚠️ Whisper preload failed ({e}); will load lazily.")

print("--- ✅ Startup preload complete ---")


# ──────────────────────────────────────────────────────────────────────────────
# Request handler
# ──────────────────────────────────────────────────────────────────────────────
def handler(job):
    try:
        job_input = job.get("input", {})
        mode = job_input.get("mode", "voice_design").lower()

        # ── TRANSCRIPTION ────────────────────────────────────────────────
        if mode == "transcribe":
            audio_b64 = job_input.get("audio_base64")
            if not audio_b64:
                return {"error": "No audio_base64 provided for transcription."}

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

        # ── TTS GENERATION ───────────────────────────────────────────────
        if mode not in MODEL_IDS:
            return {
                "error": f"Unknown mode '{mode}'. Valid: {list(MODEL_IDS.keys()) + ['transcribe']}"
            }

        text = job_input.get("text")
        if not text:
            return {"error": "No 'text' provided."}

        language = job_input.get("language", "English")
        model = load_target_model(mode)

        print(f"--- 🗣️ Generating audio (mode={mode}) ---")

        with torch.inference_mode():
            if mode == "voice_design":
                instruct = job_input.get("instruct", "Clear voice.")
                wavs, sr = model.generate_voice_design(
                    text=text, language=language, instruct=instruct,
                )
            elif mode == "custom_voice":
                speaker = job_input.get("speaker", "Anna")
                wavs, sr = model.generate_custom_voice(
                    text=text, language=language, speaker=speaker,
                )
            elif mode == "voice_clone":
                ref_audio = job_input.get("ref_audio")
                ref_text = job_input.get("ref_text")
                if not ref_audio or not ref_text:
                    return {"error": "voice_clone requires 'ref_audio' and 'ref_text'."}
                wavs, sr = model.generate_voice_clone(
                    text=text, language=language,
                    ref_audio=ref_audio, ref_text=ref_text,
                )

        # Wait for GPU to finish before encoding
        torch.cuda.synchronize()

        # ── Encode audio to base64 WAV ───────────────────────────────────
        raw_audio = wavs[0]
        if isinstance(raw_audio, np.ndarray):
            audio_tensor = torch.from_numpy(raw_audio).float()
        elif torch.is_tensor(raw_audio):
            audio_tensor = raw_audio.detach().cpu().float()
        else:
            raise ValueError(f"Unexpected audio type: {type(raw_audio)}")

        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        byte_io = io.BytesIO()
        torchaudio.save(byte_io, audio_tensor, sr, format="wav")

        print("✅ Audio generated successfully.")
        return {
            "status": "success",
            "audio_base64": base64.b64encode(byte_io.getvalue()).decode("utf-8"),
        }

    except Exception as e:
        import traceback

        err = traceback.format_exc()
        print(f"❌ Handler error:\n{err}")
        return {"error": str(e), "traceback": err}


runpod.serverless.start({"handler": handler})
