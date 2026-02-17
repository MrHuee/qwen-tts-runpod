import runpod
import torch
import torchaudio
import io
import base64
import gc
import os
import tempfile
import numpy as np
import time
import warnings

# ──────────────────────────────────────────────────────────────────────────────
# ⚡ THREAD / ENV TUNING
# ──────────────────────────────────────────────────────────────────────────────
# Reduce OpenMP/MKL threads to avoid CPU contention.
# Since we are GPU-bound, 1-2 threads is optimal for data loading/preprocessing.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Suppress warnings to keep logs clean
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
    """Load a Qwen3-TTS model with improved device consistency checks."""
    from qwen_tts import Qwen3TTSModel

    print(f"--- 🛠️ Loading {model_id} (Flash={use_flash}) ---")
    
    # Pass both dtype args to satisfy Qwen wrapper AND HuggingFace internals
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

    # 🔍 DEEP INSPECTION: extensive check of where submodules are living
    try:
        inner = model.model  # Qwen3TTSForConditionalGeneration
        first_param = next(inner.parameters())
        print(f"   📊 Main Model: dtype={first_param.dtype}, device={first_param.device}")

        # Check critical subcomponents
        components = ["code_predictor", "speech_tokenizer", "talker"]
        for name in components:
            if hasattr(inner, name):
                mod = getattr(inner, name)
                # Some submodules might be wrapped or just config
                if hasattr(mod, "parameters"):
                    try:
                        p = next(mod.parameters())
                        print(f"   🔍 {name}: device={p.device}, dtype={p.dtype}")
                        # Force move if on CPU
                        if p.device.type == "cpu":
                            print(f"   ⚠️ {name} is on CPU! Moving to cuda:0...")
                            mod.to("cuda:0")
                    except StopIteration:
                        print(f"   ⚠️ {name} has no parameters?")
                else:
                    print(f"   ℹ️ {name} found but no parameters (might be config/helper).")
            else:
                print(f"   ❓ {name} not found in model.")

    except Exception as e:
        print(f"   ⚠️ Failed to inspect model internals: {e}")

    return model


def load_target_model(target_mode):
    """Load the requested TTS model onto GPU. Caches across calls."""
    global CURRENT_MODEL, CURRENT_MODE

    if CURRENT_MODE == target_mode and CURRENT_MODEL is not None:
        return CURRENT_MODEL

    if CURRENT_MODEL is not None:
        print("--- ♻️ Unloading previous model ---")
        del CURRENT_MODEL
        CURRENT_MODEL = None
        gc.collect()
        torch.cuda.empty_cache()

    model_id = MODEL_IDS.get(target_mode)
    if not model_id:
        raise ValueError(f"Invalid mode: '{target_mode}'.")

    try:
        model = _load_tts(model_id, use_flash=True)
        print("✅ Loaded: bfloat16 + Flash Attention 2")
    except Exception as e:
        print(f"⚠️ FA2 failed ({e}), retrying with standard attention...")
        model = _load_tts(model_id, use_flash=False)
        print("✅ Loaded: bfloat16 + standard attention")

    CURRENT_MODEL = model
    CURRENT_MODE = target_mode
    return model


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
print("--- 🔥 Preloading models at startup... ---")
try:
    load_target_model("voice_design")
    # Warmup
    print("--- ⏱️ CUDA warmup (voice_design) ---")
    t0 = time.time()
    with torch.inference_mode():
        _ = CURRENT_MODEL.generate_voice_design(
            text="Hi.", language="English", instruct="Verify device placement."
        )
    torch.cuda.synchronize()
    print(f"✅ Warmup done in {time.time() - t0:.2f}s")
except Exception as e:
    print(f"⚠️ Startup preload/warmup failed: {e}")

try:
    get_whisper_model()
except Exception as e:
    print(f"⚠️ Whisper preload failed: {e}")

print("--- ✅ Startup Complete ---")


# ──────────────────────────────────────────────────────────────────────────────
# Request handler
# ──────────────────────────────────────────────────────────────────────────────
def handler(job):
    t_handler_start = time.time()
    print(f"--- 🏁 Handler started at {t_handler_start} ---")

    try:
        job_input = job.get("input", {})
        mode = job_input.get("mode", "voice_design").lower()

        # ── TRANSCRIPTION ────────────────────────────────────────────────
        if mode == "transcribe":
            audio_b64 = job_input.get("audio_base64")
            if not audio_b64:
                return {"error": "No audio_base64 provided."}

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
                print(f"⏱️ Transcription took {time.time() - t_trans:.2f}s")
                return {"status": "success", "transcription": result}
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # ── TTS GENERATION ───────────────────────────────────────────────
        if mode not in MODEL_IDS:
            return {"error": f"Unknown mode '{mode}'."}

        text = job_input.get("text")
        if not text:
            return {"error": "No text provided."}

        # Timer: Model Load
        t_load = time.time()
        model = load_target_model(mode)
        print(f"⏱️ Model access (cached/load) took {time.time() - t_load:.2f}s")

        print(f"--- 🗣️ Generating audio (mode={mode}) ---")
        t_gen_start = time.time()

        with torch.inference_mode():
            if mode == "voice_design":
                instruct = job_input.get("instruct", "Clear voice.")
                wavs, sr = model.generate_voice_design(
                    text=text, language=job_input.get("language", "English"), instruct=instruct,
                )
            elif mode == "custom_voice":
                speaker = job_input.get("speaker", "Anna")
                wavs, sr = model.generate_custom_voice(
                    text=text, language=job_input.get("language", "English"), speaker=speaker,
                )
            elif mode == "voice_clone":
                ref_audio = job_input.get("ref_audio")
                ref_text = job_input.get("ref_text")
                wavs, sr = model.generate_voice_clone(
                    text=text, language=job_input.get("language", "English"),
                    ref_audio=ref_audio, ref_text=ref_text,
                )

        torch.cuda.synchronize()
        t_gen_end = time.time()
        print(f"⏱️ Generation (inference + sync) took {t_gen_end - t_gen_start:.2f}s")

        # ── Encoding ─────────────────────────────────────────────────────
        t_enc_start = time.time()
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
        print(f"⏱️ Encoding took {time.time() - t_enc_start:.2f}s")

        t_total = time.time() - t_handler_start
        print(f"✅ Handler complete in {t_total:.2f}s")

        return {
            "status": "success",
            "audio_base64": base64.b64encode(byte_io.getvalue()).decode("utf-8"),
            "stats": {
                "generation_time": f"{t_gen_end - t_gen_start:.2f}s",
                "total_time": f"{t_total:.2f}s"
            }
        }

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"❌ Handler error:\n{err}")
        return {"error": str(e), "traceback": err}

runpod.serverless.start({"handler": handler})
