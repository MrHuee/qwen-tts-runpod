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
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_IDS = {
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_clone":  "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

CURRENT_MODEL = None
CURRENT_MODE = None
WHISPER_MODEL = None

# ──────────────────────────────────────────────────────────────────────────────
# Helper: Aggressive Device Moving
# ──────────────────────────────────────────────────────────────────────────────
def recursive_move_to_device(obj, device="cuda:0", verbose_prefix=""):
    """
    Recursively inspect object attributes to find PyTorch modules/tensors
    residing on CPU and force them to GPU.
    """
    # 1. If it's a module/tensor, move it
    if hasattr(obj, "to") and hasattr(obj, "device"):
        if obj.device.type == "cpu":
            if verbose_prefix:
                print(f"{verbose_prefix} Found CPU component {type(obj).__name__}, moving...")
            try:
                obj.to(device)
            except Exception as e:
                print(f"{verbose_prefix} ❌ Failed to move: {e}")
        return

    # 2. If it's a module without .device property but has parameters
    if hasattr(obj, "parameters") and callable(obj.parameters):
        try:
            first = next(obj.parameters())
            if first.device.type == "cpu":
                if verbose_prefix:
                    print(f"{verbose_prefix} Found CPU module {type(obj).__name__}, moving...")
                obj.to(device)
        except StopIteration:
            pass 
        except Exception:
            pass

    # 3. Recurse into attributes
    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            if k.startswith("__"): continue
            if isinstance(v, (int, float, str, bool, list, tuple, dict)):
                continue
            # Recurse
            recursive_move_to_device(v, device, verbose_prefix=f"{verbose_prefix}.{k}")

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
    
    # First pass: move static components
    try:
        print("   🚜 Device Fix Loop 1 (Load Time)...")
        recursive_move_to_device(model.model, "cuda:0", verbose_prefix="   [Fix1]")
    except Exception as e:
        print(f"   ⚠️ Fix1 failed: {e}")

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
    if not model_id: raise ValueError(f"Invalid mode: '{target_mode}'")

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
# 🔥 STARTUP PRELOAD AND FIX STRATEGY
# ──────────────────────────────────────────────────────────────────────────────
print("--- 🔥 Preloading models... ---")
try:
    load_target_model("voice_design")
    
    print("--- ⏱️ CUDA warmup (triggers lazy init) ---")
    t0 = time.time()
    with torch.inference_mode():
        # This triggers code_predictor initialization if it wasn't loaded
        _ = CURRENT_MODEL.generate_voice_design(
            text="Hi.", language="English", instruct="Warmup."
        )
    print(f"✅ Warmup done in {time.time() - t0:.2f}s")

    # 🚜 SECOND PASS: Catch lazy-loaded modules (code_predictor)
    print("--- 🚜 Device Fix Loop 2 (Post-Warmup) ---")
    recursive_move_to_device(CURRENT_MODEL.model, "cuda:0", verbose_prefix="   [Fix2]")
    
    # Inspect final state
    try:
        inner = CURRENT_MODEL.model
        if hasattr(inner, "code_predictor"):
             # It assumes code_predictor is a module or object with parameters
             cp = inner.code_predictor
             if hasattr(cp, "parameters"):
                 p = next(cp.parameters())
                 print(f"   🔍 Final code_predictor device: {p.device}")
             else:
                 print("   🔍 code_predictor has no parameters?")
        else:
             print("   ❓ code_predictor still not found in model.")
    except Exception as e:
        print(f"Inspection error: {e}")

except Exception as e:
    print(f"⚠️ Startup sequence failed: {e}")

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
    print(f"--- 🏁 Handler started at {t_start} ---")

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
                print(f"⏱️ Transcription took {time.time() - t_trans:.2f}s")
                return {"status": "success", "transcription": result}
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)

        if mode not in MODEL_IDS: return {"error": f"Unknown mode {mode}"}

        text = job_input.get("text", "")
        if not text: return {"error": "No text."}

        t_load = time.time()
        model = load_target_model(mode)
        print(f"⏱️ Model access took {time.time() - t_load:.2f}s")

        print(f"--- 🗣️ Generating ({mode}) ---")
        t_gen = time.time()
        
        with torch.inference_mode():
            if mode == "voice_design":
                wavs, sr = model.generate_voice_design(
                    text=text, 
                    language=job_input.get("language", "English"),
                    instruct=job_input.get("instruct", "Clear voice.")
                )
            elif mode == "custom_voice":
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language=job_input.get("language", "English"),
                    speaker=job_input.get("speaker", "Anna")
                )
            elif mode == "voice_clone":
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=job_input.get("language", "English"),
                    ref_audio=job_input.get("ref_audio"),
                    ref_text=job_input.get("ref_text")
                )

        torch.cuda.synchronize()
        print(f"⏱️ Generation took {time.time() - t_gen:.2f}s")

        # Encode
        t_enc = time.time()
        raw_audio = wavs[0]
        if isinstance(raw_audio, np.ndarray):
            audio_tensor = torch.from_numpy(raw_audio).float()
        elif torch.is_tensor(raw_audio):
             audio_tensor = raw_audio.detach().cpu().float()
        
        if audio_tensor.dim() == 1: audio_tensor = audio_tensor.unsqueeze(0)
        
        byte_io = io.BytesIO()
        torchaudio.save(byte_io, audio_tensor, sr, format="wav")
        audio_b64 = base64.b64encode(byte_io.getvalue()).decode("utf-8")
        print(f"⏱️ Encoding took {time.time() - t_enc:.2f}s")

        dt_total = time.time() - t_start
        print(f"✅ Complete in {dt_total:.2f}s")
        return {
            "status": "success", 
            "audio_base64": audio_b64,
            "stats": {"total_time": f"{dt_total:.2f}s"}
        }

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"❌ Error: {err}")
        return {"error": str(e), "traceback": err}

runpod.serverless.start({"handler": handler})
