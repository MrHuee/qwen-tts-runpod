import runpod
import torch
import torchaudio
import io
import base64
import gc
import os
import tempfile
import numpy as np

# --- ⚡ THREAD LIMITING ---
# Keep CPU threads low so they don't steal GPU work from CUDA streams
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# --- ✅ CUDA SETTINGS ---
# Avoid CUDA re-init overhead; allow TF32 for slight speedup on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True   # auto-tune convolution algorithms (helps vocoder)

# --- Configuration ---
MODEL_IDS = {
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_clone":  "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

CURRENT_MODEL = None
CURRENT_MODE = None
WHISPER_MODEL = None


def _force_model_to_gpu(model):
    """
    Force ALL submodules — including the vocoder — onto CUDA.

    `device_map="cuda:0"` in from_pretrained only moves the transformer
    (LLM/Talker) layers. The vocoder is a plain nn.Module that the HuggingFace
    device-map logic skips, so it silently runs on CPU. That's the root cause
    of CPU=100% / GPU=26%. We walk every named child and move anything still on
    CPU to cuda:0 in bfloat16.
    """
    for name, module in model.named_children():
        try:
            # Check where this submodule currently lives
            params = list(module.parameters())
            if params and params[0].device.type == "cpu":
                print(f"  ↳ Moving {name} to cuda:0 (was on CPU)")
                module.to(device="cuda:0", dtype=torch.bfloat16)
        except Exception as e:
            print(f"  ⚠️  Could not move {name} to GPU: {e}")


def _warmup_model(model, mode):
    """
    Run a short dummy inference to trigger CUDA kernel compilation.
    This ensures the first real request isn't penalised by JIT overhead.
    """
    print("--- 🔥 Warming up model (dummy inference)... ---")
    try:
        with torch.inference_mode():
            if mode == "voice_design":
                model.generate_voice_design(
                    text="Hi.", language="English", instruct="Clear voice."
                )
            elif mode == "custom_voice":
                model.generate_custom_voice(
                    text="Hi.", language="English", speaker="Anna"
                )
            # voice_clone needs ref audio — skip warmup for that mode
        print("✅ Warmup complete.")
    except Exception as e:
        print(f"⚠️  Warmup skipped: {e}")


def load_target_model(target_mode):
    """Load the requested TTS model, force vocoder to GPU, and warm up."""
    global CURRENT_MODEL, CURRENT_MODE

    if CURRENT_MODE == target_mode and CURRENT_MODEL is not None:
        return CURRENT_MODEL

    # Unload the previous model if switching modes
    if CURRENT_MODEL is not None:
        print(f"--- 🔄 Switching model from {CURRENT_MODE} → {target_mode} ---")
        del CURRENT_MODEL
        CURRENT_MODEL = None
        gc.collect()
        torch.cuda.empty_cache()

    model_id = MODEL_IDS.get(target_mode)
    if not model_id:
        raise ValueError(f"Invalid mode: '{target_mode}'. Must be one of: {list(MODEL_IDS.keys())}")

    print(f"--- 🚀 Loading {target_mode} ({model_id}) ---")

    from qwen_tts import Qwen3TTSModel

    try:
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        print("✅ Loaded: bfloat16 + Flash Attention 2")
    except Exception as e:
        print(f"⚠️  FA2 failed ({e}), retrying with standard attention...")
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        print("✅ Loaded: bfloat16 + standard attention")

    # ✅ KEY FIX: Push vocoder (and any other CPU-resident submodules) to GPU
    print("--- 🔧 Verifying all submodules are on GPU... ---")
    _force_model_to_gpu(model)

    # Put the model in eval mode explicitly
    model.eval()

    CURRENT_MODEL = model
    CURRENT_MODE = target_mode

    # Warm up to pre-compile CUDA kernels (no JIT penalty on first real job)
    _warmup_model(model, target_mode)

    return model


def get_whisper_model():
    """Load Whisper on first use."""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        import whisper_timestamped as whisper
        print("--- 🎙️ Loading Whisper base on GPU ---")
        WHISPER_MODEL = whisper.load_model("base", device="cuda")
        print("✅ Whisper ready")
    return WHISPER_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# Preload voice_design at startup — the most common mode.
# We do this INSIDE a try/except so a load failure never crashes the worker.
# If it fails, the first request will trigger lazy-loading (still safe).
# ─────────────────────────────────────────────────────────────────────────────
print("--- ⚡ Pre-loading voice_design at startup ---")
try:
    load_target_model("voice_design")
    print("✅ voice_design model ready — worker hot.")
except Exception as e:
    import traceback
    print(f"⚠️  Startup preload failed (will lazy-load on first request):\n{traceback.format_exc()}")


def handler(job):
    try:
        job_input = job.get("input", {})
        mode = job_input.get("mode", "voice_design").lower()

        # --- TRANSCRIPTION ---
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

        # --- TTS GENERATION ---
        if mode not in MODEL_IDS:
            return {"error": f"Unknown mode '{mode}'. Valid: {list(MODEL_IDS.keys()) + ['transcribe']}"}

        text = job_input.get("text")
        if not text:
            return {"error": "No 'text' provided."}

        language = job_input.get("language", "English")

        model = load_target_model(mode)

        print(f"--- 🗣️ Generating audio (mode={mode}, len={len(text)}) ---")

        # torch.inference_mode() disables gradient tracking and autograd
        # overhead — faster and uses less VRAM than torch.no_grad()
        with torch.inference_mode():
            if mode == "voice_design":
                instruct = job_input.get("instruct", "Clear voice.")
                wavs, sr = model.generate_voice_design(
                    text=text, language=language, instruct=instruct
                )
            elif mode == "custom_voice":
                speaker = job_input.get("speaker", "Anna")
                wavs, sr = model.generate_custom_voice(
                    text=text, language=language, speaker=speaker
                )
            elif mode == "voice_clone":
                ref_audio = job_input.get("ref_audio")
                ref_text  = job_input.get("ref_text")
                if not ref_audio or not ref_text:
                    return {"error": "voice_clone requires 'ref_audio' and 'ref_text'."}
                wavs, sr = model.generate_voice_clone(
                    text=text, language=language, ref_audio=ref_audio, ref_text=ref_text
                )

        # --- Encode audio ---
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
