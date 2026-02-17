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
    
    # Just verify main device
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
# 🔥 STARTUP PRELOAD
# ──────────────────────────────────────────────────────────────────────────────
print("--- 🔥 Preloading models... ---")
try:
    load_target_model("voice_design")
    # Warmup with explicit max tokens to test
    print("--- ⏱️ CUDA warmup ---")
    t0 = time.time()
    with torch.inference_mode():
        _ = CURRENT_MODEL.generate_voice_design(
            text="Hi.", 
            language="English", 
            instruct="Warmup.",
            max_new_tokens=50 # Force short warmup
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

        # Dynamic max_new_tokens to prevent runaway generation (EOS failure)
        # 12Hz model = 12 tokens/sec. 
        # Safety margin: allow 1 second per character + 5s buffer worth of tokens
        # 1 char ~ 12 tokens? No, that's excessive. 
        # Let's simple cap at a reasonable large limit, or user provided.
        # Default to 1024 (85 seconds of audio) if not specified.
        user_max_tokens = job_input.get("max_new_tokens")
        if user_max_tokens:
            max_tokens = int(user_max_tokens)
        else:
            # Conservative auto-limit: 
            # 12 tokens/sec * (len(text)*0.3 + 5 sec) ?
            # Let's just use a hard safety cap of 1536 (approx 2 mins) 
            # OR better: 512 (42s) for typical usage to prove speed fix.
            max_tokens = 1024 

        model = load_target_model(mode)
        
        print(f"--- 🗣️ Generating ({mode}) | Len: {len(text)} | MaxTokens: {max_tokens} ---")
        t_gen = time.time()
        
        with torch.inference_mode():
            if mode == "voice_design":
                wavs, sr = model.generate_voice_design(
                    text=text, 
                    language=job_input.get("language", "English"),
                    instruct=job_input.get("instruct", "Clear voice."),
                    max_new_tokens=max_tokens
                )
            elif mode == "custom_voice":
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language=job_input.get("language", "English"),
                    speaker=job_input.get("speaker", "Anna"),
                    max_new_tokens=max_tokens
                )
            elif mode == "voice_clone":
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=job_input.get("language", "English"),
                    ref_audio=job_input.get("ref_audio"),
                    ref_text=job_input.get("ref_text"),
                    max_new_tokens=max_tokens
                )

        torch.cuda.synchronize()
        dt_gen = time.time() - t_gen
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
        print(f"✅ Complete in {dt_total:.2f}s")
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
