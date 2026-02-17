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
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

# --- Configuration ---
MODEL_IDS = {
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_clone":  "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

CURRENT_MODEL = None
CURRENT_MODE = None
WHISPER_MODEL = None

def load_target_model(target_mode):
    """Load the requested TTS model. Raises on failure — caller handles."""
    global CURRENT_MODEL, CURRENT_MODE

    if CURRENT_MODE == target_mode and CURRENT_MODEL is not None:
        return CURRENT_MODEL

    if CURRENT_MODEL is not None:
        del CURRENT_MODEL
        CURRENT_MODEL = None
        gc.collect()
        torch.cuda.empty_cache()

    model_id = MODEL_IDS.get(target_mode)
    if not model_id:
        raise ValueError(f"Invalid mode: '{target_mode}'. Must be one of: {list(MODEL_IDS.keys())}")

    print(f"--- 🚀 Loading {target_mode} ({model_id}) ---")

    # Official Qwen3-TTS API from HuggingFace docs:
    # dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="flash_attention_2"
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
        print(f"⚠️ FA2 failed ({e}), retrying with standard attention...")
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        print("✅ Loaded: bfloat16 + standard attention")

    CURRENT_MODEL = model
    CURRENT_MODE = target_mode
    return model


def get_whisper_model():
    """Load Whisper on first use. Raises on failure — caller handles."""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        import whisper_timestamped as whisper
        print("--- 🎙️ Loading Whisper base on GPU ---")
        WHISPER_MODEL = whisper.load_model("base", device="cuda")
        print("✅ Whisper ready")
    return WHISPER_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: No model loading at module level.
#
# Previously, preloading models here caused the worker process to crash on
# startup whenever model loading failed (wrong dtype, missing attr, etc).
# RunPod then restarts the worker, it crashes again, and after several retries
# marks ALL workers as permanently unhealthy — breaking the entire endpoint.
#
# With lazy loading, the process starts cleanly every time. If a model fails
# to load, the handler returns a JSON error to the caller instead of crashing.
# Workers stay healthy regardless of model load failures.
# ─────────────────────────────────────────────────────────────────────────────

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

        print(f"--- 🗣️ Generating audio (mode={mode}) ---")

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
