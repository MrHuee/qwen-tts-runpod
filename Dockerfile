FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# --- 1. SYSTEM DEPENDENCIES ---
# Added 'ffmpeg' (Required for Whisper) and 'git' (Useful for pip installing from git)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- 2. PYTHON DEPENDENCIES ---
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# --- 3. COPY CODE ---
# We copy code EARLIER so we can use it for the caching step below
COPY . .

# --- 4. CACHE MODELS (The "Baker's Dozen" Step) ---
# We download ALL models (VoiceDesign, Custom, Clone, and Whisper) 
# so the user never waits during a live request.
RUN python3 -c " \
import torch; \
from qwen_tts import Qwen3TTSModel; \
import whisper_timestamped as whisper; \
\
print('--- Downloading Qwen VoiceDesign ---'); \
Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign'); \
\
print('--- Downloading Qwen CustomVoice ---'); \
Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'); \
\
print('--- Downloading Qwen Base (Clone) ---'); \
Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base'); \
\
print('--- Downloading Whisper ---'); \
whisper.load_model('base', device='cpu'); \
"

# --- 5. START ---
CMD [ "python3", "-u", "handler.py" ]