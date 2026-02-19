FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# ─────────────────────────────────────────────────────────────────────────────
# LAYER ORDER MATTERS FOR CACHING
# Layers 1-5 are heavy (30 min total). They are FROZEN once built.
# Layers 6-7 are lightweight. Only these rebuild when you change code or voices.
#
#   Change handler.py  → rebuilds layer 6 + 7  (~5s)
#   Change voices/     → rebuilds layer 7 only  (~5s)
#   Change requirements.txt → rebuilds layer 2 onward (full rebuild)
# ─────────────────────────────────────────────────────────────────────────────

WORKDIR /app

# ── LAYER 1: System packages (frozen) ────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    sox \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# ── LAYER 2: Python dependencies (frozen — only changes if requirements.txt changes) ─
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── LAYER 3: Flash Attention wheel (frozen) ──────────────────────────────────
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# ── LAYER 4: Download & bake models (frozen — only changes if download_models.py changes) ─
COPY download_models.py .
RUN python3 download_models.py && rm download_models.py

# ── LAYER 5: ENV tuning (frozen) ─────────────────────────────────────────────
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# ─────────────────────────────────────────────────────────────────────────────
# ⬇  FAST LAYERS — only these rebuild on code or voice changes
# ─────────────────────────────────────────────────────────────────────────────

# ── LAYER 6: Application code (~1s) ──────────────────────────────────────────
COPY handler.py .

# ── LAYER 7: Bundled voices (~5s depending on file size) ─────────────────────
# To add a voice without rebuilding everything:
#   1. Drop the file into voices/
#   2. Add it to VOICE_REGISTRY in handler.py
#   3. Push to GitHub → RunPod rebuilds only this layer
#
# Runtime override (NO rebuild needed at all):
#   Set env var VOICE_<NAME>_URL=https://... in RunPod dashboard
#   e.g. VOICE_DEFAULT_URL=https://your-cdn.com/voice.mp3
#   handler.py downloads & caches it at startup automatically.
RUN mkdir -p /app/voices
COPY voices/ /app/voices/

# ── START ─────────────────────────────────────────────────────────────────────
CMD [ "python3", "-u", "handler.py" ]
