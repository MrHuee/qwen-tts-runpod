FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# --- 1. SYSTEM DEPENDENCIES ---
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    sox \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# --- 2. PYTHON DEPENDENCIES ---
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- 3. INSTALL FLASH ATTENTION (pre-built wheel — fast) ---
# cu122 wheel is used here; CUDA 12.1 runtime is backwards-compatible with it.
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# --- 4. COPY CODE ---
COPY . .

# --- 5. CACHE MODELS AT BUILD TIME ---
# Bakes weights into the image so workers start without downloading.
RUN python3 download_models.py && rm download_models.py

# --- 6. RUNTIME ENVIRONMENT ---
# Keep CPU threads low — they don't help GPU work and compete with CUDA streams.
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
# Avoid CUDA device re-init overhead across fork/spawn boundaries
ENV CUDA_LAUNCH_BLOCKING=0
# Let cuDNN auto-select the fastest algorithms for fixed-size conv inputs (vocoder)
ENV CUDNN_BENCHMARK=1

# --- 7. START ---
CMD [ "python3", "-u", "handler.py" ]
