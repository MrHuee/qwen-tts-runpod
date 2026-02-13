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

# Replace step 3 with this:
# --- 3. INSTALL FLASH ATTENTION (Fast Method) ---
# Instead of compiling (slow), we download the pre-built wheel for this specific CUDA/Torch version.
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# --- 4. COPY CODE ---
COPY . .

# --- 5. CACHE MODELS ---
# This ensures models are downloaded into the image, so the server starts faster.
RUN python3 download_models.py && rm download_models.py

# --- 6. START ---
# Limit runtime threads to avoid CPU choking (your existing fix)
ENV OMP_NUM_THREADS=4
CMD [ "python3", "-u", "handler.py" ]