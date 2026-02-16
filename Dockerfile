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

# --- 3. INSTALL FLASH ATTENTION (Fast Method) ---
# The cu122 pre-built wheel is used here. Although the base image is CUDA 12.1.1,
# CUDA is backwards compatible so cu122 wheels run correctly on cu121 runtimes.
# A cu121-specific wheel does not exist for this flash-attention release.
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# --- 4. COPY CODE ---
COPY . .

# --- 5. CACHE MODELS ---
# Models are baked into the image so the server starts without downloading.
RUN python3 download_models.py && rm download_models.py

# --- 6. START ---
ENV OMP_NUM_THREADS=4
CMD [ "python3", "-u", "handler.py" ]
