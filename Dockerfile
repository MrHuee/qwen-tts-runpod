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

# --- 3. INSTALL FLASH ATTENTION (The Fix) ---
# We limit the build to 4 parallel jobs to prevent running out of RAM (OOM Killed error).
ENV MAX_JOBS=4
RUN pip install flash-attn --no-build-isolation

# --- 4. COPY CODE ---
COPY . .

# --- 5. CACHE MODELS ---
# This ensures models are downloaded into the image, so the server starts faster.
RUN python3 download_models.py && rm download_models.py

# --- 6. START ---
# Limit runtime threads to avoid CPU choking (your existing fix)
ENV OMP_NUM_THREADS=4
CMD [ "python3", "-u", "handler.py" ]