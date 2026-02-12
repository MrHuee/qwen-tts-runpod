FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# --- 1. SYSTEM DEPENDENCIES ---
# Added 'sox' (fixes the log error) and 'ninja-build' (critical for compiling flash-attn)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    sox \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# --- 2. PYTHON DEPENDENCIES ---
COPY requirements.txt .
# Split install to ensure flash-attn compiles correctly
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- 3. COPY CODE ---
COPY . .

# --- 4. CACHE MODELS ---
RUN python3 download_models.py && rm download_models.py

# --- 5. START ---
# Added explicit thread limit here too as a safety net
ENV OMP_NUM_THREADS=4
CMD [ "python3", "-u", "handler.py" ]