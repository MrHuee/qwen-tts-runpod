FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# --- 1. SYSTEM DEPENDENCIES ---
# We verify git, ninja-build, and sox are here
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
# We use --no-build-isolation so it sees the pre-installed PyTorch
RUN pip install flash-attn --no-build-isolation

# --- 4. COPY CODE ---
COPY . .

# --- 5. CACHE MODELS ---
RUN python3 download_models.py && rm download_models.py

# --- 6. START ---
ENV OMP_NUM_THREADS=4
CMD [ "python3", "-u", "handler.py" ]