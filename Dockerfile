FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# --- 1. SYSTEM DEPENDENCIES ---
# Install ffmpeg (critical for Whisper) and git
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- 2. PYTHON DEPENDENCIES ---
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# --- 3. COPY CODE ---
COPY . .

# --- 4. CACHE MODELS ---
# Run the external script we just created.
# This avoids the IndentationError caused by multi-line Docker commands.
RUN python3 download_models.py && rm download_models.py

# --- 5. START ---
CMD [ "python3", "-u", "handler.py" ]