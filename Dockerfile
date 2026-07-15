FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies including RNNoise build requirements
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libsndfile1 \
    ffmpeg \
    # RNNoise build dependencies
    cmake \
    autoconf \
    automake \
    libtool \
    pkg-config \
    git \
    # Audio processing libraries
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Set OpenBLAS to use a safe CPU type (prevents crashes on some AMD CPUs)
ENV OPENBLAS_CORETYPE=Haswell

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml uv.lock requirements.lock.txt ./

# Install Python dependencies pinned to production versions (reproducible build).
# Previously these were installed unpinned, which silently pulled newer
# livekit-agents on each rebuild and broke the app against API changes.
RUN uv pip install --system -r requirements.lock.txt

COPY ./AUDIO_DIR ./AUDIO_DIR
COPY main.py server.py prompt.py schemas.py \
    events.py noise_cancellation.py audio_player.py \
    audio_responses.py session_monitor.py audio_processor.py ./

RUN python main.py download-files

CMD ["python", "main.py", "start"]
