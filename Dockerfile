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

COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv pip install --system \
    'livekit-agents[silero,turn-detector]' \
    livekit-plugins-google \
    livekit-plugins-silero \
    python-dotenv \
    google-genai \
    redisvl \
    'fastapi[all]' \
    # Noise cancellation - try pyrnnoise first, fall back to noisereduce
    pyrnnoise || echo "pyrnnoise not available"

# Install noisereduce as fallback (lighter weight, still effective)
RUN uv pip install --system \
    noisereduce \
    scipy \
    numpy

COPY main.py server.py prompt.py schemas.py events.py noise_cancellation.py ./

RUN python main.py download-files

CMD ["python", "main.py", "start"]
