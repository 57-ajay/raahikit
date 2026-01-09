FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml uv.lock ./

RUN uv pip install --system \
    'livekit-agents[silero,turn-detector]' \
    livekit-plugins-google \
    livekit-plugins-silero \
    livekit-plugins-noise-cancellation \
    python-dotenv \
    google-genai \
    redisvl \
    'fastapi[all]'

COPY main.py server.py prompt.py ./

RUN python main.py download-files

CMD ["python", "main.py", "start"]
