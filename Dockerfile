FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies (needed for audio processing)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency definitions
COPY pyproject.toml uv.lock ./

RUN uv pip install --system livekit-agents livekit-plugins-google livekit-plugins-silero python-dotenv

# Copy the application code
COPY main.py server.py prompt.py ./

RUN uv run main.py download-files

CMD ["uv", "run", "main.py", "start"]
