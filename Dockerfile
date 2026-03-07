FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (libsndfile for audio file handling, libsox for librosa)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --no-cache-dir \
    requests \
    pyyaml \
    numpy \
    pandas \
    openai \
    python-dotenv \
    tqdm \
    soundfile \
    websockets \
    librosa

# Copy core modules (will be overridden by volume mount in dev)
COPY core/ /app/core/

# Default command (will be overridden in docker-compose)
CMD ["python", "/app/core/bench_runner.py"]
