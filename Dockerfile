FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    requests \
    pyyaml \
    numpy \
    pandas \
    openai \
    python-dotenv \
    tqdm

# Copy core modules (will be overridden by volume mount in dev)
COPY core/ /app/core/

# Default command (will be overridden in docker-compose)
CMD ["python", "/app/core/bench_runner.py"]
