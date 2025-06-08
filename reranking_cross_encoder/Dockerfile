# Use a modern Python base image (ensures SQLite >= 3.35.0)
FROM python:3.10-slim

# Set environment variables for cleaner, reproducible builds
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# (Optional but recommended) Install system dependencies for scientific packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip first to latest
RUN pip install --upgrade pip

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Remove hash-checking mode from pip for torch (if present)
# Install torch first to isolate big download (if torch fails, you retry only torch, not all packages)
RUN pip install --no-cache-dir torch

# Then install all other requirements (except torch)
RUN pip install --no-cache-dir --no-deps -r requirements.txt

# Copy code
COPY . .

CMD ["python", "reranking.py"]
