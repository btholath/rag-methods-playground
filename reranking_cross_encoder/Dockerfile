# Use a modern Python base image (3.10+ includes SQLite >= 3.35.0)
FROM python:3.10-slim

# Set environment variables for safer Python and consistent output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (if you need to build packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy requirements if available, then install them
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy your code into the container
COPY . .

# Set entrypoint for running your script (can override in Codespace terminal)
CMD ["python", "reranking.py"]
