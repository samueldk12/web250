FROM python:3.10-slim-bookworm

WORKDIR /app

# Set environment variables for model paths and keras compatibility
ENV DEEPFACE_HOME=/app/.deepface
ENV INSIGHTFACE_HOME=/app/.insightface
ENV PYTHONUNBUFFERED=1
ENV TF_USE_LEGACY_KERAS=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create model directories
RUN mkdir -p /app/.deepface/weights /app/.insightface/models /app/data/faces

# Copy download script and run it to download models
COPY download_models.py .
RUN python download_models.py

# Copy application code
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
