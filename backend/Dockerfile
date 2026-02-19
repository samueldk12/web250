FROM python:3.10-slim-bookworm

WORKDIR /app

# Set environment variables for model paths and keras compatibility
ENV DEEPFACE_HOME=/app/.deepface
ENV INSIGHTFACE_HOME=/app/.insightface
ENV ESRGAN_MODEL_PATH=/app/.esrgan/models/RealESRGAN_x4plus.pth
ENV TORCH_HOME=/app/.torch
ENV PYTHONUNBUFFERED=1
ENV TF_USE_LEGACY_KERAS=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Install system dependencies
# Note: build-essential + cmake are needed for dlib (face_recognition) compilation
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
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only BEFORE requirements.txt.
# Using the PyTorch index avoids pulling the large CUDA build from PyPI.
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model directories
RUN mkdir -p \
    /app/.deepface/weights \
    /app/.insightface/models \
    /app/.esrgan/models \
    /app/.torch/hub \
    /app/data/faces

# Copy download script and pre-download all models at build time
COPY download_models.py .
RUN python download_models.py

# Copy application code
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
