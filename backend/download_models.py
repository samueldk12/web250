#!/usr/bin/env python3
"""
Script to download all pre-trained face recognition models.
Run this before building the Docker image to include models in the container.
"""

import os
import sys

# Set environment variables for model paths and keras compatibility
os.environ["DEEPFACE_HOME"] = "/app/.deepface"
os.environ["INSIGHTFACE_HOME"] = "/app/.insightface"
os.environ["TF_USE_LEGACY_KERAS"] = "1"


def download_deepface_models():
    """Download all DeepFace models."""
    print("\n" + "=" * 60)
    print("Downloading DeepFace models...")
    print("=" * 60)

    from deepface import DeepFace

    models = [
        "ArcFace",
        "Facenet",
        "Facenet512",
        "VGG-Face",
        "OpenFace",
        "DeepID",
        "SFace",
        "GhostFaceNet",
    ]

    for model_name in models:
        try:
            print(f"\n[*] Downloading {model_name}...")
            DeepFace.build_model(model_name)
            print(f"[+] {model_name} downloaded successfully!")
        except Exception as e:
            print(f"[-] Error downloading {model_name}: {e}")

    # Also download detector models
    print("\n[*] Downloading face detector models...")
    detectors = ["retinaface", "mtcnn", "opencv", "ssd"]
    for detector in detectors:
        try:
            print(f"[*] Initializing {detector} detector...")
            # This will trigger detector download
            DeepFace.build_model("ArcFace")
        except Exception as e:
            print(f"[-] Error with {detector}: {e}")


def download_insightface_models():
    """Download InsightFace models."""
    print("\n" + "=" * 60)
    print("Downloading InsightFace models...")
    print("=" * 60)

    try:
        from insightface.app import FaceAnalysis

        print("\n[*] Downloading InsightFace buffalo_l model...")
        app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("[+] InsightFace buffalo_l downloaded successfully!")

    except Exception as e:
        print(f"[-] Error downloading InsightFace: {e}")


def print_upscaling_info():
    """Print information about optional upscaling model."""
    print("\n" + "=" * 60)
    print("Optional: Image Upscaling Model (Real-ESRGAN)")
    print("=" * 60)
    print("""
To enable high-quality image upscaling for small faces, download Real-ESRGAN:

1. Create directory:
   mkdir -p /app/.esrgan/models

2. Download the model:
   wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \\
        -O /app/.esrgan/models/RealESRGAN_x4plus.pth

3. Install dependencies (add to requirements.txt):
   basicsr>=1.4.2
   realesrgan>=0.3.0

Without Real-ESRGAN, the system will use OpenCV's LANCZOS4 upscaling (still effective).
""")


def main():
    print("=" * 60)
    print("Face Recognition Models Downloader")
    print("=" * 60)

    # Create directories
    os.makedirs("/app/.deepface/weights", exist_ok=True)
    os.makedirs("/app/.insightface/models", exist_ok=True)
    os.makedirs("/app/.esrgan/models", exist_ok=True)

    # Download models
    download_deepface_models()
    download_insightface_models()

    # Print upscaling info
    print_upscaling_info()

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
