#!/usr/bin/env python3
"""
Script to download all pre-trained face recognition models locally.
Run this to test models without Docker.

Usage:
    pip install -r requirements.txt
    python download_models_local.py
"""

import os
import sys
from pathlib import Path

# Create local model directories
LOCAL_MODEL_DIR = Path(__file__).parent / "models_cache"
DEEPFACE_DIR = LOCAL_MODEL_DIR / ".deepface"
INSIGHTFACE_DIR = LOCAL_MODEL_DIR / ".insightface"

# Set environment variables
os.environ["DEEPFACE_HOME"] = str(DEEPFACE_DIR)
os.environ["INSIGHTFACE_HOME"] = str(INSIGHTFACE_DIR)


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


def main():
    print("=" * 60)
    print("Face Recognition Models Downloader (Local)")
    print("=" * 60)
    print(f"\nModels will be saved to: {LOCAL_MODEL_DIR}")

    # Create directories
    DEEPFACE_DIR.mkdir(parents=True, exist_ok=True)
    INSIGHTFACE_DIR.mkdir(parents=True, exist_ok=True)

    # Download models
    download_deepface_models()
    download_insightface_models()

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Models saved to: {LOCAL_MODEL_DIR}")
    print("=" * 60)

    # Print instructions
    print("\nTo use these models, set environment variables:")
    print(f'  export DEEPFACE_HOME="{DEEPFACE_DIR}"')
    print(f'  export INSIGHTFACE_HOME="{INSIGHTFACE_DIR}"')


if __name__ == "__main__":
    main()
