"""Image upscaling service for improving low-resolution face images."""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import tempfile

# Minimum face size for good recognition
MIN_FACE_SIZE = 112

# Real-ESRGAN model path
ESRGAN_MODEL_PATH = os.getenv("ESRGAN_MODEL_PATH", "/app/.esrgan/models/RealESRGAN_x4plus.pth")

# Try to import Real-ESRGAN
ESRGAN_AVAILABLE = False
RealESRGANer = None

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer as _RealESRGANer
    RealESRGANer = _RealESRGANer
    ESRGAN_AVAILABLE = True
except ImportError:
    print("Real-ESRGAN not available, using OpenCV upscaling fallback")


class ImageUpscaler:
    """Service for upscaling low-resolution images."""

    _instance = None
    _esrgan_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._initialized = False

    def _init_esrgan(self):
        """Initialize Real-ESRGAN model lazily."""
        if self._initialized:
            return

        if ESRGAN_AVAILABLE and os.path.exists(ESRGAN_MODEL_PATH):
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet

                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4
                )

                self._esrgan_model = RealESRGANer(
                    scale=4,
                    model_path=ESRGAN_MODEL_PATH,
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=False  # Use full precision for CPU
                )
                print("Real-ESRGAN initialized successfully")
            except Exception as e:
                print(f"Error initializing Real-ESRGAN: {e}")
                self._esrgan_model = None

        self._initialized = True

    def needs_upscaling(self, bbox: Tuple[int, int, int, int]) -> bool:
        """Check if a face bounding box is too small and needs upscaling."""
        x, y, w, h = bbox
        return w < MIN_FACE_SIZE or h < MIN_FACE_SIZE

    def upscale_image(self, image: np.ndarray, scale: int = 4) -> np.ndarray:
        """
        Upscale an image using Real-ESRGAN or OpenCV fallback.

        Args:
            image: Input image (BGR format)
            scale: Upscaling factor (default 4x)

        Returns:
            Upscaled image
        """
        self._init_esrgan()

        if self._esrgan_model is not None:
            try:
                output, _ = self._esrgan_model.enhance(image, outscale=scale)
                return output
            except Exception as e:
                print(f"Real-ESRGAN failed, falling back to OpenCV: {e}")

        # Fallback to OpenCV INTER_LANCZOS4 upscaling
        return self._opencv_upscale(image, scale)

    def _opencv_upscale(self, image: np.ndarray, scale: int) -> np.ndarray:
        """Upscale using OpenCV's high-quality interpolation."""
        h, w = image.shape[:2]
        new_size = (w * scale, h * scale)
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)

    def upscale_face_region(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: float = 0.3
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract and upscale a face region from an image.

        Args:
            image: Full image
            bbox: Face bounding box (x, y, w, h)
            padding: Padding around face as fraction of face size

        Returns:
            Tuple of (upscaled_region, new_bbox)
        """
        h, w = image.shape[:2]
        x, y, fw, fh = bbox

        # Add padding
        pad_x = int(fw * padding)
        pad_y = int(fh * padding)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + fw + pad_x)
        y2 = min(h, y + fh + pad_y)

        # Extract region
        region = image[y1:y2, x1:x2]

        # Calculate scale needed
        min_dim = min(fw, fh)
        scale = max(2, int(np.ceil(MIN_FACE_SIZE * 1.5 / min_dim)))
        scale = min(scale, 4)  # Cap at 4x

        # Upscale
        upscaled = self.upscale_image(region, scale)

        # Calculate new bbox in upscaled image
        new_x = int((x - x1) * scale)
        new_y = int((y - y1) * scale)
        new_w = int(fw * scale)
        new_h = int(fh * scale)

        return upscaled, (new_x, new_y, new_w, new_h)

    def upscale_file(self, image_path: str, scale: int = 4) -> str:
        """
        Upscale an image file and return path to upscaled version.

        Args:
            image_path: Path to input image
            scale: Upscaling factor

        Returns:
            Path to upscaled image (temporary file)
        """
        image = cv2.imread(image_path)
        if image is None:
            return image_path

        upscaled = self.upscale_image(image, scale)

        # Save to temp file
        suffix = Path(image_path).suffix or '.jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            cv2.imwrite(tmp.name, upscaled)
            return tmp.name


# Singleton instance
_upscaler: Optional[ImageUpscaler] = None


def get_upscaler() -> ImageUpscaler:
    """Get or create upscaler instance."""
    global _upscaler
    if _upscaler is None:
        _upscaler = ImageUpscaler()
    return _upscaler
