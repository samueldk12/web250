import cv2
import numpy as np
import os
from typing import Tuple, List, Optional
import tempfile
from pathlib import Path
from app.services.upscaler import get_upscaler

class ImagePreprocessor:
    """Service to handle various image preprocessing techniques."""
    
    @staticmethod
    def apply_clahe(image_path: str) -> Optional[str]:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.
        Good for lighting variations.
        """
        try:
            img = cv2.imread(image_path)
            if img is None: 
                return None
                
            # Convert to LAB
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L-channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # Merge and convert back
            limg = cv2.merge((cl, a, b))
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            return ImagePreprocessor._save_temp(final, image_path, "clahe")
        except Exception as e:
            print(f"CLAHE failed: {e}")
            return None

    @staticmethod
    def apply_sharpening(image_path: str) -> Optional[str]:
        """Apply unsharp mask sharpening."""
        try:
            img = cv2.imread(image_path)
            if img is None: 
                return None
                
            gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
            final = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0, img)
            
            return ImagePreprocessor._save_temp(final, image_path, "sharp")
        except Exception as e:
            print(f"Sharpening failed: {e}")
            return None

    @staticmethod
    def apply_upscaling(image_path: str) -> Optional[str]:
        """Use the centralized upscaler service."""
        try:
            upscaler = get_upscaler()
            # Force upscale even if not strictly needed by size
            # We use a temp file wrapper here
            img = cv2.imread(image_path)
            if img is None: 
                return None
            
            # Check size, if already huge, don't upscale too much
            h, w = img.shape[:2]
            if h * w > 2000 * 2000:
                return None # Too big already
                
            scale = 2 # Moderate upscale for general recognition
            if h < 500 or w < 500:
                scale = 4
                
            processed = upscaler.upscale_image(img, scale=scale)
            return ImagePreprocessor._save_temp(processed, image_path, "upscale")
        except Exception as e:
            print(f"Upscaling failed: {e}")
            return None
            
    @staticmethod
    def apply_brightness_normalization(image_path: str) -> Optional[str]:
        """Normalize brightness/contrast automatically."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            # Automatic brightness and contrast optimization
            # Clip histogram to remove outliers
            # Convert to YUV
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            # Equalize Y channel
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            final = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            
            return ImagePreprocessor._save_temp(final, image_path, "norm")
        except Exception as e:
            print(f"Normalization failed: {e}")
            return None

    @staticmethod
    def _save_temp(image: np.ndarray, original_path: str, suffix: str) -> str:
        """Helper to save temp file."""
        orig_suffix = Path(original_path).suffix or ".jpg"
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{suffix}{orig_suffix}")
        cv2.imwrite(temp.name, image)
        return temp.name

    @staticmethod
    def cleanup(paths: List[Optional[str]]):
        """Cleanup temp files."""
        for p in paths:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except:
                    pass
