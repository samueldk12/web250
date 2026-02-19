"""DeepFace-based model implementations."""

import os
import cv2
import tempfile
from typing import List, Optional
from deepface import DeepFace

from app.models.base import FaceRecognitionModel, FaceData
from app.models.registry import ModelRegistry
from app.config import DEFAULT_DETECTOR
from app.services.upscaler import get_upscaler, MIN_FACE_SIZE


class DeepFaceModelBase(FaceRecognitionModel):
    """Base class for DeepFace-backed models."""

    def __init__(self):
        self._model_loaded = False
        self._model_available = None  # None = not checked, True/False = checked
        self._detector = DEFAULT_DETECTOR

    def _ensure_model_loaded(self):
        """Lazy load the model on first use."""
        if not self._model_loaded:
            try:
                DeepFace.build_model(self._model_name)
                self._model_loaded = True
                self._model_available = True
            except Exception as e:
                print(f"Error loading model {self._model_name}: {e}")
                self._model_loaded = True
                self._model_available = False

    def is_available(self) -> bool:
        """Check if the model is available."""
        if self._model_available is None:
            self._ensure_model_loaded()
        return self._model_available if self._model_available is not None else True

    @property
    def _model_name(self) -> str:
        """DeepFace model name."""
        return self.name

    def get_embedding(self, image_path: str) -> Optional[List[float]]:
        """Extract face embedding from image (largest face only)."""
        self._ensure_model_loaded()
        try:
            result = DeepFace.represent(
                img_path=image_path,
                model_name=self._model_name,
                detector_backend=self._detector,
                enforce_detection=True
            )
            if result and len(result) > 0:
                # Return largest face
                if len(result) == 1:
                    return result[0]["embedding"]
                # Find largest face by area
                largest = max(result, key=lambda x: x.get("facial_area", {}).get("w", 0) * x.get("facial_area", {}).get("h", 0))
                return largest["embedding"]
            return None
        except Exception as e:
            print(f"Error extracting embedding with {self.name}: {e}")
            return None

    def get_all_embeddings(self, image_path: str) -> List[FaceData]:
        """Extract embeddings for ALL faces in image, with upscaling for small faces."""
        self._ensure_model_loaded()
        try:
            result = DeepFace.represent(
                img_path=image_path,
                model_name=self._model_name,
                detector_backend=self._detector,
                enforce_detection=True
            )

            faces = []
            upscaler = get_upscaler()
            image = None  # Lazy load only if needed

            for i, face_result in enumerate(result):
                facial_area = face_result.get("facial_area", {})
                bbox = (
                    facial_area.get("x", 0),
                    facial_area.get("y", 0),
                    facial_area.get("w", 0),
                    facial_area.get("h", 0)
                )
                confidence = face_result.get("face_confidence", 1.0)
                embedding = face_result["embedding"]

                # Check if face is too small and might benefit from upscaling
                if upscaler.needs_upscaling(bbox):
                    try:
                        # Load image if not already loaded
                        if image is None:
                            image = cv2.imread(image_path)

                        if image is not None:
                            # Upscale the face region
                            upscaled_region, new_bbox = upscaler.upscale_face_region(image, bbox)

                            # Save to temp file and re-extract embedding
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                                cv2.imwrite(tmp.name, upscaled_region)
                                try:
                                    upscaled_result = DeepFace.represent(
                                        img_path=tmp.name,
                                        model_name=self._model_name,
                                        detector_backend=self._detector,
                                        enforce_detection=True
                                    )
                                    if upscaled_result and len(upscaled_result) > 0:
                                        # Use the upscaled embedding (better quality)
                                        embedding = upscaled_result[0]["embedding"]
                                        print(f"Face {i}: Upscaled from {bbox[2]}x{bbox[3]} for better embedding")
                                except Exception as e:
                                    print(f"Upscaling extraction failed for face {i}: {e}")
                                finally:
                                    os.unlink(tmp.name)
                    except Exception as e:
                        print(f"Error during upscaling for face {i}: {e}")

                faces.append(FaceData(
                    embedding=embedding,
                    bbox=bbox,
                    confidence=confidence,
                    face_index=i
                ))

            return faces
        except Exception as e:
            print(f"Error extracting embeddings with {self.name}: {e}")
            return []

    def compare(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compare two embeddings using cosine distance."""
        return self.cosine_distance(embedding1, embedding2)


@ModelRegistry.register
class ArcFaceModel(DeepFaceModelBase):
    """ArcFace model - State-of-the-art face recognition."""

    @property
    def name(self) -> str:
        return "ArcFace"

    @property
    def display_name(self) -> str:
        return "ArcFace"

    @property
    def description(self) -> str:
        return "High accuracy model using Additive Angular Margin Loss. Best for general face recognition."

    @property
    def embedding_size(self) -> int:
        return 512


@ModelRegistry.register
class FacenetModel(DeepFaceModelBase):
    """Facenet model - Google's face recognition model."""

    @property
    def name(self) -> str:
        return "Facenet"

    @property
    def display_name(self) -> str:
        return "FaceNet"

    @property
    def description(self) -> str:
        return "Google's deep learning model using triplet loss. Good balance of speed and accuracy."

    @property
    def embedding_size(self) -> int:
        return 128


@ModelRegistry.register
class Facenet512Model(DeepFaceModelBase):
    """Facenet512 model - Larger embedding version."""

    @property
    def name(self) -> str:
        return "Facenet512"

    @property
    def _model_name(self) -> str:
        return "Facenet512"

    @property
    def display_name(self) -> str:
        return "FaceNet-512"

    @property
    def description(self) -> str:
        return "FaceNet with 512-dimensional embeddings for higher accuracy."

    @property
    def embedding_size(self) -> int:
        return 512


@ModelRegistry.register
class VGGFaceModel(DeepFaceModelBase):
    """VGG-Face model - Oxford's face recognition model."""

    @property
    def name(self) -> str:
        return "VGG-Face"

    @property
    def display_name(self) -> str:
        return "VGG-Face"

    @property
    def description(self) -> str:
        return "Deep face recognition model based on VGGNet architecture."

    @property
    def embedding_size(self) -> int:
        return 4096


@ModelRegistry.register
class OpenFaceModel(DeepFaceModelBase):
    """OpenFace model - Lightweight face recognition."""

    @property
    def name(self) -> str:
        return "OpenFace"

    @property
    def display_name(self) -> str:
        return "OpenFace"

    @property
    def description(self) -> str:
        return "Lightweight model optimized for real-time applications."

    @property
    def embedding_size(self) -> int:
        return 128


@ModelRegistry.register
class DeepIDModel(DeepFaceModelBase):
    """DeepID model."""

    @property
    def name(self) -> str:
        return "DeepID"

    @property
    def display_name(self) -> str:
        return "DeepID"

    @property
    def description(self) -> str:
        return "Deep learning model for face identification."

    @property
    def embedding_size(self) -> int:
        return 160


@ModelRegistry.register
class SFaceModel(DeepFaceModelBase):
    """SFace model - Sigmoid-Constrained Hypersphere Loss."""

    @property
    def name(self) -> str:
        return "SFace"

    @property
    def display_name(self) -> str:
        return "SFace"

    @property
    def description(self) -> str:
        return "Uses sigmoid-constrained hypersphere loss for better discrimination."

    @property
    def embedding_size(self) -> int:
        return 128


@ModelRegistry.register
class GhostFaceNetModel(DeepFaceModelBase):
    """GhostFaceNet model - Efficient face recognition."""

    @property
    def name(self) -> str:
        return "GhostFaceNet"

    @property
    def display_name(self) -> str:
        return "GhostFaceNet"

    @property
    def description(self) -> str:
        return "Efficient model using ghost modules for faster inference."

    @property
    def embedding_size(self) -> int:
        return 512


@ModelRegistry.register
class DeepFaceMetaModel(DeepFaceModelBase):
    """DeepFace (Meta/Facebook) - Historical deep learning baseline."""

    @property
    def name(self) -> str:
        return "DeepFaceMeta"

    @property
    def _model_name(self) -> str:
        return "DeepFace"

    @property
    def display_name(self) -> str:
        return "DeepFace (Meta)"

    @property
    def description(self) -> str:
        return "Meta/Facebook's original DeepFace CNN. Historical baseline that popularized deep face recognition."

    @property
    def embedding_size(self) -> int:
        return 4096


@ModelRegistry.register
class DlibModel(DeepFaceModelBase):
    """Dlib ResNet model - Same backbone used by face_recognition library."""

    @property
    def name(self) -> str:
        return "Dlib"

    @property
    def display_name(self) -> str:
        return "Dlib ResNet"

    @property
    def description(self) -> str:
        return "dlib's ResNet-based model (same as face_recognition library). Fast and widely used."

    @property
    def embedding_size(self) -> int:
        return 128
