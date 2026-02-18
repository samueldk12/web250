"""DeepFace-based model implementations."""

from typing import List, Optional
from deepface import DeepFace

from app.models.base import FaceRecognitionModel
from app.models.registry import ModelRegistry
from app.config import DEFAULT_DETECTOR


class DeepFaceModelBase(FaceRecognitionModel):
    """Base class for DeepFace-backed models."""

    def __init__(self):
        self._model_loaded = False
        self._detector = DEFAULT_DETECTOR

    def _ensure_model_loaded(self):
        """Lazy load the model on first use."""
        if not self._model_loaded:
            try:
                DeepFace.build_model(self._model_name)
                self._model_loaded = True
            except Exception:
                self._model_loaded = True

    @property
    def _model_name(self) -> str:
        """DeepFace model name."""
        return self.name

    def get_embedding(self, image_path: str) -> Optional[List[float]]:
        """Extract face embedding from image."""
        self._ensure_model_loaded()
        try:
            result = DeepFace.represent(
                img_path=image_path,
                model_name=self._model_name,
                detector_backend=self._detector,
                enforce_detection=True
            )
            if result and len(result) > 0:
                return result[0]["embedding"]
            return None
        except Exception as e:
            print(f"Error extracting embedding with {self.name}: {e}")
            return None

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
