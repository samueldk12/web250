from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import numpy as np


class FaceData:
    """Data class for a detected face."""
    def __init__(
        self,
        embedding: List[float],
        bbox: Optional[Tuple[int, int, int, int]] = None,
        confidence: float = 1.0,
        face_index: int = 0
    ):
        self.embedding = embedding
        self.bbox = bbox  # (x, y, w, h)
        self.confidence = confidence
        self.face_index = face_index

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding": self.embedding,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "face_index": self.face_index
        }


class FaceRecognitionModel(ABC):
    """Base interface for face recognition models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Display name for UI."""
        pass

    @property
    def description(self) -> str:
        """Model description."""
        return ""

    @property
    def embedding_size(self) -> int:
        """Size of embedding vector."""
        return 512

    @abstractmethod
    def get_embedding(self, image_path: str) -> Optional[List[float]]:
        """Extract face embedding from image (largest face only)."""
        pass

    def get_all_embeddings(self, image_path: str) -> List[FaceData]:
        """Extract embeddings for ALL faces in image. Override in subclass for better performance."""
        # Default implementation: just return single face
        embedding = self.get_embedding(image_path)
        if embedding:
            return [FaceData(embedding=embedding, face_index=0)]
        return []

    @abstractmethod
    def compare(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compare two embeddings and return distance (0 = identical, 1 = different)."""
        pass

    def cosine_distance(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine distance between two vectors."""
        a = np.array(a)
        b = np.array(b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1 - np.dot(a, b) / (norm_a * norm_b)

    def euclidean_distance(self, a: List[float], b: List[float]) -> float:
        """Calculate normalized euclidean distance."""
        a = np.array(a)
        b = np.array(b)
        return float(np.linalg.norm(a - b))

    def is_available(self) -> bool:
        """Check if the model is available and can be loaded."""
        return True  # Override in subclass if needed

    def to_dict(self) -> Dict[str, Any]:
        """Convert model info to dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "embedding_size": self.embedding_size,
            "available": self.is_available()
        }
