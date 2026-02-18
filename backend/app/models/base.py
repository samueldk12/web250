from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np


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
        """Extract face embedding from image."""
        pass

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert model info to dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "embedding_size": self.embedding_size
        }
