"""Ensemble service for combining multiple face recognition models."""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from enum import Enum

from app.models.registry import ModelRegistry
from app.models.base import FaceRecognitionModel


class EnsembleMethod(str, Enum):
    """Methods for combining model predictions."""
    AVERAGE = "average"       # Average distances
    WEIGHTED = "weighted"     # Weighted average (by model accuracy)
    VOTING = "voting"         # Majority voting with threshold
    MIN = "min"               # Minimum distance (most confident match)
    MAX = "max"               # Maximum distance (most conservative)


# Default weights based on general model performance
MODEL_WEIGHTS = {
    "ArcFace": 1.0,
    "InsightFace": 1.0,
    "Facenet512": 0.9,
    "Facenet": 0.85,
    "VGG-Face": 0.8,
    "SFace": 0.85,
    "GhostFaceNet": 0.9,
    "OpenFace": 0.7,
    "DeepID": 0.75,
}


class EnsembleService:
    """Service for ensemble face recognition."""

    def __init__(self, model_names: List[str], method: EnsembleMethod = EnsembleMethod.AVERAGE):
        """
        Initialize ensemble with specified models.

        Args:
            model_names: List of model names to use
            method: Ensemble method for combining predictions
        """
        self.models: List[FaceRecognitionModel] = []
        self.method = method

        for name in model_names:
            model = ModelRegistry.get(name)
            if model:
                self.models.append(model)

        if not self.models:
            raise ValueError("No valid models specified for ensemble")

    def get_embeddings(self, image_path: str) -> Dict[str, Optional[List[float]]]:
        """
        Get embeddings from all models.

        Returns dict mapping model name to embedding (or None if failed).
        """
        embeddings = {}
        for model in self.models:
            try:
                embedding = model.get_embedding(image_path)
                embeddings[model.name] = embedding
            except Exception as e:
                print(f"Error getting embedding from {model.name}: {e}")
                embeddings[model.name] = None
        return embeddings

    def compare_embeddings(
        self,
        embeddings1: Dict[str, Optional[List[float]]],
        embeddings2: Dict[str, Optional[List[float]]]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compare embeddings using ensemble method.

        Returns:
            Tuple of (combined_distance, per_model_distances)
        """
        distances = {}
        valid_distances = []
        weights = []

        for model in self.models:
            name = model.name
            emb1 = embeddings1.get(name)
            emb2 = embeddings2.get(name)

            if emb1 is not None and emb2 is not None:
                distance = model.compare(emb1, emb2)
                distances[name] = distance
                valid_distances.append(distance)
                weights.append(MODEL_WEIGHTS.get(name, 1.0))

        if not valid_distances:
            return 1.0, distances

        # Combine distances based on method
        combined = self._combine_distances(valid_distances, weights)
        return combined, distances

    def _combine_distances(self, distances: List[float], weights: List[float]) -> float:
        """Combine distances using the specified method."""
        if not distances:
            return 1.0

        if self.method == EnsembleMethod.AVERAGE:
            return float(np.mean(distances))

        elif self.method == EnsembleMethod.WEIGHTED:
            weights = np.array(weights)
            distances = np.array(distances)
            return float(np.average(distances, weights=weights))

        elif self.method == EnsembleMethod.MIN:
            return float(np.min(distances))

        elif self.method == EnsembleMethod.MAX:
            return float(np.max(distances))

        elif self.method == EnsembleMethod.VOTING:
            # Count how many models agree it's a match (distance < 0.5)
            threshold = 0.5
            votes = sum(1 for d in distances if d < threshold)
            # If majority says match, use average of matching distances
            if votes > len(distances) / 2:
                matching = [d for d in distances if d < threshold]
                return float(np.mean(matching))
            else:
                return float(np.mean(distances))

        return float(np.mean(distances))


def create_ensemble(
    model_names: List[str],
    method: str = "average"
) -> EnsembleService:
    """
    Factory function to create an ensemble.

    Args:
        model_names: List of model names
        method: Ensemble method name
    """
    try:
        ensemble_method = EnsembleMethod(method)
    except ValueError:
        ensemble_method = EnsembleMethod.AVERAGE

    return EnsembleService(model_names, ensemble_method)
