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


# Default weights based on general model performance / competition results
MODEL_WEIGHTS = {
    # --- High accuracy ---
    "ArcFace": 1.0,
    "InsightFace": 1.0,        # buffalo_l  (ArcFace R100)
    "AntelopeV2": 1.0,         # Latest InsightFace model pack
    "Facenet512": 0.9,
    "GhostFaceNet": 0.9,
    # --- Medium accuracy ---
    "EdgeFaceS": 0.88,         # EFaR 2023 winner — S variant
    "BuffaloM": 0.87,          # InsightFace buffalo_m (R50)
    "EdgeFaceXS": 0.85,        # EFaR 2023 winner — XS (ultra-compact)
    "SFace": 0.85,
    "Facenet": 0.85,
    "Dlib": 0.80,              # dlib ResNet (same as face_recognition lib)
    "FaceRecognition": 0.80,   # face_recognition library (dlib ResNet)
    "VGG-Face": 0.80,
    # --- Lighter / legacy ---
    "DeepFaceMeta": 0.75,      # Historical Meta/Facebook DeepFace CNN
    "DeepID": 0.75,
    "OpenFace": 0.70,
}


class EnsembleService:
    """Service for ensemble face recognition."""

    def __init__(
        self,
        model_names: List[str],
        method: EnsembleMethod = EnsembleMethod.AVERAGE,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ensemble with specified models.

        Args:
            model_names: List of model names to use
            method: Ensemble method for combining predictions
            options: Additional options (e.g. min_votes, vote_threshold)
        """
        self.models: List[FaceRecognitionModel] = []
        self.method = method
        self.options = options or {}

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
            # Get voting options
            # Default threshold 0.5 (approx 50% match)
            # Default min_votes is majority (> 50%)
            vote_threshold = self.options.get("vote_threshold", 0.5)
            min_votes = self.options.get("min_votes", int(len(distances) / 2) + 1)
            
            # Count how many models agree it's a match (distance < threshold)
            votes = sum(1 for d in distances if d < vote_threshold)
            
            # If we have enough votes, return the average of the matching distances
            # This ensures the result is a "match" (low distance)
            if votes >= min_votes:
                matching = [d for d in distances if d < vote_threshold]
                return float(np.mean(matching))
            else:
                # If not enough votes, return a high distance (average of non-matching or just 1.0)
                # To be less harsh, we return the average of ALL distances, which will likely be high
                return float(np.mean(distances))

        return float(np.mean(distances))


def create_ensemble(
    model_names: List[str],
    method: str = "average",
    **kwargs
) -> EnsembleService:
    """
    Factory function to create an ensemble.

    Args:
        model_names: List of model names
        method: Ensemble method name
        **kwargs: Additional options (e.g. min_votes, vote_threshold)
    """
    try:
        ensemble_method = EnsembleMethod(method)
    except ValueError:
        ensemble_method = EnsembleMethod.AVERAGE

    return EnsembleService(model_names, ensemble_method, options=kwargs)
