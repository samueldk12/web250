import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable

from app.config import FACES_PATH, EMBEDDINGS_FILE


class FaceDatabase:
    """JSON-based face database with multi-model embedding support."""

    def __init__(self):
        self._load()

    def _load(self):
        """Load embeddings from file."""
        if EMBEDDINGS_FILE.exists():
            with open(EMBEDDINGS_FILE, "r") as f:
                data = json.load(f)
                self.faces = data.get("faces", [])
        else:
            self.faces = []

    def _save(self):
        """Save embeddings to file."""
        with open(EMBEDDINGS_FILE, "w") as f:
            json.dump({"faces": self.faces}, f, indent=2)

    def add_face(
        self,
        name: str,
        embeddings: Dict[str, List[float]],
        image_path: str
    ) -> Dict[str, Any]:
        """
        Add a new face to the database.

        Args:
            name: Person's name
            embeddings: Dict mapping model name to embedding vector
            image_path: Path to the uploaded image
        """
        face_id = str(uuid.uuid4())

        # Copy image to faces directory
        src_path = Path(image_path)
        dest_filename = f"{face_id}{src_path.suffix}"
        dest_path = FACES_PATH / dest_filename
        shutil.copy2(src_path, dest_path)

        face = {
            "id": face_id,
            "name": name,
            "embeddings": embeddings,  # Multi-model embeddings
            "image": dest_filename,
            "created_at": datetime.utcnow().isoformat()
        }
        self.faces.append(face)
        self._save()
        return face

    def update_embeddings(
        self,
        face_id: str,
        embeddings: Dict[str, List[float]]
    ) -> bool:
        """Update embeddings for a face (add new model embeddings)."""
        for face in self.faces:
            if face["id"] == face_id:
                # Merge new embeddings with existing
                if "embeddings" not in face:
                    face["embeddings"] = {}
                face["embeddings"].update(embeddings)
                self._save()
                return True
        return False

    def get_face(self, face_id: str) -> Optional[Dict[str, Any]]:
        """Get face by ID."""
        for face in self.faces:
            if face["id"] == face_id:
                return face
        return None

    def get_all_faces(self) -> List[Dict[str, Any]]:
        """Get all faces (without embeddings for API response)."""
        result = []
        for face in self.faces:
            face_info = {
                "id": face["id"],
                "name": face["name"],
                "image": face["image"],
                "created_at": face["created_at"],
                "models": list(face.get("embeddings", {}).keys())
            }
            result.append(face_info)
        return result

    def delete_face(self, face_id: str) -> bool:
        """Delete a face from the database."""
        for i, face in enumerate(self.faces):
            if face["id"] == face_id:
                # Delete image file
                image_path = FACES_PATH / face["image"]
                if image_path.exists():
                    image_path.unlink()
                # Remove from list
                self.faces.pop(i)
                self._save()
                return True
        return False

    def find_matches(
        self,
        embeddings: Dict[str, List[float]],
        compare_fn: Callable,
        threshold: float = 0.4,
        model_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find matching faces by embedding comparison.

        Args:
            embeddings: Dict mapping model name to query embedding
            compare_fn: Function to compare embeddings (returns combined distance)
            threshold: Distance threshold for match
            model_names: List of model names to use (None = use all available)
        """
        matches = []

        for face in self.faces:
            face_embeddings = face.get("embeddings", {})

            # Filter to requested models
            if model_names:
                query_embs = {k: v for k, v in embeddings.items() if k in model_names}
                face_embs = {k: v for k, v in face_embeddings.items() if k in model_names}
            else:
                query_embs = embeddings
                face_embs = face_embeddings

            # Need at least one common model
            common_models = set(query_embs.keys()) & set(face_embs.keys())
            if not common_models:
                continue

            # Compare using the provided function
            distance, model_distances = compare_fn(query_embs, face_embs)

            if distance < threshold:
                matches.append({
                    "id": face["id"],
                    "name": face["name"],
                    "image": face["image"],
                    "distance": distance,
                    "confidence": max(0, 1 - distance),
                    "model_distances": model_distances
                })

        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x["confidence"], reverse=True)
        return matches

    def find_matches_single_model(
        self,
        embedding: List[float],
        model_name: str,
        compare_fn: Callable,
        threshold: float = 0.4
    ) -> List[Dict[str, Any]]:
        """Find matches using a single model."""
        matches = []

        for face in self.faces:
            face_embeddings = face.get("embeddings", {})
            face_embedding = face_embeddings.get(model_name)

            if face_embedding is None:
                continue

            distance = compare_fn(embedding, face_embedding)

            if distance < threshold:
                matches.append({
                    "id": face["id"],
                    "name": face["name"],
                    "image": face["image"],
                    "distance": distance,
                    "confidence": max(0, 1 - distance),
                    "model": model_name
                })

        matches.sort(key=lambda x: x["confidence"], reverse=True)
        return matches


# Singleton instance
_db: Optional[FaceDatabase] = None


def get_face_db() -> FaceDatabase:
    """Get or create database instance."""
    global _db
    if _db is None:
        _db = FaceDatabase()
    return _db
