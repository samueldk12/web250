"""InsightFace-based model implementations."""

from typing import List, Optional
import numpy as np
import cv2

from app.models.base import FaceRecognitionModel, FaceData
from app.models.registry import ModelRegistry

# InsightFace imports - optional
INSIGHTFACE_AVAILABLE = False
FaceAnalysis = None

try:
    from insightface.app import FaceAnalysis as _FaceAnalysis
    FaceAnalysis = _FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError as e:
    print(f"InsightFace not available: {e}")
except AttributeError as e:
    print(f"InsightFace compatibility issue: {e}")
except Exception as e:
    print(f"Error loading InsightFace: {e}")


class InsightFaceModelBase(FaceRecognitionModel):
    """Base class for InsightFace-backed models."""

    _app = None  # Shared FaceAnalysis instance

    def __init__(self):
        self._initialized = False

    @classmethod
    def _get_app(cls):
        """Get or create shared FaceAnalysis instance."""
        if cls._app is None and INSIGHTFACE_AVAILABLE and FaceAnalysis is not None:
            try:
                cls._app = FaceAnalysis(
                    name='buffalo_l',
                    providers=['CPUExecutionProvider']
                )
                cls._app.prepare(ctx_id=-1, det_size=(640, 640))
            except Exception as e:
                print(f"Error initializing InsightFace: {e}")
                cls._app = None
        return cls._app

    def get_embedding(self, image_path: str) -> Optional[List[float]]:
        """Extract face embedding from image using InsightFace."""
        if not INSIGHTFACE_AVAILABLE:
            print("InsightFace not available")
            return None

        app = self._get_app()
        if app is None:
            return None

        try:
            img = cv2.imread(image_path)
            if img is None:
                return None

            faces = app.get(img)
            if faces and len(faces) > 0:
                # Get the largest face
                largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                embedding = largest_face.embedding
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
                return embedding.tolist()
            return None
        except Exception as e:
            print(f"Error extracting embedding with InsightFace: {e}")
            return None

    def get_all_embeddings(self, image_path: str) -> List[FaceData]:
        """Extract embeddings for ALL faces in image using InsightFace."""
        if not INSIGHTFACE_AVAILABLE:
            return []

        app = self._get_app()
        if app is None:
            return []

        try:
            img = cv2.imread(image_path)
            if img is None:
                return []

            faces = app.get(img)
            result = []
            for i, face in enumerate(faces):
                embedding = face.embedding
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
                bbox = (
                    int(face.bbox[0]),
                    int(face.bbox[1]),
                    int(face.bbox[2] - face.bbox[0]),
                    int(face.bbox[3] - face.bbox[1])
                )
                result.append(FaceData(
                    embedding=embedding.tolist(),
                    bbox=bbox,
                    confidence=float(face.det_score) if hasattr(face, 'det_score') else 1.0,
                    face_index=i
                ))
            return result
        except Exception as e:
            print(f"Error extracting embeddings with InsightFace: {e}")
            return []

    def compare(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compare two embeddings using cosine distance."""
        return self.cosine_distance(embedding1, embedding2)


# Only register InsightFace model if the library is available
if INSIGHTFACE_AVAILABLE:
    @ModelRegistry.register
    class InsightFaceModel(InsightFaceModelBase):
        """InsightFace Buffalo_L model - High accuracy face recognition."""

        @property
        def name(self) -> str:
            return "InsightFace"

        @property
        def display_name(self) -> str:
            return "InsightFace (Buffalo-L)"

        @property
        def description(self) -> str:
            return "State-of-the-art model from InsightFace project. High accuracy with fast inference."

        @property
        def embedding_size(self) -> int:
            return 512
else:
    print("InsightFace model not registered due to import issues")
