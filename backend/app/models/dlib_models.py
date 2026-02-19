"""Standalone face_recognition (dlib) model implementation.

Uses the `face_recognition` library by Adam Geitgey, which wraps dlib's
ResNet-based face encoder. This produces the same 128-d embedding as
DeepFace's "Dlib" backend but via a completely independent pipeline with
a simpler HOG/CNN detector.

Repo: https://github.com/ageitgey/face_recognition
"""

from typing import List, Optional

from app.models.base import FaceRecognitionModel, FaceData
from app.models.registry import ModelRegistry

FACE_RECOGNITION_AVAILABLE = False
_fr = None

try:
    import face_recognition as _face_recognition
    _fr = _face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    print("face_recognition library not available. FaceRecognition model will not be registered.")
except Exception as e:
    print(f"Error loading face_recognition: {e}")


if FACE_RECOGNITION_AVAILABLE:

    @ModelRegistry.register
    class FaceRecognitionLibModel(FaceRecognitionModel):
        """
        face_recognition library (dlib ResNet) — very simple API, great for prototyping.

        Detection: HOG (fast, CPU-friendly)
        Recognition: dlib's ResNet, 128-d embedding
        Same underlying model weights as DeepFace's "Dlib" backend.
        """

        @property
        def name(self) -> str:
            return "FaceRecognition"

        @property
        def display_name(self) -> str:
            return "face_recognition (dlib)"

        @property
        def description(self) -> str:
            return (
                "Standalone face_recognition library by Adam Geitgey. "
                "Wraps dlib's ResNet with an extremely simple API. "
                "Great for rapid prototyping."
            )

        @property
        def embedding_size(self) -> int:
            return 128

        def get_embedding(self, image_path: str) -> Optional[List[float]]:
            try:
                image = _fr.load_image_file(image_path)
                # Use HOG detector — fast on CPU
                locations = _fr.face_locations(image, model="hog")
                encodings = _fr.face_encodings(image, locations)
                if not encodings:
                    return None
                # Return the encoding of the largest face
                if len(encodings) == 1:
                    return encodings[0].tolist()
                # Pick largest face by bounding-box area
                largest_idx = max(
                    range(len(locations)),
                    key=lambda i: (
                        (locations[i][2] - locations[i][0]) *
                        (locations[i][1] - locations[i][3])
                    )
                )
                return encodings[largest_idx].tolist()
            except Exception as e:
                print(f"FaceRecognition get_embedding error: {e}")
                return None

        def get_all_embeddings(self, image_path: str) -> List[FaceData]:
            try:
                image = _fr.load_image_file(image_path)
                locations = _fr.face_locations(image, model="hog")
                encodings = _fr.face_encodings(image, locations)

                result = []
                for i, (encoding, loc) in enumerate(zip(encodings, locations)):
                    # face_recognition returns (top, right, bottom, left)
                    top, right, bottom, left = loc
                    bbox = (left, top, right - left, bottom - top)
                    result.append(FaceData(
                        embedding=encoding.tolist(),
                        bbox=bbox,
                        confidence=1.0,
                        face_index=i
                    ))
                return result
            except Exception as e:
                print(f"FaceRecognition get_all_embeddings error: {e}")
                return []

        def compare(self, embedding1: List[float], embedding2: List[float]) -> float:
            # face_recognition normally uses Euclidean distance;
            # we use cosine here to stay consistent with the rest of the system.
            return self.cosine_distance(embedding1, embedding2)

else:
    print("FaceRecognitionLibModel not registered (face_recognition library unavailable).")
