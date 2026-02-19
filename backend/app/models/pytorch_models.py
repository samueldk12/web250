"""PyTorch-based model implementations.

Includes EdgeFace models (winner of EFaR 2023 Efficient Face Recognition competition).
Loaded via torch.hub from: https://github.com/otroshi/edgeface

Requires: torch, torchvision (CPU-only build recommended)
Face detection/alignment uses InsightFace (already a project dependency).
"""

from typing import List, Optional
import numpy as np
import cv2

from app.models.base import FaceRecognitionModel, FaceData
from app.models.registry import ModelRegistry

# --- Optional dependency checks ---

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available. EdgeFace models will not be registered.")

INSIGHTFACE_ALIGN_AVAILABLE = False
_face_align_module = None
_FaceAnalysis_cls = None
try:
    from insightface.utils import face_align as _fa
    from insightface.app import FaceAnalysis as _FA
    _face_align_module = _fa
    _FaceAnalysis_cls = _FA
    INSIGHTFACE_ALIGN_AVAILABLE = True
except Exception as _e:
    print(f"InsightFace alignment unavailable for EdgeFace: {_e}")


class EdgeFaceModelBase(FaceRecognitionModel):
    """
    Base class for EdgeFace models loaded via torch.hub.

    Detection + 5-point alignment is done with InsightFace (buffalo_l).
    The 112x112 aligned face crop is then fed into the EdgeFace PyTorch model.
    """

    _detector = None          # Shared InsightFace detector (class-level)
    _torch_models: dict = {}  # Cache: variant_name -> torch model

    # --- Detection ---

    @classmethod
    def _get_detector(cls):
        if cls._detector is None and INSIGHTFACE_ALIGN_AVAILABLE:
            try:
                cls._detector = _FaceAnalysis_cls(
                    name='buffalo_l',
                    providers=['CPUExecutionProvider']
                )
                cls._detector.prepare(ctx_id=-1, det_size=(640, 640))
            except Exception as e:
                print(f"EdgeFace: detector init error: {e}")
                cls._detector = None
        return cls._detector

    # --- Model loading ---

    @property
    def _edgeface_variant(self) -> str:
        """torch.hub model name — override in subclasses."""
        return "edgeface_xs_gamma_06"

    def _get_torch_model(self):
        variant = self._edgeface_variant
        if variant not in self._torch_models:
            try:
                print(f"Loading EdgeFace {variant} via torch.hub …")
                model = torch.hub.load(
                    'otroshi/edgeface',
                    variant,
                    source='github',
                    pretrained=True,
                    trust_repo=True
                )
                model.eval()
                self._torch_models[variant] = model
                print(f"EdgeFace {variant} loaded successfully.")
            except Exception as e:
                print(f"EdgeFace {variant} load error: {e}")
                self._torch_models[variant] = None
        return self._torch_models[variant]

    # --- Preprocessing ---

    def _preprocess_face(self, face_bgr: np.ndarray) -> "torch.Tensor":
        """Convert a BGR 112×112 aligned crop to a normalised float tensor."""
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_f = face_rgb.astype(np.float32) / 255.0
        face_f = (face_f - 0.5) / 0.5
        face_chw = np.transpose(face_f, (2, 0, 1))      # HWC → CHW
        return torch.FloatTensor(face_chw).unsqueeze(0)  # add batch dim

    # --- FaceRecognitionModel interface ---

    def is_available(self) -> bool:
        if not TORCH_AVAILABLE or not INSIGHTFACE_ALIGN_AVAILABLE:
            return False
        return self._get_torch_model() is not None

    def get_embedding(self, image_path: str) -> Optional[List[float]]:
        model = self._get_torch_model()
        detector = self._get_detector()
        if model is None or detector is None:
            return None

        img = cv2.imread(image_path)
        if img is None:
            return None

        faces = detector.get(img)
        if not faces:
            return None

        # Use the largest detected face
        largest = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        try:
            aligned = _face_align_module.norm_crop(img, landmark=largest.kps)
        except Exception as e:
            print(f"EdgeFace alignment error: {e}")
            return None

        tensor = self._preprocess_face(aligned)
        with torch.no_grad():
            emb = model(tensor).squeeze().numpy()

        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.tolist()

    def get_all_embeddings(self, image_path: str) -> List[FaceData]:
        model = self._get_torch_model()
        detector = self._get_detector()
        if model is None or detector is None:
            return []

        img = cv2.imread(image_path)
        if img is None:
            return []

        faces = detector.get(img)
        result = []

        for i, face in enumerate(faces):
            try:
                aligned = _face_align_module.norm_crop(img, landmark=face.kps)
            except Exception as e:
                print(f"EdgeFace alignment error (face {i}): {e}")
                continue

            tensor = self._preprocess_face(aligned)
            with torch.no_grad():
                emb = model(tensor).squeeze().numpy()

            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            bbox = (
                int(face.bbox[0]),
                int(face.bbox[1]),
                int(face.bbox[2] - face.bbox[0]),
                int(face.bbox[3] - face.bbox[1]),
            )
            result.append(FaceData(
                embedding=emb.tolist(),
                bbox=bbox,
                confidence=float(face.det_score) if hasattr(face, 'det_score') else 1.0,
                face_index=i
            ))

        return result

    def compare(self, embedding1: List[float], embedding2: List[float]) -> float:
        return self.cosine_distance(embedding1, embedding2)


# Register EdgeFace models only when their dependencies are present
if TORCH_AVAILABLE and INSIGHTFACE_ALIGN_AVAILABLE:

    @ModelRegistry.register
    class EdgeFaceXSModel(EdgeFaceModelBase):
        """EdgeFace XS — compact winner of EFaR 2023 efficiency track (IJCB 2023)."""

        @property
        def _edgeface_variant(self) -> str:
            return "edgeface_xs_gamma_06"

        @property
        def name(self) -> str:
            return "EdgeFaceXS"

        @property
        def display_name(self) -> str:
            return "EdgeFace XS (EFaR'23)"

        @property
        def description(self) -> str:
            return (
                "Winner of EFaR 2023 efficiency challenge (IJCB 2023). "
                "Ultra-lightweight model designed for edge devices."
            )

        @property
        def embedding_size(self) -> int:
            return 512

    @ModelRegistry.register
    class EdgeFaceSModel(EdgeFaceModelBase):
        """EdgeFace S — slightly larger variant with improved accuracy."""

        @property
        def _edgeface_variant(self) -> str:
            return "edgeface_s_gamma_05"

        @property
        def name(self) -> str:
            return "EdgeFaceS"

        @property
        def display_name(self) -> str:
            return "EdgeFace S (EFaR'23)"

        @property
        def description(self) -> str:
            return (
                "EdgeFace S from EFaR 2023. Larger than XS with better accuracy "
                "while still suitable for edge deployment."
            )

        @property
        def embedding_size(self) -> int:
            return 512

else:
    print(
        f"EdgeFace models not registered "
        f"(PyTorch={TORCH_AVAILABLE}, InsightFace-align={INSIGHTFACE_ALIGN_AVAILABLE})"
    )
