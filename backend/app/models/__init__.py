# Models module - import all models to register them
from app.models.base import FaceRecognitionModel
from app.models.registry import ModelRegistry

# Import DeepFace model implementations (always available)
from app.models.deepface_models import (
    ArcFaceModel,
    FacenetModel,
    Facenet512Model,
    VGGFaceModel,
    OpenFaceModel,
    DeepIDModel,
    SFaceModel,
    GhostFaceNetModel,
    DeepFaceMetaModel,
    DlibModel,
)

# InsightFace models (optional — registered internally if insightface is installed)
try:
    from app.models.insightface_models import (
        InsightFaceModel,
        InsightFaceAntelopeV2Model,
        InsightFaceBuffaloMModel,
    )
    _insightface_available = True
except ImportError:
    InsightFaceModel = None
    InsightFaceAntelopeV2Model = None
    InsightFaceBuffaloMModel = None
    _insightface_available = False

# PyTorch / EdgeFace models (optional — registered internally if torch is installed)
try:
    from app.models.pytorch_models import EdgeFaceXSModel, EdgeFaceSModel
    _pytorch_available = True
except ImportError:
    EdgeFaceXSModel = None
    EdgeFaceSModel = None
    _pytorch_available = False
except Exception:
    EdgeFaceXSModel = None
    EdgeFaceSModel = None
    _pytorch_available = False

# face_recognition library (optional — registered internally if library is installed)
try:
    from app.models.dlib_models import FaceRecognitionLibModel
    _face_recognition_available = True
except ImportError:
    FaceRecognitionLibModel = None
    _face_recognition_available = False
except Exception:
    FaceRecognitionLibModel = None
    _face_recognition_available = False

__all__ = [
    'FaceRecognitionModel',
    'ModelRegistry',
    # DeepFace models
    'ArcFaceModel',
    'FacenetModel',
    'Facenet512Model',
    'VGGFaceModel',
    'OpenFaceModel',
    'DeepIDModel',
    'SFaceModel',
    'GhostFaceNetModel',
    'DeepFaceMetaModel',
    'DlibModel',
]

if _insightface_available:
    __all__ += ['InsightFaceModel', 'InsightFaceAntelopeV2Model', 'InsightFaceBuffaloMModel']

if _pytorch_available:
    __all__ += ['EdgeFaceXSModel', 'EdgeFaceSModel']

if _face_recognition_available:
    __all__ += ['FaceRecognitionLibModel']
