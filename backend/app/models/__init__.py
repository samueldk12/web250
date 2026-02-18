# Models module - import all models to register them
from app.models.base import FaceRecognitionModel
from app.models.registry import ModelRegistry

# Import model implementations to register them
from app.models.deepface_models import (
    ArcFaceModel,
    FacenetModel,
    Facenet512Model,
    VGGFaceModel,
    OpenFaceModel,
    DeepIDModel,
    SFaceModel,
    GhostFaceNetModel,
)

# InsightFace is optional - only import if available
try:
    from app.models.insightface_models import InsightFaceModel
    _insightface_available = True
except ImportError:
    InsightFaceModel = None
    _insightface_available = False

__all__ = [
    'FaceRecognitionModel',
    'ModelRegistry',
    'ArcFaceModel',
    'FacenetModel',
    'Facenet512Model',
    'VGGFaceModel',
    'OpenFaceModel',
    'DeepIDModel',
    'SFaceModel',
    'GhostFaceNetModel',
]

if _insightface_available:
    __all__.append('InsightFaceModel')
