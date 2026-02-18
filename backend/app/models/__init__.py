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

from app.models.insightface_models import (
    InsightFaceModel,
)

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
    'InsightFaceModel',
]
