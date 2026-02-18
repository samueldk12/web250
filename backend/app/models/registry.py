from typing import Dict, List, Optional, Type
from app.models.base import FaceRecognitionModel


class ModelRegistry:
    """Registry for face recognition models."""

    _models: Dict[str, FaceRecognitionModel] = {}
    _model_classes: Dict[str, Type[FaceRecognitionModel]] = {}

    @classmethod
    def register(cls, model_class: Type[FaceRecognitionModel]) -> Type[FaceRecognitionModel]:
        """Decorator to register a model class."""
        # Create instance to get name
        instance = model_class()
        cls._model_classes[instance.name] = model_class
        return model_class

    @classmethod
    def get(cls, name: str) -> Optional[FaceRecognitionModel]:
        """Get or create a model instance by name."""
        if name not in cls._models:
            if name not in cls._model_classes:
                return None
            cls._models[name] = cls._model_classes[name]()
        return cls._models[name]

    @classmethod
    def get_all(cls) -> List[FaceRecognitionModel]:
        """Get all registered model instances."""
        for name in cls._model_classes:
            if name not in cls._models:
                cls._models[name] = cls._model_classes[name]()
        return list(cls._models.values())

    @classmethod
    def list_names(cls) -> List[str]:
        """List all registered model names."""
        return list(cls._model_classes.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a model is registered."""
        return name in cls._model_classes
