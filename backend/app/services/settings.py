import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from app.config import DATA_PATH

SETTINGS_FILE = DATA_PATH / "settings.json"

class AppSettings(BaseModel):
    enabled_models: List[str] = [
        "ArcFace", "VGG-Face", "Facenet512", "InsightFace"
    ]
    default_threshold: float = 0.4
    det_backend: str = "retinaface"  # opencv, ssd, dlib, mtcnn, retinaface, mediapipe
    det_score_threshold: float = 0.9
    
    # Advanced detection params
    det_iou_threshold: float = 0.4
    
    # Appearance
    theme: str = "light"


class SettingsService:
    _instance = None
    _settings: Optional[AppSettings] = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = SettingsService()
        return cls._instance

    def __init__(self):
        self._load()

    def _load(self):
        """Load settings from file or create defaults."""
        print(f"Loading settings from {SETTINGS_FILE}")
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, "r") as f:
                    data = json.load(f)
                self._settings = AppSettings(**data)
                print("Settings loaded successfully")
            except Exception as e:
                print(f"Error loading settings: {e}")
                self._settings = AppSettings()
        else:
            print("Settings file not found, creating defaults")
            self._settings = AppSettings()
            self._save()

    def _save(self):
        """Save settings to file."""
        if self._settings:
            try:
                # Ensure directory exists
                SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
                print(f"Saving settings to {SETTINGS_FILE}")
                with open(SETTINGS_FILE, "w") as f:
                    f.write(self._settings.json(indent=2))
                print("Settings saved successfully")
            except Exception as e:
                print(f"FAILED to save settings: {e}")

    def get_settings(self) -> AppSettings:
        if not self._settings:
            self._load()
        return self._settings

    def update_settings(self, new_settings: Dict[str, Any]) -> AppSettings:
        """Update settings with provided values."""
        print(f"Updating settings with: {new_settings}")
        current = self.get_settings().dict()
        current.update(new_settings)
        self._settings = AppSettings(**current)
        self._save()
        return self._settings

    def get_enabled_models(self) -> List[str]:
        return self.get_settings().enabled_models

def get_settings_service():
    return SettingsService.get_instance()
