import os
from pathlib import Path

# Set TF_USE_LEGACY_KERAS before importing tensorflow
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Data paths
DATA_PATH = Path(os.getenv("DATA_PATH", "/app/data"))
FACES_PATH = DATA_PATH / "faces"
EMBEDDINGS_FILE = DATA_PATH / "embeddings.json"

# Model paths
DEEPFACE_HOME = os.getenv("DEEPFACE_HOME", str(Path.home() / ".deepface"))
INSIGHTFACE_HOME = os.getenv("INSIGHTFACE_HOME", str(Path.home() / ".insightface"))

# Set environment variables for libraries
os.environ["DEEPFACE_HOME"] = DEEPFACE_HOME
os.environ["INSIGHTFACE_HOME"] = INSIGHTFACE_HOME

# Ensure directories exist
FACES_PATH.mkdir(parents=True, exist_ok=True)

# Model settings
DEFAULT_MODEL = "ArcFace"
DEFAULT_DETECTOR = "retinaface"
DISTANCE_THRESHOLD = 0.4  # Cosine distance threshold for match
