import os
import tempfile
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import FACES_PATH, DISTANCE_THRESHOLD
from app.models import ModelRegistry
from app.services.ensemble import EnsembleService, EnsembleMethod, create_ensemble
from app.storage.face_db import get_face_db


app = FastAPI(
    title="Face Recognition API",
    description="API para registro e reconhecimento facial com multiplos modelos",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class ModelInfo(BaseModel):
    name: str
    display_name: str
    description: str
    embedding_size: int


class FaceResponse(BaseModel):
    id: str
    name: str
    image: str
    created_at: str
    models: List[str] = []


class MatchResponse(BaseModel):
    id: str
    name: str
    image: str
    distance: float
    confidence: float
    model_distances: Optional[dict] = None


class RecognitionResponse(BaseModel):
    success: bool
    matches: List[MatchResponse]
    message: str
    models_used: List[str]
    ensemble_method: Optional[str] = None


class RegisterResponse(BaseModel):
    success: bool
    face: FaceResponse
    message: str
    models_used: List[str]


# Health check
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "available_models": ModelRegistry.list_names()
    }


# List available models
@app.get("/api/models", response_model=List[ModelInfo])
async def list_models():
    models = ModelRegistry.get_all()
    return [model.to_dict() for model in models]


# List ensemble methods
@app.get("/api/ensemble-methods")
async def list_ensemble_methods():
    return [
        {"value": "average", "label": "Media", "description": "Media das distancias de todos os modelos"},
        {"value": "weighted", "label": "Media Ponderada", "description": "Media ponderada pela precisao do modelo"},
        {"value": "voting", "label": "Votacao", "description": "Maioria dos modelos decide o match"},
        {"value": "min", "label": "Minimo", "description": "Usa a menor distancia (mais confiante)"},
        {"value": "max", "label": "Maximo", "description": "Usa a maior distancia (mais conservador)"},
    ]


# List all registered faces
@app.get("/api/faces", response_model=List[FaceResponse])
async def list_faces():
    db = get_face_db()
    return db.get_all_faces()


# Get face image
@app.get("/api/faces/{face_id}/image")
async def get_face_image(face_id: str):
    db = get_face_db()
    face = db.get_face(face_id)
    if not face:
        raise HTTPException(status_code=404, detail="Face not found")

    image_path = FACES_PATH / face["image"]
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(image_path)


# Register a new face
@app.post("/api/faces/register", response_model=RegisterResponse)
async def register_face(
    name: str = Form(...),
    image: UploadFile = File(...),
    models: str = Form("ArcFace"),  # Comma-separated model names
):
    # Validate file type
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Parse model names
    model_names = [m.strip() for m in models.split(",") if m.strip()]
    if not model_names:
        model_names = ["ArcFace"]

    # Validate models
    valid_models = []
    for name_model in model_names:
        if ModelRegistry.is_registered(name_model):
            valid_models.append(name_model)
        else:
            print(f"Warning: Model {name_model} not found, skipping")

    if not valid_models:
        raise HTTPException(status_code=400, detail="No valid models specified")

    # Save uploaded file temporarily
    suffix = Path(image.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Get embeddings from all specified models
        embeddings = {}
        for model_name in valid_models:
            model = ModelRegistry.get(model_name)
            if model:
                embedding = model.get_embedding(tmp_path)
                if embedding:
                    embeddings[model_name] = embedding

        if not embeddings:
            raise HTTPException(
                status_code=400,
                detail="No face detected in image. Please upload a clear photo with a visible face."
            )

        # Save to database
        db = get_face_db()
        face = db.add_face(name, embeddings, tmp_path)

        return RegisterResponse(
            success=True,
            face=FaceResponse(
                id=face["id"],
                name=face["name"],
                image=face["image"],
                created_at=face["created_at"],
                models=list(embeddings.keys())
            ),
            message=f"Face registered successfully for {name}",
            models_used=list(embeddings.keys())
        )
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


# Recognize face in image
@app.post("/api/faces/recognize", response_model=RecognitionResponse)
async def recognize_face(
    image: UploadFile = File(...),
    models: str = Form("ArcFace"),  # Comma-separated model names
    ensemble_method: str = Form("average"),  # Ensemble method
    threshold: float = Form(DISTANCE_THRESHOLD),
):
    # Validate file type
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Parse model names
    model_names = [m.strip() for m in models.split(",") if m.strip()]
    if not model_names:
        model_names = ["ArcFace"]

    # Validate models
    valid_models = []
    for name_model in model_names:
        if ModelRegistry.is_registered(name_model):
            valid_models.append(name_model)

    if not valid_models:
        raise HTTPException(status_code=400, detail="No valid models specified")

    # Save uploaded file temporarily
    suffix = Path(image.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Get embeddings from all specified models
        embeddings = {}
        for model_name in valid_models:
            model = ModelRegistry.get(model_name)
            if model:
                embedding = model.get_embedding(tmp_path)
                if embedding:
                    embeddings[model_name] = embedding

        if not embeddings:
            return RecognitionResponse(
                success=False,
                matches=[],
                message="No face detected in image",
                models_used=[],
                ensemble_method=None
            )

        # Create ensemble for comparison
        ensemble = create_ensemble(list(embeddings.keys()), ensemble_method)

        # Find matches
        db = get_face_db()

        if len(embeddings) == 1:
            # Single model - use simple comparison
            model_name = list(embeddings.keys())[0]
            model = ModelRegistry.get(model_name)
            matches = db.find_matches_single_model(
                embeddings[model_name],
                model_name,
                model.compare,
                threshold
            )
            used_ensemble = None
        else:
            # Multiple models - use ensemble
            matches = db.find_matches(
                embeddings,
                ensemble.compare_embeddings,
                threshold,
                list(embeddings.keys())
            )
            used_ensemble = ensemble_method

        if matches:
            return RecognitionResponse(
                success=True,
                matches=[MatchResponse(**m) for m in matches],
                message=f"Found {len(matches)} match(es)",
                models_used=list(embeddings.keys()),
                ensemble_method=used_ensemble
            )
        else:
            return RecognitionResponse(
                success=True,
                matches=[],
                message="No matches found",
                models_used=list(embeddings.keys()),
                ensemble_method=used_ensemble
            )
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


# Delete a registered face
@app.delete("/api/faces/{face_id}")
async def delete_face(face_id: str):
    db = get_face_db()
    if db.delete_face(face_id):
        return {"success": True, "message": "Face deleted successfully"}
    raise HTTPException(status_code=404, detail="Face not found")


# Re-compute embeddings for a face with additional models
@app.post("/api/faces/{face_id}/add-models")
async def add_models_to_face(
    face_id: str,
    models: str = Form(...),  # Comma-separated model names
):
    db = get_face_db()
    face = db.get_face(face_id)

    if not face:
        raise HTTPException(status_code=404, detail="Face not found")

    # Parse model names
    model_names = [m.strip() for m in models.split(",") if m.strip()]

    # Get image path
    image_path = FACES_PATH / face["image"]
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Face image not found")

    # Generate new embeddings
    new_embeddings = {}
    for model_name in model_names:
        if not ModelRegistry.is_registered(model_name):
            continue
        # Skip if already has this model
        if model_name in face.get("embeddings", {}):
            continue

        model = ModelRegistry.get(model_name)
        if model:
            embedding = model.get_embedding(str(image_path))
            if embedding:
                new_embeddings[model_name] = embedding

    if new_embeddings:
        db.update_embeddings(face_id, new_embeddings)

    return {
        "success": True,
        "message": f"Added {len(new_embeddings)} model(s)",
        "models_added": list(new_embeddings.keys())
    }
