import os
import tempfile
import zipfile
import shutil
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

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}


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
    available: bool = True


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


class FaceDetection(BaseModel):
    face_index: int
    bbox: Optional[List[int]] = None  # [x, y, w, h]
    matches: List[MatchResponse]


class MultiRecognitionResponse(BaseModel):
    success: bool
    faces_detected: int
    faces: List[FaceDetection]
    message: str
    models_used: List[str]
    ensemble_method: Optional[str] = None


class RecognitionResponse(BaseModel):
    success: bool
    matches: List[MatchResponse]
    message: str
    models_used: List[str]
    ensemble_method: Optional[str] = None
    # Multi-face support
    faces_detected: int = 1
    faces: Optional[List[FaceDetection]] = None


class RegisterResponse(BaseModel):
    success: bool
    face: FaceResponse
    message: str
    models_used: List[str]


class BulkRegisterResult(BaseModel):
    name: str
    images_processed: int
    images_success: int
    images_failed: int
    face_ids: List[str]


class BulkRegisterResponse(BaseModel):
    success: bool
    total_persons: int
    total_images: int
    successful_registrations: int
    failed_registrations: int
    results: List[BulkRegisterResult]
    models_used: List[str]
    message: str


class BulkRecognizeResult(BaseModel):
    filename: str
    success: bool
    faces_detected: int = 0
    faces: List[FaceDetection] = []
    message: str


class BulkRecognizeResponse(BaseModel):
    success: bool
    total_images: int
    processed_images: int
    results: List[BulkRecognizeResult]
    models_used: List[str]
    ensemble_method: Optional[str] = None
    message: str


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


# Recognize faces in image (supports multiple faces)
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
        # Get ALL face embeddings from the primary model
        primary_model = ModelRegistry.get(valid_models[0])
        all_faces = primary_model.get_all_embeddings(tmp_path)

        if not all_faces:
            return RecognitionResponse(
                success=False,
                matches=[],
                message="No face detected in image",
                models_used=[],
                ensemble_method=None,
                faces_detected=0,
                faces=[]
            )

        db = get_face_db()
        face_detections = []
        all_matches = []

        # Process each detected face
        for face_data in all_faces:
            # Get embeddings from all models for this face
            # For multi-model, we need the specific face region
            # For simplicity, we use the primary model's embedding
            embeddings = {valid_models[0]: face_data.embedding}

            # If multiple models, get embeddings from each
            # Note: This uses the largest face for other models
            # A more sophisticated approach would crop the face and re-detect
            if len(valid_models) > 1:
                for model_name in valid_models[1:]:
                    model = ModelRegistry.get(model_name)
                    if model:
                        # Use get_all_embeddings to try to match face by index
                        model_faces = model.get_all_embeddings(tmp_path)
                        if model_faces and len(model_faces) > face_data.face_index:
                            embeddings[model_name] = model_faces[face_data.face_index].embedding
                        elif model_faces:
                            # Fallback to first face
                            embeddings[model_name] = model_faces[0].embedding

            # Create ensemble for comparison
            used_ensemble = None
            if len(embeddings) == 1:
                model_name = list(embeddings.keys())[0]
                model = ModelRegistry.get(model_name)
                matches = db.find_matches_single_model(
                    embeddings[model_name],
                    model_name,
                    model.compare,
                    threshold
                )
            else:
                ensemble = create_ensemble(list(embeddings.keys()), ensemble_method)
                matches = db.find_matches(
                    embeddings,
                    ensemble.compare_embeddings,
                    threshold,
                    list(embeddings.keys())
                )
                used_ensemble = ensemble_method

            face_detection = FaceDetection(
                face_index=face_data.face_index,
                bbox=list(face_data.bbox) if face_data.bbox else None,
                matches=[MatchResponse(**m) for m in matches] if matches else []
            )
            face_detections.append(face_detection)
            all_matches.extend(matches if matches else [])

        # For backward compatibility, include all matches in flat list
        total_matches = sum(len(f.matches) for f in face_detections)

        return RecognitionResponse(
            success=True,
            matches=[MatchResponse(**m) for m in all_matches] if all_matches else [],
            message=f"Found {len(all_faces)} face(s), {total_matches} match(es)",
            models_used=valid_models,
            ensemble_method=used_ensemble if len(valid_models) > 1 else None,
            faces_detected=len(all_faces),
            faces=face_detections
        )
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


# Register faces from ZIP file
@app.post("/api/faces/register-zip", response_model=BulkRegisterResponse)
async def register_faces_from_zip(
    zipfile_upload: UploadFile = File(...),
    models: str = Form("ArcFace"),
):
    """
    Register faces from a ZIP file.
    ZIP structure: each folder name is the person's name, containing their face images.
    """
    if not zipfile_upload.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")

    # Parse model names
    model_names = [m.strip() for m in models.split(",") if m.strip()]
    if not model_names:
        model_names = ["ArcFace"]

    # Validate models
    valid_models = [m for m in model_names if ModelRegistry.is_registered(m)]
    if not valid_models:
        raise HTTPException(status_code=400, detail="No valid models specified")

    # Create temp directory for extraction
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "upload.zip")

    try:
        # Save uploaded ZIP
        content = await zipfile_upload.read()
        with open(zip_path, 'wb') as f:
            f.write(content)

        # Extract ZIP
        extract_dir = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Process each folder (person)
        db = get_face_db()
        results = []
        total_images = 0
        total_success = 0
        total_failed = 0

        # Walk through extracted directory
        for root, dirs, files in os.walk(extract_dir):
            # Get relative path to determine person name
            rel_path = os.path.relpath(root, extract_dir)

            # Skip root directory
            if rel_path == '.':
                # Check if there are image files directly in root (no subfolders)
                image_files = [f for f in files if Path(f).suffix.lower() in IMAGE_EXTENSIONS]
                if image_files and not dirs:
                    # Images directly in ZIP without folders - use filename as name
                    for img_file in image_files:
                        total_images += 1
                        img_path = os.path.join(root, img_file)
                        person_name = Path(img_file).stem  # Use filename without extension

                        # Get embeddings
                        embeddings = {}
                        for model_name in valid_models:
                            model = ModelRegistry.get(model_name)
                            if model:
                                embedding = model.get_embedding(img_path)
                                if embedding:
                                    embeddings[model_name] = embedding

                        if embeddings:
                            face = db.add_face(person_name, embeddings, img_path)
                            total_success += 1
                            results.append(BulkRegisterResult(
                                name=person_name,
                                images_processed=1,
                                images_success=1,
                                images_failed=0,
                                face_ids=[face["id"]]
                            ))
                        else:
                            total_failed += 1
                continue

            # Get person name from folder structure (first level folder)
            path_parts = rel_path.split(os.sep)
            person_name = path_parts[0]

            # Skip if this isn't the direct subfolder (avoid double processing)
            if len(path_parts) > 1:
                continue

            # Get image files in this folder
            image_files = [f for f in files if Path(f).suffix.lower() in IMAGE_EXTENSIONS]

            if not image_files:
                continue

            person_success = 0
            person_failed = 0
            face_ids = []

            for img_file in image_files:
                total_images += 1
                img_path = os.path.join(root, img_file)

                # Get embeddings from all models
                embeddings = {}
                for model_name in valid_models:
                    model = ModelRegistry.get(model_name)
                    if model:
                        embedding = model.get_embedding(img_path)
                        if embedding:
                            embeddings[model_name] = embedding

                if embeddings:
                    face = db.add_face(person_name, embeddings, img_path)
                    face_ids.append(face["id"])
                    person_success += 1
                    total_success += 1
                else:
                    person_failed += 1
                    total_failed += 1

            if person_success > 0 or person_failed > 0:
                results.append(BulkRegisterResult(
                    name=person_name,
                    images_processed=len(image_files),
                    images_success=person_success,
                    images_failed=person_failed,
                    face_ids=face_ids
                ))

        return BulkRegisterResponse(
            success=total_success > 0,
            total_persons=len(results),
            total_images=total_images,
            successful_registrations=total_success,
            failed_registrations=total_failed,
            results=results,
            models_used=valid_models,
            message=f"Registered {total_success} faces from {len(results)} persons"
        )

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing ZIP: {str(e)}")
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


# Recognize faces from ZIP file
@app.post("/api/faces/recognize-zip", response_model=BulkRecognizeResponse)
async def recognize_faces_from_zip(
    zipfile_upload: UploadFile = File(...),
    models: str = Form("ArcFace"),
    ensemble_method: str = Form("average"),
    threshold: float = Form(DISTANCE_THRESHOLD),
):
    """
    Recognize faces from a ZIP file containing images.
    Returns recognition results for each image.
    """
    if not zipfile_upload.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")

    # Parse model names
    model_names = [m.strip() for m in models.split(",") if m.strip()]
    if not model_names:
        model_names = ["ArcFace"]

    # Validate models
    valid_models = [m for m in model_names if ModelRegistry.is_registered(m)]
    if not valid_models:
        raise HTTPException(status_code=400, detail="No valid models specified")

    # Create temp directory for extraction
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "upload.zip")

    try:
        # Save uploaded ZIP
        content = await zipfile_upload.read()
        with open(zip_path, 'wb') as f:
            f.write(content)

        # Extract ZIP
        extract_dir = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Find all images in ZIP
        image_files = []
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, extract_dir)
                    image_files.append((rel_path, full_path))

        if not image_files:
            raise HTTPException(status_code=400, detail="No images found in ZIP")

        # Create ensemble if multiple models
        ensemble = create_ensemble(valid_models, ensemble_method) if len(valid_models) > 1 else None
        db = get_face_db()
        results = []
        processed = 0

        for rel_path, img_path in image_files:
            processed += 1

            # Get ALL face embeddings from the primary model
            primary_model = ModelRegistry.get(valid_models[0])
            all_faces = primary_model.get_all_embeddings(img_path)

            if not all_faces:
                results.append(BulkRecognizeResult(
                    filename=rel_path,
                    success=False,
                    faces_detected=0,
                    faces=[],
                    message="No face detected"
                ))
                continue

            face_detections = []

            # Process each detected face
            for face_data in all_faces:
                embeddings = {valid_models[0]: face_data.embedding}

                # Get embeddings from other models if multi-model
                if len(valid_models) > 1:
                    for model_name in valid_models[1:]:
                        model = ModelRegistry.get(model_name)
                        if model:
                            model_faces = model.get_all_embeddings(img_path)
                            if model_faces and len(model_faces) > face_data.face_index:
                                embeddings[model_name] = model_faces[face_data.face_index].embedding
                            elif model_faces:
                                embeddings[model_name] = model_faces[0].embedding

                # Find matches
                if len(embeddings) == 1:
                    model_name = list(embeddings.keys())[0]
                    model = ModelRegistry.get(model_name)
                    matches = db.find_matches_single_model(
                        embeddings[model_name],
                        model_name,
                        model.compare,
                        threshold
                    )
                else:
                    matches = db.find_matches(
                        embeddings,
                        ensemble.compare_embeddings,
                        threshold,
                        valid_models
                    )

                face_detections.append(FaceDetection(
                    face_index=face_data.face_index,
                    bbox=list(face_data.bbox) if face_data.bbox else None,
                    matches=[MatchResponse(**m) for m in matches] if matches else []
                ))

            total_matches = sum(len(f.matches) for f in face_detections)
            results.append(BulkRecognizeResult(
                filename=rel_path,
                success=True,
                faces_detected=len(all_faces),
                faces=face_detections,
                message=f"Found {len(all_faces)} face(s), {total_matches} match(es)"
            ))

        return BulkRecognizeResponse(
            success=True,
            total_images=len(image_files),
            processed_images=processed,
            results=results,
            models_used=valid_models,
            ensemble_method=ensemble_method if len(valid_models) > 1 else None,
            message=f"Processed {processed} images"
        )

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing ZIP: {str(e)}")
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


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
