import os
import tempfile
import zipfile
import shutil
from typing import List, Optional
from pathlib import Path

import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import FACES_PATH, DISTANCE_THRESHOLD
from app.storage.face_db import FaceDatabase, get_face_db
from app.models import ModelRegistry
from app.services.ensemble import EnsembleService, EnsembleMethod, create_ensemble
from app.services.settings import get_settings_service, AppSettings
from app.services.image_processing import ImagePreprocessor

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

# Folders to ignore when extracting ZIPs (macOS artifacts, etc.)
_ZIP_SKIP_DIRS = {'__MACOSX', '__pycache__', '.git'}


def _resolve_models(models_str: str) -> list[str]:
    """
    Parse a comma-separated model string.
    Returns all registered available models when the string is empty.
    """
    names = [m.strip() for m in models_str.split(",") if m.strip()]
    if not names:
        # Default: every model that is registered AND available
        names = [m.name for m in ModelRegistry.get_all() if m.is_available()]
    # Keep only models that actually exist in the registry
    return [n for n in names if ModelRegistry.is_registered(n)]


def _find_persons_in_zip(extract_dir: str) -> dict:
    """
    Scan an extracted ZIP directory and return {person_name: [abs_image_paths]}.

    Supported layouts
    ─────────────────
    Layout A – flat folders at root:
        extracted/
        ├── joao/foto1.jpg
        └── maria/foto1.jpg

    Layout B – single wrapper folder (common when zipping a directory):
        extracted/
        └── pessoas/
            ├── joao/foto1.jpg
            └── maria/foto1.jpg

    Layout C – images directly at root (no person sub-folders):
        extracted/foto_joao.jpg  →  person name = filename stem
    """
    from collections import defaultdict

    # Gather every image grouped by its immediate parent folder
    by_folder: dict = defaultdict(list)
    for root, dirs, files in os.walk(extract_dir):
        # Prune folders we should ignore
        dirs[:] = [d for d in dirs
                   if not d.startswith('.') and d not in _ZIP_SKIP_DIRS]
        for fname in files:
            if fname.startswith('.'):
                continue
            if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
                by_folder[root].append(os.path.join(root, fname))

    if not by_folder:
        return {}

    # Detect a single top-level wrapper folder and treat it as the new root
    root_entries = [e for e in os.listdir(extract_dir)
                    if not e.startswith('.') and e not in _ZIP_SKIP_DIRS]
    base_dir = extract_dir
    if len(root_entries) == 1:
        candidate = os.path.join(extract_dir, root_entries[0])
        if os.path.isdir(candidate):
            base_dir = candidate

    # Build person → images mapping
    persons: dict = defaultdict(list)
    for folder, images in by_folder.items():
        rel = os.path.relpath(folder, base_dir)
        if rel == '.':
            # Images sit directly in the base — use filename stem as person name
            for img_path in images:
                persons[Path(img_path).stem].append(img_path)
        else:
            # Person name = first path component under base_dir
            person_name = rel.split(os.sep)[0]
            persons[person_name].extend(images)

    return dict(persons)


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
    enabled: bool = True  # New field for UI config


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
async def list_models(all_models: bool = Query(False)):
    """
    List models.
    If all_models is True, returns all registered models.
    If False (default), returns only enabled models from settings.
    """
    settings = get_settings_service().get_settings()
    enabled_set = set(settings.enabled_models)
    
    models = ModelRegistry.get_all()
    result = []
    
    for model in models:
        info = model.to_dict()
        info["enabled"] = info["name"] in enabled_set
        
        # If we only want enabled models (for the main UI), skip disabled ones
        # But for settings page (all_models=True), show everything
        if all_models or info["enabled"]:
            result.append(info)
            
    return result


# Get settings
@app.get("/api/settings", response_model=AppSettings)
async def get_settings():
    return get_settings_service().get_settings()


# Update settings
@app.post("/api/settings", response_model=AppSettings)
async def update_settings(settings: AppSettings):
    return get_settings_service().update_settings(settings.dict())


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
    models: str = Form(""),  # Comma-separated names; empty = all available models
):
    # Validate file type
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    valid_models = _resolve_models(models)
    if not valid_models:
        raise HTTPException(status_code=400, detail="No valid models available")

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
    models: str = Form(""),  # Comma-separated model names; empty = all available
    ensemble_method: str = Form("average"),  # Ensemble method
    threshold: float = Form(DISTANCE_THRESHOLD),
    min_votes: Optional[int] = Form(None),
    min_vote_confidence: Optional[float] = Form(None),
):
    # Validate file type
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    valid_models = _resolve_models(models)
    if not valid_models:
        raise HTTPException(status_code=400, detail="No valid models specified")

    # Save uploaded file temporarily
    suffix = Path(image.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Define strategies: None = Original
        strategies = [None, "upscale", "clahe", "sharpen", "norm"]
        best_response = None
        best_match_score = -1.0 # Higher is better (confidence)
        processed_files = [] # Track for cleanup

        matches_found = False
        
        # Primary Model (first one selected)
        primary_model = ModelRegistry.get(valid_models[0])

        for strategy in strategies:
            # Prepare image path
            current_path = tmp_path
            
            if strategy:
                # Apply preprocessing
                if strategy == "upscale":
                    new_path = ImagePreprocessor.apply_upscaling(tmp_path)
                elif strategy == "clahe":
                    new_path = ImagePreprocessor.apply_clahe(tmp_path)
                elif strategy == "sharpen":
                    new_path = ImagePreprocessor.apply_sharpening(tmp_path)
                elif strategy == "norm":
                    new_path = ImagePreprocessor.apply_brightness_normalization(tmp_path)
                
                if new_path:
                    current_path = new_path
                    processed_files.append(new_path)
                else:
                    continue # Skip if processing failed

            print(f"Trying recognition with strategy: {strategy or 'original'}")
            
            # --- Detection & Recognition Logic (Reused) ---
            try:
                # Detect faces
                all_faces = primary_model.get_all_embeddings(current_path)

                if not all_faces:
                    continue # Try next strategy

                # Process matches
                db = get_face_db()
                face_detections = []
                all_matches = []
                
                # Setup models for this pass
                # Note: We re-extract embeddings for all models using current_path
                
                for face_data in all_faces:
                    # Collect embeddings
                    embeddings = {valid_models[0]: face_data.embedding}
                    
                    if len(valid_models) > 1:
                        for model_name in valid_models[1:]:
                            model = ModelRegistry.get(model_name)
                            if model:
                                model_faces = model.get_all_embeddings(current_path)
                                # Naive matching by index/size - ideally need IoU matching
                                # Simplification: take largest/corresponding index
                                if model_faces and len(model_faces) > face_data.face_index:
                                    embeddings[model_name] = model_faces[face_data.face_index].embedding
                                elif model_faces:
                                    embeddings[model_name] = model_faces[0].embedding
                    
                    # Compute Matches
                    if len(embeddings) == 1:
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
                        ensemble = create_ensemble(
                            list(embeddings.keys()),
                            ensemble_method,
                            min_votes=min_votes,
                            vote_threshold=1-min_vote_confidence if min_vote_confidence is not None else None
                        )
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

                # Evaluate this strategy's success
                current_match_count = len(all_matches)
                current_max_conf = max([m.confidence for m in all_matches]) if all_matches else 0.0

                # Create response object
                total_matches = sum(len(f.matches) for f in face_detections)
                response = RecognitionResponse(
                    success=True,
                    matches=[MatchResponse(**m) for m in all_matches] if all_matches else [],
                    message=f"Found {len(all_faces)} face(s), {total_matches} match(es) ({strategy or 'original'})",
                    models_used=valid_models,
                    ensemble_method=used_ensemble if len(valid_models) > 1 else None,
                    faces_detected=len(all_faces),
                    faces=face_detections
                )

                # HEURISTIC: Is this better than what we have?
                # 1. More matches is usually better? Or higher confidence?
                # Let's prioritize max confidence of top match
                
                is_better = False
                if best_response is None:
                    is_better = True
                elif current_max_conf > best_match_score:
                    # Significant improvement
                    is_better = True
                elif current_max_conf == best_match_score and current_match_count > len(best_response.matches):
                    # Same confidence, more matches
                    is_better = True
                
                if is_better:
                    best_response = response
                    best_match_score = current_max_conf
                    
                # Early Exit: If we found a very strong match (>85% confidence), stop trying
                # But allow at least trying 'original' and maybe 'upscale' if original failed
                if current_max_conf > 0.75: # High confidence
                     matches_found = True
                     break

            except Exception as e:
                print(f"Strategy {strategy} failed: {e}")
                continue

        # End of strategy loop
        
        # Cleanup processed files
        ImagePreprocessor.cleanup(processed_files)
        
        if best_response:
            return best_response
            
        # If absolutely nothing worked (no detection in any strategy)
        return RecognitionResponse(
            success=False,
            matches=[],
            message="No face detected in image (tried comprehensive enhancement)",
            models_used=[],
            ensemble_method=None,
            faces_detected=0,
            faces=[]
        )

    finally:
        # Clean up original temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# Register faces from ZIP file
@app.post("/api/faces/register-zip", response_model=BulkRegisterResponse)
def register_faces_from_zip(
    zipfile_upload: UploadFile = File(...),
    models: str = Form(""),  # Empty = all available models
):
    """
    Register faces from a ZIP file.

    Expected ZIP structure (each folder = one person):
        pessoas.zip
        ├── joao/
        │   ├── foto1.jpg
        │   └── foto2.jpg
        ├── maria/
        │   └── foto1.jpg
        └── pedro/
            └── img1.png

    A single top-level wrapper folder is handled automatically:
        archive.zip/
        └── pessoas/       ← auto-detected wrapper
            ├── joao/...
            └── maria/...
    """
    if not zipfile_upload.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")

    valid_models = _resolve_models(models)
    if not valid_models:
        raise HTTPException(status_code=400, detail="No valid models available")

    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "upload.zip")

    try:
        # Save and extract ZIP
        content = zipfile_upload.file.read()
        with open(zip_path, 'wb') as f:
            f.write(content)

        extract_dir = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)

        # Resolve person → image paths using the smart helper
        persons = _find_persons_in_zip(extract_dir)

        if not persons:
            raise HTTPException(
                status_code=400,
                detail="No images found in ZIP. "
                       "Expected structure: one sub-folder per person containing their photos."
            )

        db = get_face_db()
        results = []
        total_images = 0
        total_success = 0
        total_failed = 0

        for person_name, image_paths in persons.items():
            person_success = 0
            person_failed = 0
            face_ids = []

            for img_path in image_paths:
                total_images += 1
                embeddings = {}

                for model_name in valid_models:
                    model = ModelRegistry.get(model_name)
                    if model:
                        try:
                            embedding = model.get_embedding(img_path)
                            if embedding:
                                embeddings[model_name] = embedding
                        except Exception as e:
                            print(f"[ZIP] {model_name} failed on {img_path}: {e}")

                if embeddings:
                    face = db.add_face(person_name, embeddings, img_path)
                    face_ids.append(face["id"])
                    person_success += 1
                    total_success += 1
                else:
                    person_failed += 1
                    total_failed += 1
                    print(f"[ZIP] No face detected in {img_path} for person '{person_name}'")

            results.append(BulkRegisterResult(
                name=person_name,
                images_processed=len(image_paths),
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
            message=f"Registered {total_success} faces across {len(results)} person(s)"
        )

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing ZIP: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Recognize faces from ZIP file
@app.post("/api/faces/recognize-zip", response_model=BulkRecognizeResponse)
def recognize_faces_from_zip(
    zipfile_upload: UploadFile = File(...),
    models: str = Form(""),  # Empty = all available
    ensemble_method: str = Form("average"),
    threshold: float = Form(DISTANCE_THRESHOLD),
    min_votes: Optional[int] = Form(None),
    min_vote_confidence: Optional[float] = Form(None),
):
    """
    Recognize faces from a ZIP file containing images.
    Returns recognition results for each image.
    """
    if not zipfile_upload.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")

    valid_models = _resolve_models(models)
    if not valid_models:
        raise HTTPException(status_code=400, detail="No valid models specified")

    # Create temp directory for extraction
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "upload.zip")

    try:
        # Save uploaded ZIP
        content = zipfile_upload.file.read()
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
        ensemble = None
        if len(valid_models) > 1:
            ensemble = create_ensemble(
                valid_models,
                ensemble_method,
                min_votes=min_votes,
                vote_threshold=1-min_vote_confidence if min_vote_confidence is not None else None
            )
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
