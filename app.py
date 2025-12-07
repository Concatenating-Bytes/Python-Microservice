from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import numpy as np
import cv2
import base64
import os
import logging
from face_utils import FacePipeline
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_PATH = "./data/face_database.npy"
VERIFICATION_THRESHOLD = 0.5 

app = FastAPI(title="Face Auth Microservice")

pipeline = None
face_db = {}
db_lock = asyncio.Lock()

# Input Schemas
class EnrollRequest(BaseModel):
    user_id: str
    image_b64: str

class VerifyRequest(BaseModel):
    user_id: str
    image_b64: str
    threshold: Optional[float] = VERIFICATION_THRESHOLD

class IdentifyRequest(BaseModel):
    image_b64: str
    threshold: Optional[float] = VERIFICATION_THRESHOLD

# --- Helper Functions ---
def load_db():
    global face_db
    if os.path.exists(DATABASE_PATH):
        try:
            face_db = np.load(DATABASE_PATH, allow_pickle=True).item()
            logger.info(f"Database loaded. {len(face_db)} users enrolled.")
        except Exception as e:
            logger.error(f"Failed to load DB: {e}")
            face_db = {}
    else:
        logger.warning("Database not found. Creating new.")
        face_db = {}

def save_db():
    # Save asynchronously to avoid blocking
    np.save(DATABASE_PATH, face_db)
    logger.info("Database saved to disk.")

def decode_image(b64_string):
    try:
        # Remove header if present (e.g., "data:image/jpeg;base64,")
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        
        image_data = base64.b64decode(b64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        return None

# --- Lifespan Events ---

@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = FacePipeline()
    load_db()

    os.makedirs("./data", exist_ok=True)  # Create data directory if not exists

# --- API Endpoints ---

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "enrolled_users": len(face_db),
        "model_loaded": pipeline is not None
    }

@app.post("/enroll")
async def enroll_user(request: EnrollRequest, background_tasks: BackgroundTasks):
    img = decode_image(request.image_b64)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    try:
        embedding = pipeline.process_image(img)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        async with db_lock:
            face_db[request.user_id] = embedding
            np.save(DATABASE_PATH, face_db)
        background_tasks.add_task(save_db)
        
        return {"success": True, "message": f"User {request.user_id} enrolled successfully"}
    
    except Exception as e:
        logger.error(f"Enrollment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify")
async def verify_user(request: VerifyRequest):
    if request.user_id not in face_db:
        raise HTTPException(status_code=404, detail="User not found")
        
    img = decode_image(request.image_b64)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    try:
        live_embedding = pipeline.process_image(img)
        if live_embedding is None:
            raise HTTPException(status_code=400, detail="No face detected")

        stored_embedding = face_db[request.user_id]
        score = pipeline.cosine_similarity(live_embedding, stored_embedding)
        
        is_match = bool(score > request.threshold)
        
        return {
            "verified": is_match,
            "similarity": float(score),
            "threshold_used": request.threshold
        }
    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(status_code=500, detail="Processing error")

@app.post("/identify")
async def identify_user(request: IdentifyRequest):
    img = decode_image(request.image_b64)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid base64 image")
        
    if not face_db:
        raise HTTPException(status_code=404, detail="Database is empty")

    try:
        live_embedding = pipeline.process_image(img)
        if live_embedding is None:
            raise HTTPException(status_code=400, detail="No face detected")

        best_score = -1.0
        best_user = "unknown"

        for user_id, stored_emb in face_db.items():
            score = pipeline.cosine_similarity(live_embedding, stored_emb)
            if score > best_score:
                best_score = score
                best_user = user_id
        
        if best_score < request.threshold:
            best_user = "unknown"

        return {
            "user_id": best_user,
            "similarity": float(best_score)
        }

    except Exception as e:
        logger.error(f"Identification error: {e}")
        raise HTTPException(status_code=500, detail="Processing error")