from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from deepface import DeepFace

import os
import shutil

app = FastAPI(title="Insurance Face Verification")

# Create folders
os.makedirs("uploads/reference", exist_ok=True)
os.makedirs("uploads/live", exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading ArcFace Model...")

# Preload model
arcface_model = DeepFace.build_model("ArcFace")

print("ArcFace Model Loaded Successfully")


@app.get("/")
def home():
    return {
        "status": "running"
    }


@app.post("/compare-face")
async def compare_face(
    reference_image: UploadFile = File(...),
    live_image: UploadFile = File(...)
):

    try:

        ref_path = f"uploads/reference/{reference_image.filename}"
        live_path = f"uploads/live/{live_image.filename}"

        with open(ref_path, "wb") as buffer:
            shutil.copyfileobj(reference_image.file, buffer)

        with open(live_path, "wb") as buffer:
            shutil.copyfileobj(live_image.file, buffer)

        result = DeepFace.verify(
            img1_path=ref_path,
            img2_path=live_path,
            model_name="ArcFace",
            enforce_detection=False
        )

        distance = float(result["distance"])

        threshold = 0.68

        score = round(
            max(
                0,
                min(
                    100,
                    (1 - (distance / threshold)) * 100
                )
            ),
            2
        )

        if score >= 90:
            status = "Strong Match"

        elif score >= 70:
            status = "Match"

        elif score >= 40:
            status = "Review"

        else:
            status = "Fail"

        return JSONResponse(
            {
                "success": True,
                "verified": bool(result["verified"]),
                "face_score": score,
                "distance": distance,
                "status": status
            }
        )

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )