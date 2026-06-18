from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import os

app = FastAPI(title="Insurance Face Verification")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "running"}

def decode_image(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def get_features(img):
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bh, bw = h // 4, w // 4
    features = []
    for row in range(4):
        for col in range(4):
            block = gray[row*bh:(row+1)*bh, col*bw:(col+1)*bw]
            hist = cv2.calcHist([block], [0], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
    return np.array(features, dtype=np.float32)

@app.post("/compare-face")
async def compare_face(
    reference_image: UploadFile = File(...),
    live_image: UploadFile = File(...)
):
    try:
        ref_img  = decode_image(await reference_image.read())
        live_img = decode_image(await live_image.read())

        ref_feat  = get_features(ref_img)
        live_feat = get_features(live_img)

        dot  = np.dot(ref_feat, live_feat)
        norm = np.linalg.norm(ref_feat) * np.linalg.norm(live_feat)
        similarity = float(dot / norm) if norm > 0 else 0.0

        score = round(max(0, min(100, (similarity - 0.5) * 200)), 2)

        if score >= 90:   status = "Strong Match"
        elif score >= 70: status = "Match"
        elif score >= 40: status = "Review"
        else:             status = "Fail"

        return JSONResponse({
            "success": True,
            "verified": score >= 40,
            "face_score": score,
            "status": status
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False, "error": str(e)
        })
