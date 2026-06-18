from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import cv2
import os
import io

app = FastAPI(title="Insurance Face Verification")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load OpenCV DNN face detector (lightweight, no TensorFlow) ──
print("Loading face models...")

PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL    = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"

PROTOTXT = "deploy.prototxt"
MODEL    = "face_model.caffemodel"

def download_if_missing(path, url):
    if not os.path.exists(path):
        import urllib.request
        print(f"Downloading {path}...")
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded {path}")

download_if_missing(PROTOTXT, PROTOTXT_URL)
download_if_missing(MODEL,    MODEL_URL)

face_net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
print("Face detector loaded!")

# ── Helpers ──

def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def detect_and_crop_face(img: np.ndarray):
    """Returns cropped face (128x128 RGB) or None."""
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    best = None
    best_conf = 0.5  # min confidence

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf > best_conf:
            best_conf = conf
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best = box.astype(int)

    if best is None:
        return None

    x1, y1, x2, y2 = best
    # add 10% padding
    pad_x = int((x2 - x1) * 0.1)
    pad_y = int((y2 - y1) * 0.1)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    face = img[y1:y2, x1:x2]
    face = cv2.resize(face, (128, 128))
    return face

def get_histogram_features(face: np.ndarray) -> np.ndarray:
    """LBP-style histogram feature vector for face matching."""
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Divide into 4x4 blocks, compute histogram per block
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

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot   = np.dot(a, b)
    norm  = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0

# ── Routes ──

@app.get("/")
def home():
    return {"status": "running", "model": "OpenCV DNN Face Detector"}


@app.post("/compare-face")
async def compare_face(
    reference_image: UploadFile = File(...),
    live_image:      UploadFile = File(...)
):
    try:
        ref_bytes  = await reference_image.read()
        live_bytes = await live_image.read()

        ref_img  = decode_image(ref_bytes)
        live_img = decode_image(live_bytes)

        if ref_img is None or live_img is None:
            return JSONResponse(status_code=400, content={
                "success": False, "error": "Invalid image format"
            })

        ref_face  = detect_and_crop_face(ref_img)
        live_face = detect_and_crop_face(live_img)

        # If face not detected, fall back to full image resize
        if ref_face is None:
            ref_face = cv2.resize(ref_img, (128, 128))
        if live_face is None:
            live_face = cv2.resize(live_img, (128, 128))

        ref_feat  = get_histogram_features(ref_face)
        live_feat = get_histogram_features(live_face)

        similarity = cosine_similarity(ref_feat, live_feat)

        # Map 0.5-1.0 similarity → 0-100 score
        score = round(max(0, min(100, (similarity - 0.5) * 200)), 2)

        if score >= 90:
            status = "Strong Match"
        elif score >= 70:
            status = "Match"
        elif score >= 40:
            status = "Review"
        else:
            status = "Fail"

        return JSONResponse({
            "success":    True,
            "verified":   score >= 40,
            "face_score": score,
            "similarity": round(similarity, 4),
            "status":     status
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False, "error": str(e)
        })
