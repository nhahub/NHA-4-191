import io
import time
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List

app = FastAPI(
    title="Road-Sense Detection API",
    description="YOLOv5 object detection for autonomous driving",
    version="1.0.0",
)

#  Fake model (replace with real YOLOv5 later)

def run_inference(image_bytes: bytes) -> dict:
    """Simulate model inference — replace with real model."""
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")

    # Simulate processing time
    time.sleep(0.01)

    return {
        "detections": [
            {"class": "Car",        "confidence": 0.92, "bbox": [100, 150, 300, 250]},
            {"class": "Pedestrian", "confidence": 0.85, "bbox": [400, 100, 480, 300]},
        ],
        "image_shape": list(img.shape[:2]),
    }


 
# Health check

@app.get("/health")
def health():
    return {"status": "ok"}


# Single-image endpoint

@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        result = run_inference(contents)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return JSONResponse(content={
        "filename":   file.filename,
        "detections": result["detections"],
        "image_shape": result["image_shape"],
    })


# Batch-image endpoint
@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Run inference on multiple images in one request.
    Returns list of detections per image.
    """
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > 16:
        raise HTTPException(status_code=400, detail="Max 16 images per batch")

    results = []
    for file in files:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not an image")
        contents = await file.read()
        try:
            result = run_inference(contents)
            results.append({
                "filename":   file.filename,
                "detections": result["detections"],
                "image_shape": result["image_shape"],
            })
        except ValueError as e:
            results.append({"filename": file.filename, "error": str(e)})

    return JSONResponse(content={"batch_size": len(files), "results": results})


# Run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
