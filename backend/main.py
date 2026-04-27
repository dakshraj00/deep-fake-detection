from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from mangum import Mangum
import torch
import io

from model import model, fake_label
from utils import preprocess

app = FastAPI(title="Deepfake Detector API")
handler = Mangum(app, lifespan="off")  # Lambda entry point

@app.get("/")
def root():
    return {"status": "running", "model": "ResNet-18 Deepfake Detector"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPG/PNG supported")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")

    tensor = preprocess(image)

    with torch.no_grad():
        output = model(tensor).squeeze()
        prob = torch.sigmoid(output).item()  # raw probability

    THRESHOLD = 0.5   # start with 0.5 for clarity

    # ✅ FIX: Use label_map properly
    # fake_label = 1 → model outputs P(Fake)
    # fake_label = 0 → model outputs P(Real)

    if fake_label == 1:
        # model predicts probability of FAKE
        prob_fake = prob
        prob_real = 1 - prob
        prediction = "Fake" if prob_fake >= THRESHOLD else "Real"
        confidence = prob_fake if prediction == "Fake" else prob_real

    else:
        # model predicts probability of REAL
        prob_real = prob
        prob_fake = 1 - prob
        prediction = "Real" if prob_real >= THRESHOLD else "Fake"
        confidence = prob_real if prediction == "Real" else prob_fake

    return {
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "probability_real": round(prob_real * 100, 2),
        "probability_fake": round(prob_fake * 100, 2),
    }