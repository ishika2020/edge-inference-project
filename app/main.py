import io
import json
import time

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI(title="Edge Inference Service")

MODEL_PATH = "model/model.onnx"
LABELS_PATH = "model/imagenet_classes.json"

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

with open(LABELS_PATH) as f:
    labels = json.load(f)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    input_tensor = preprocess(image)

    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0]
    top_idx = int(np.argmax(logits))
    confidence = float(np.exp(logits[top_idx]) / np.sum(np.exp(logits)))

    latency_ms = (time.time() - start) * 1000

    return {
        "predicted_class": labels.get(str(top_idx), "unknown"),
        "confidence": round(confidence, 4),
        "latency_ms": round(latency_ms, 2),
    }
