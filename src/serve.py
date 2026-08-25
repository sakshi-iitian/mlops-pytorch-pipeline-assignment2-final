import io
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from model import get_model

app = FastAPI(title="PyTorch CIFAR-10 Serving API")

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

MODEL_PATH = Path(os.getenv(
    "MODEL_PATH", "/app/checkpoints/classifier_v1.pt"
))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.4914, 0.4822, 0.4465],
        [0.2470, 0.2435, 0.2616],
    ),
])


def load_model():
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = get_model(
        checkpoint.get("architecture", "resnet18"),
        checkpoint.get("num_classes", 10),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()


@app.on_event("startup")
def startup_event():
    try:
        load_model()
    except Exception as exc:
        print(f"Model loading failed: {exc}")


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        content = await image.read()
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
        tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

        predicted_index = int(torch.tensor(probabilities).argmax().item())
        return {
            "predicted_class": CLASS_NAMES[predicted_index],
            "class_index": predicted_index,
            "probabilities": {
                CLASS_NAMES[i]: round(probabilities[i], 6)
                for i in range(len(CLASS_NAMES))
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
