import base64
import io
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

THRESHOLD = 0.8
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
DISPLAY = {
    "brown-rust": "Brown rust",
    "healthy": "Healthy",
    "powdery-mildew": "Powdery mildew",
    "septoria": "Septoria",
    "yellow-rust": "Yellow rust",
}
NOTE = (
    "ResNet18 trained on the John Innes Centre wheat photos from the UK and Ireland. "
    "It can only choose brown rust, yellow rust, powdery mildew, septoria, or healthy. "
    "This is not a PAU diagnosis, and it has not been tested on Punjab fields."
)

_model = None
_classes = None
_temperature = None


def _checkpoint_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "models",
        "wheat_classifier_resnet18_seed0.pt",
    )


def _load():
    global _model, _classes, _temperature
    if _model is not None:
        return
    path = _checkpoint_path()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    _model = model
    _classes = classes
    _temperature = float(checkpoint["temperature"])


def classify_base64(image_base64):
    _load()
    raw = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    batch = EVAL_TF(image).unsqueeze(0)
    with torch.no_grad():
        logits = _model(batch)
        probs = F.softmax(logits / _temperature, dim=1)[0]
    index = int(torch.argmax(probs).item())
    probability = float(probs[index].item())
    return {
        "model": "ResNet18",
        "label": DISPLAY.get(_classes[index], _classes[index]),
        "probability": round(probability, 3),
        "abstained": probability < THRESHOLD,
        "threshold": THRESHOLD,
        "note": NOTE,
    }