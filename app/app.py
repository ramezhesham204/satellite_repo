"""
Milestone 4: inference API for the EuroSAT-trained model.

Run:
    pip install flask torch torchvision pillow
    python app/app.py

POST an image file (jpg/png, any size — it's resized to 64x64) to /predict.
"""

import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from model import build_model  # noqa: E402

MODEL_NAME = "cnn"  # match whichever checkpoint you trained: "cnn" or "resnet18"
CKPT = ROOT / "models" / f"best_{MODEL_NAME}.pt"
CLASSES_FILE = ROOT / "models" / "classes.json"

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = json.loads(CLASSES_FILE.read_text())
model = build_model(MODEL_NAME, len(classes)).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

tf = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(device), "classes": classes})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "send an image file under key 'file'"}), 400
    try:
        img = Image.open(io.BytesIO(request.files["file"].read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"could not read image: {e}"}), 400

    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())

    return jsonify({
        "predicted_class": classes[idx],
        "confidence": float(probs[idx]),
        "all_probabilities": {c: float(p) for c, p in zip(classes, probs)},
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
