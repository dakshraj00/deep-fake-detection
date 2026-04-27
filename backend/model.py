import os
import torch
import torch.nn as nn
from torchvision import models

MODEL_PATH = "best_resnet_deepfake.pt"

def build_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    base = models.resnet18(weights=None)
    base.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1)
    )

    # ✅ IMPORTANT FIX HERE
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    state = checkpoint.get("model_state", checkpoint)
    base.load_state_dict(state)
    base.eval()

    print("✅ Model loaded")
    return base, checkpoint
# Load everything once
model, _ckpt = build_model()

_label_map = _ckpt.get("label_map", {"Real": 1, "Fake": 0})
fake_label = _label_map["Fake"]

print("📌 Label map:", _label_map)
print("📌 Fake label:", fake_label)