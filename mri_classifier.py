"""
MRI Type Classifier — Inference Module
=======================================
Classifies images as Sagittal T1, Sagittal T2/STIR, Axial T2, or Not MRI.

Public API:
    from mri_classifier import classify_mri
    result = classify_mri("path/to/image.png")
    # result = {
    #     'class': 'Sagittal T2/STIR',
    #     'confidence': 0.95,
    #     'probabilities': {'Sagittal T1': 0.02, 'Sagittal T2/STIR': 0.95, 'Axial T2': 0.03},
    #     'is_mri': True
    # }

CLI:
    python mri_classifier.py path/to/image.png
"""

import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Config ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CKPT_PATH = BASE_DIR / "checkpoints" / "mri_classifier.pth"

CLASSES = ["Sagittal T1", "Sagittal T2/STIR", "Axial T2"]
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.70

# ── Cached model ─────────────────────────────────────────────
_model = None
_device = None


def _get_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def _load_model():
    """Load model once and cache it."""
    global _model, _device

    if _model is not None:
        return _model, _device

    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"MRI classifier checkpoint not found at {CKPT_PATH}. "
            "Train first with: python train_mri_classifier.py"
        )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(CKPT_PATH, map_location=_device, weights_only=False)

    # Restore image size if saved
    img_size = checkpoint.get("image_size", IMAGE_SIZE)
    num_classes = len(checkpoint.get("classes", CLASSES))

    _model = timm.create_model("mobilenetv3_small_100", pretrained=False,
                                num_classes=num_classes)
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model = _model.to(_device)
    _model.eval()

    return _model, _device


def classify_mri(image_path_or_array) -> dict:
    """
    Classify an image as one of: Sagittal T1, Sagittal T2/STIR, Axial T2, or Not MRI.

    Args:
        image_path_or_array: str/Path to an image file, or numpy array (BGR format).

    Returns:
        dict with keys: 'class', 'confidence', 'probabilities', 'is_mri'
    """
    model, device = _load_model()
    transform = _get_transform()

    # Load image
    if isinstance(image_path_or_array, (str, Path)):
        img = cv2.imread(str(image_path_or_array))
        if img is None:
            return {
                "class": "Not MRI",
                "confidence": 0.0,
                "probabilities": {c: 0.0 for c in CLASSES},
                "is_mri": False,
            }
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_path_or_array, np.ndarray):
        if image_path_or_array.ndim == 3 and image_path_or_array.shape[2] == 3:
            img = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB)
        else:
            img = image_path_or_array
    else:
        raise TypeError(f"Expected str, Path, or numpy array, got {type(image_path_or_array)}")

    # Preprocess
    tensor = transform(image=img)["image"].unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    max_idx = int(np.argmax(probs))
    max_conf = float(probs[max_idx])
    predicted_class = CLASSES[max_idx]

    # Confidence thresholding: low confidence → "Not MRI"
    is_mri = max_conf >= CONFIDENCE_THRESHOLD
    if not is_mri:
        predicted_class = "Not MRI"

    return {
        "class": predicted_class,
        "confidence": max_conf,
        "probabilities": {c: float(probs[i]) for i, c in enumerate(CLASSES)},
        "is_mri": is_mri,
    }


# ── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mri_classifier.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    result = classify_mri(image_path)

    print(f"Class:      {result['class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Is MRI:     {result['is_mri']}")
    print("Probabilities:")
    for cls, prob in result["probabilities"].items():
        bar = "#" * int(prob * 40)
        print(f"  {cls:20s} {prob:.4f} {bar}")
