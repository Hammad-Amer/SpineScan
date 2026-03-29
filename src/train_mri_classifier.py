"""
Train MRI Type Classifier — MobileNetV3-Small (3 classes)
=========================================================
Classes: Sagittal T1, Sagittal T2/STIR, Axial T2
"Not MRI" is handled at inference time via confidence thresholding.

Usage:
    conda activate rsna
    python train_mri_classifier.py

Saves best checkpoint to checkpoints/mri_classifier.pth (~10 min on RTX 4060).
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

# ── Config ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BASE_DIR / "spine_model"
CKPT_DIR = BASE_DIR / "checkpoints"

CLASSES = ["Sagittal T1", "Sagittal T2/STIR", "Axial T2"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

SAMPLES_PER_CLASS = 5000
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 10
LR = 1e-3
NUM_WORKERS = 4
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Dataset ──────────────────────────────────────────────────
class MRIDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]

        img = cv2.imread(str(img_path))
        if img is None:
            # Fallback: black image
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, label


def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


# ── Data loading ─────────────────────────────────────────────
def load_and_sample_data():
    """Load CSVs, assign labels, subsample, and split."""
    sag_csv = REPO_DIR / "input" / "sagittal_df.csv"
    ax_csv = REPO_DIR / "input" / "axial_df.csv"

    if not sag_csv.exists():
        raise FileNotFoundError(f"Sagittal CSV not found: {sag_csv}")
    if not ax_csv.exists():
        raise FileNotFoundError(f"Axial CSV not found: {ax_csv}")

    # Sagittal: has series_description with "Sagittal T1" or "Sagittal T2/STIR"
    sag_df = pd.read_csv(sag_csv, usecols=["path", "series_description"])
    sag_df = sag_df[sag_df["series_description"].isin(["Sagittal T1", "Sagittal T2/STIR"])]
    sag_df["label"] = sag_df["series_description"]

    # Axial: all are "Axial T2"
    ax_df = pd.read_csv(ax_csv, usecols=["path"])
    ax_df["label"] = "Axial T2"

    # Resolve paths (relative to repo dir)
    sag_df["full_path"] = sag_df["path"].apply(lambda p: str(REPO_DIR / p))
    ax_df["full_path"] = ax_df["path"].apply(lambda p: str(REPO_DIR / p))

    # Subsample each class
    samples = []
    for cls_name, df in [("Sagittal T1", sag_df[sag_df["label"] == "Sagittal T1"]),
                          ("Sagittal T2/STIR", sag_df[sag_df["label"] == "Sagittal T2/STIR"]),
                          ("Axial T2", ax_df)]:
        n = min(SAMPLES_PER_CLASS, len(df))
        sampled = df.sample(n=n, random_state=SEED)
        print(f"  {cls_name}: {n} samples (from {len(df)} available)")
        samples.append(sampled[["full_path", "label"]])

    combined = pd.concat(samples, ignore_index=True)

    # Filter out images that don't exist
    exists_mask = combined["full_path"].apply(lambda p: os.path.exists(p))
    n_missing = (~exists_mask).sum()
    if n_missing > 0:
        print(f"  Warning: {n_missing} images not found on disk, skipping them")
    combined = combined[exists_mask].reset_index(drop=True)

    paths = combined["full_path"].tolist()
    labels = [CLASS_TO_IDX[l] for l in combined["label"]]

    # Stratified split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    print(f"  Train: {len(train_paths)} | Val: {len(val_paths)}")
    return train_paths, val_paths, train_labels, val_labels


# ── Training ─────────────────────────────────────────────────
def train():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    train_paths, val_paths, train_labels, val_labels = load_and_sample_data()

    train_ds = MRIDataset(train_paths, train_labels, transform=get_transforms(train=True))
    val_ds = MRIDataset(val_paths, val_labels, transform=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    print("Creating model: mobilenetv3_small_100")
    model = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    ckpt_path = CKPT_DIR / "mri_classifier.pth"

    for epoch in range(NUM_EPOCHS):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                with autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  lr={lr_now:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": CLASSES,
                "val_acc": val_acc,
                "epoch": epoch + 1,
                "image_size": IMAGE_SIZE,
            }, ckpt_path)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_acc:.4f}")
    print(f"Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    train()
