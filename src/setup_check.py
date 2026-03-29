"""
SpineSCAN FYP — Environment & Dependency Checker
Run this BEFORE starting any training to verify everything is in order.

Usage:
    conda activate rsna
    python setup_check.py
"""

import sys
import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BASE_DIR / "spine_model"
INPUT_DIR = REPO_DIR / "input"

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

failures = []
warnings = []

def check(name, condition, fail_msg, warn=False):
    if condition:
        print(f"{PASS} {name}")
    elif warn:
        print(f"{WARN} {name} — {fail_msg}")
        warnings.append(name)
    else:
        print(f"{FAIL} {name} — {fail_msg}")
        failures.append(name)

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ── Section 1: Python & CUDA ──────────────────────────────────
section("Python & CUDA")

check("Python >= 3.10",
      sys.version_info >= (3, 10),
      f"Got {sys.version_info.major}.{sys.version_info.minor}, need 3.10+")

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    check("CUDA available (torch.cuda.is_available())", cuda_ok,
          "CUDA not detected — check drivers and PyTorch install")
    if cuda_ok:
        dev_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  {INFO} GPU: {dev_name}")
        print(f"  {INFO} VRAM: {vram_gb:.1f} GB")
        check("VRAM >= 6 GB", vram_gb >= 6,
              f"Only {vram_gb:.1f} GB detected — may OOM during training")
        cuda_ver = torch.version.cuda
        print(f"  {INFO} CUDA version (PyTorch): {cuda_ver}")
        check("FP16 support (CUDA >= 11.0)",
              float(cuda_ver.split(".")[0]) >= 11,
              f"CUDA {cuda_ver} may not support AMP properly")
    torch_ver = torch.__version__
    print(f"  {INFO} PyTorch: {torch_ver}")
    check("PyTorch >= 2.0",
          int(torch_ver.split(".")[0]) >= 2,
          f"Got {torch_ver}, need 2.0+ for best FP16 support", warn=True)
except ImportError:
    check("PyTorch installed", False, "Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

# ── Section 2: Required packages ─────────────────────────────
section("Required Python Packages")

PACKAGES = {
    "pydicom":              "pip install pydicom",
    "cv2":                  "pip install opencv-python",
    "sklearn":              "pip install scikit-learn",
    "pandas":               "pip install pandas",
    "numpy":                "pip install numpy",
    "albumentations":       "pip install albumentations",
    "timm":                 "pip install timm",
    "tqdm":                 "pip install tqdm",
    "pytorch_lightning":    "pip install pytorch-lightning==1.9.4",
    "dicomsdl":             "pip install dicomsdl",
    "monai":                "pip install monai",
    "itk":                  "pip install itk",
    "streamlit":            "pip install streamlit",
    "pytorch_grad_cam":     "pip install grad-cam",
}

for pkg, install_cmd in PACKAGES.items():
    try:
        __import__(pkg)
        check(pkg, True, "")
    except ImportError:
        check(pkg, False, f"Run: {install_cmd}")

# ── Section 3: Repository ─────────────────────────────────────
section("Repository")

check("Repo cloned",
      REPO_DIR.exists(),
      f"Clone repo:\n  cd {BASE_DIR}\n  git clone https://github.com/yujiariyasu/spine_model.git")

if REPO_DIR.exists():
    check("train_one_fold.py present", (REPO_DIR / "train_one_fold.py").exists(), "Repo may be incomplete")
    check("predict.py present",        (REPO_DIR / "predict.py").exists(),         "Repo may be incomplete")
    check("dcm_to_png.py present",     (REPO_DIR / "dcm_to_png.py").exists(),      "Repo may be incomplete")
    check("src/configs.py present",    (REPO_DIR / "src" / "configs.py").exists(), "Repo may be incomplete")
    check("patch applied",             (REPO_DIR / ".rtx4060_patch_applied").exists(),
          "Run: python patch_for_rtx4060.py", warn=True)

# ── Section 4: Data files ─────────────────────────────────────
section("Data Files")

check("input/ dir exists", INPUT_DIR.exists(),
      f"Create it: mkdir {INPUT_DIR}")

if INPUT_DIR.exists():
    train_csv   = INPUT_DIR / "train.csv"
    label_csv   = INPUT_DIR / "train_label_coordinates.csv"
    train_imgs  = INPUT_DIR / "train_images"
    axial_imgs  = INPUT_DIR / "axial_all_images"
    sag_imgs    = INPUT_DIR / "sagittal_all_images"
    axial_df    = INPUT_DIR / "axial_df.csv"
    sag_df      = INPUT_DIR / "sagittal_df.csv"

    check("train.csv",                    train_csv.exists(),
          "Download competition data first")
    check("train_label_coordinates.csv",  label_csv.exists(),
          "Download competition data first")
    check("train_images/ DICOM folder",   train_imgs.exists() and any(train_imgs.iterdir()),
          "Download and unzip competition DICOMs")
    check("axial_all_images/ (PNGs)",     axial_imgs.exists(),
          "Run: python dcm_to_png.py  (step 3 in pipeline)", warn=True)
    check("sagittal_all_images/ (PNGs)",  sag_imgs.exists(),
          "Run: python dcm_to_png.py  (step 3 in pipeline)", warn=True)
    check("axial_df.csv",                 axial_df.exists(),
          "Run: python dcm_to_png.py  (step 3 in pipeline)", warn=True)
    check("sagittal_df.csv",              sag_df.exists(),
          "Run: python dcm_to_png.py  (step 3 in pipeline)", warn=True)

    # count studies
    if train_imgs.exists():
        n_studies = len(list(train_imgs.iterdir()))
        print(f"  {INFO} Training studies found: {n_studies}")

# ── Section 5: Kaggle API ─────────────────────────────────────
section("Kaggle API")

kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
check("~/.kaggle/kaggle.json exists", kaggle_json.exists(),
      "Download from Kaggle → Settings → API → Create New Token")
if kaggle_json.exists():
    import stat
    mode = kaggle_json.stat().st_mode & 0o777
    check("kaggle.json permissions (600)", mode == 0o600 or os.name == 'nt',
          "Run: chmod 600 ~/.kaggle/kaggle.json", warn=True)

# ── Section 6: Disk space ──────────────────────────────────────
section("Disk Space")

try:
    import shutil
    total, used, free = shutil.disk_usage(str(BASE_DIR))
    free_gb = free / 1e9
    print(f"  {INFO} Free disk space: {free_gb:.1f} GB")
    check("At least 80 GB free", free_gb >= 80,
          f"Only {free_gb:.1f} GB free — need ~100 GB total for data + PNGs + checkpoints", warn=True)
except Exception as e:
    print(f"  {WARN} Could not check disk space: {e}")

# ── Summary ────────────────────────────────────────────────────
section("Summary")
if failures:
    print(f"\n{FAIL} {len(failures)} check(s) FAILED — fix these before proceeding:")
    for f in failures:
        print(f"     • {f}")
if warnings:
    print(f"\n{WARN} {len(warnings)} warning(s) — review before training:")
    for w in warnings:
        print(f"     • {w}")
if not failures:
    print(f"\n{PASS} All critical checks passed!")
    if not warnings:
        print("  Ready to run the pipeline. Start with: run_pipeline.bat")
    else:
        print("  Critical checks OK — address warnings when possible.")
