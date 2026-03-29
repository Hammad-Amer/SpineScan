# SpineSCAN — Automated Lumbar Spine Degenerative Classification from MRI

An end-to-end deep learning system for classifying **25 degenerative conditions** of the lumbar spine from MRI scans, optimised to run on a single consumer-grade GPU (NVIDIA RTX 4060 8 GB).

Built as a Final Year Project, SpineSCAN takes raw DICOM MRI studies and produces severity predictions (Normal/Mild, Moderate, Severe) across 5 pathologies at 5 spinal levels, with Grad-CAM visual explanations to support clinical interpretability.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Pathologies Detected

| # | Condition | Levels | Severity Classes |
|---|-----------|--------|-----------------|
| 1 | Spinal Canal Stenosis | L1/L2 – L5/S1 | Normal/Mild, Moderate, Severe |
| 2 | Left Neural Foraminal Narrowing | L1/L2 – L5/S1 | Normal/Mild, Moderate, Severe |
| 3 | Right Neural Foraminal Narrowing | L1/L2 – L5/S1 | Normal/Mild, Moderate, Severe |
| 4 | Left Subarticular Stenosis | L1/L2 – L5/S1 | Normal/Mild, Moderate, Severe |
| 5 | Right Subarticular Stenosis | L1/L2 – L5/S1 | Normal/Mild, Moderate, Severe |

**5 conditions x 5 levels x 3 severity grades = 25 classification outputs per study**

---

## Key Features

- **Multi-Model Ensemble** — Up to 28 models (14 base + 14 noise-reduced) averaged per condition group for robust predictions
- **Consumer GPU Optimisation** — Image-size-aware batch clamping, FP16 mixed precision, and gradient accumulation to fit within 8 GB VRAM
- **Noise-Aware Training** — Automated detection of mislabelled samples using ensemble disagreement, followed by clean-label retraining
- **Grad-CAM Visualisation** — Heatmap overlays showing which regions of the MRI influenced each prediction
- **MRI Type Detection** — MobileNetV3-based classifier to automatically identify sagittal T1, sagittal T2/STIR, and axial T2 sequences
- **Web Interface** — Streamlit application with study browser, results dashboard, and interactive Grad-CAM viewer

---

## System Architecture

```
MRI DICOM Study
       │
       ▼
┌─────────────────────────────────────────────────┐
│              PREPROCESSING PIPELINE              │
│  DICOM→PNG ─→ Slice Estimation ─→ YOLO Crops    │
└───────────────────────┬─────────────────────────┘
                        │
       ┌────────────────┼────────────────┐
       ▼                ▼                ▼
  ┌──────────┐   ┌──────────────┐   ┌──────────┐
  │ Sagittal │   │  Sagittal    │   │  Axial   │
  │ Spinal   │   │  NFN + SS    │   │  Models  │
  │ Models   │   │  Models      │   │          │
  └────┬─────┘   └──────┬───────┘   └────┬─────┘
       │                │                 │
       └────────────────┼─────────────────┘
                        ▼
              ┌──────────────────┐
              │  Ensemble Fusion │
              │  (50/50 Sag+Ax)  │
              └────────┬─────────┘
                       ▼
              25 Severity Predictions
              + Grad-CAM Heatmaps
```

### Model Groups

| Group | View | Condition | Crop Variants | Reduce-Noise Variants |
|-------|------|-----------|:---:|:---:|
| Sagittal Spinal | Sagittal T2 | Spinal Canal Stenosis | 2 | 2 |
| Sagittal NFN | Sagittal T1/T2 | Neural Foraminal Narrowing | 3 | 3 |
| Sagittal SS | Sagittal T1/T2 | Subarticular Stenosis | 4 | 4 |
| Axial Spinal | Axial T2 | Spinal Canal Stenosis | 2 | 2 |
| Axial NFN+SS | Axial T2 | NFN + SS (combined) | 3 | 3 |
| **Total** | | | **14** | **14** |

The ensemble gracefully degrades — if only a subset of models are trained, only those are used.

---

## Project Structure

```
SpineSCAN/
│
├── src/                            # Core source code
│   ├── inference.py                # Multi-model ensemble inference engine
│   ├── gradcam.py                  # Grad-CAM heatmap generation
│   ├── mri_classifier.py           # MRI sequence type classifier
│   ├── train_mri_classifier.py     # MRI classifier training script
│   ├── noise_detection.py          # Noisy label detection pipeline
│   ├── gpu_patch.py                # GPU adaptation patches
│   └── setup_check.py             # Environment validation
│
├── scripts/                        # Training pipeline automation
│   ├── run_pipeline.bat            # Main training pipeline (10 steps)
│   ├── run_extra_training.bat      # Additional crop-variant training (9 models)
│   └── run_noise_reduction.bat     # Clean-label retraining (10 models)
│
├── website/                        # Web application (React + Node.js)
│   ├── frontend/                   # React UI
│   └── backend/                    # Node.js API server
│
├── app.py                          # Streamlit web interface (entry point)
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- **GPU**: NVIDIA GPU with 8+ GB VRAM (tested on RTX 4060)
- **CUDA**: 11.8 or 12.x
- **Python**: 3.10
- **Conda**: Anaconda or Miniconda

### 1. Clone the repository

```bash
git clone https://github.com/Hammad-Amer/SpineScan.git
cd SpineScan
```

### 2. Create the conda environment

```bash
conda create -n rsna python=3.10 -y
conda activate rsna
pip install -r requirements.txt
```

### 3. Clone the base model repository

```bash
git clone https://github.com/yujiariyasu/rsna_2024_lumbar_spine_degenerative_classification.git spine_model
```

### 4. Apply GPU optimisation patches

```bash
python src/gpu_patch.py
```

### 5. Verify setup

```bash
python src/setup_check.py
```

### 6. Prepare data

Place RSNA 2024 competition data into `spine_model/input/`:
- `train.csv`, `train_label_coordinates.csv`
- `train_images/` (DICOM files)

---

## Training

Training is split into three stages designed to run overnight:

### Stage 1: Main Pipeline (~24–40h)

Trains preprocessing models (slice estimation, YOLO object detection) and the 5 base classification models.

```bash
conda activate rsna
scripts\run_pipeline.bat
```

### Stage 2: Crop-Variant Expansion (~12–15h)

Trains 9 additional classification models with different crop ratios to improve ensemble diversity.

```bash
scripts\run_extra_training.bat
```

### Stage 3: Noise Reduction (~10–15h)

Detects mislabelled training samples using ensemble disagreement, then retrains all models on the cleaned dataset.

```bash
python src/noise_detection.py
scripts\run_noise_reduction.bat
```

---

## Inference

### Command Line

```bash
conda activate rsna

# Run analysis on a training study
python inference_single.py --study_id 12345 --output results.json

# List available studies
python inference_single.py --list_studies
```

### Streamlit App

```bash
streamlit run app.py
```

Four-tab interface:
1. **Select Study & Analyse** — Browse studies, view MRI series, run inference
2. **Results** — Severity table, annotated spine diagram, probability breakdown
3. **Grad-CAM** — Interactive heatmap viewer with condition/level/view selection
4. **Pipeline Guide** — Setup walkthrough and training commands

---

## GPU Optimisation Details

The system uses image-size-aware batch clamping to maximise VRAM utilisation on consumer GPUs:

| Image Size | Max Batch Size | VRAM Usage | Model Types |
|:---:|:---:|:---:|---|
| ≤ 128px | 8 | ~3–4 GB | Sagittal MIL models |
| ≤ 256px | 4 | ~4–5 GB | Medium resolution models |
| > 256px | 2 | ~5–6 GB | Axial 384px models |

When the configured batch size exceeds the limit, gradient accumulation steps are increased proportionally to preserve the effective batch size. All training uses FP16 mixed precision.

---

## Technical Approach

This project builds on the architecture of the [2nd-place solution](https://github.com/yujiariyasu/spine_model) from the RSNA 2024 Lumbar Spine Degenerative Classification competition by Yuji Ariyasu, with significant adaptations:

1. **Single-GPU adaptation** — Replaced multi-GPU distributed training with single-device training, automatic batch size scaling, and gradient accumulation
2. **Self-contained noise detection** — Developed `find_noisy_label_local.py` to perform ensemble-based noise detection using only our own model predictions, eliminating the dependency on external team outputs
3. **Dynamic ensemble inference** — Built `inference_single.py` to automatically discover and ensemble all available model checkpoints at inference time
4. **Clinical interface** — Developed the Streamlit application and Grad-CAM pipeline for interpretable predictions
5. **MRI type classification** — Added automated MRI sequence detection using MobileNetV3-Small

---

## License

This project is for educational purposes as part of a Final Year Project.
The base model architecture is from the [RSNA 2024 competition](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification).
