# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpineSCAN is an FYP project that replicates the RSNA 2024 Lumbar Spine Degenerative Classification 2nd-place solution (Yuji Ariyasu) adapted for a single RTX 4060 8GB GPU. It classifies 25 conditions (5 pathologies × 5 disc levels, 3 severity grades each) from lumbar MRI DICOM files.

**Two-layer structure:**
- `SpineSCAN_FYP/` — this directory, containing wrapper scripts, the Streamlit app, and helper utilities
- `SpineSCAN_FYP/spine_model/` — the cloned upstream repo where all actual training/inference runs. All training commands (`python train_one_fold.py`, `python predict.py`, etc.) must be run from **inside** the repo directory.

## Key Commands

All commands assume `conda activate rsna` is already done.

```bash
# Verify environment (run before anything else)
python setup_check.py

# Apply RTX 4060 patch to the cloned repo (run once after cloning)
python patch_for_rtx4060.py

# ── Main training pipeline (~24-40h on RTX 4060) ──
run_pipeline.bat

# ── Extended training (run after main pipeline) ──
run_extra_training.bat          # 9 additional crop variants (~12-15h)
python find_noisy_label_local.py  # Noise detection (~2 min, run from SpineSCAN_FYP/)
run_noise_reduction.bat         # 10 _reduce_noise retrains (~10-15h)

# ── Inference & visualization ──
python inference_single.py --study_id 12345 --output results.json
python inference_single.py --list_studies
python gradcam_viz.py --image sagittal.png --condition spinal_canal_stenosis --level l4_l5
streamlit run app.py

# ── Inside the repo: train one config for fold 0 ──
cd spine_model
python train_one_fold.py -c rsna_sagittal_level_cl_spinal_v1 -f 0
python predict.py -c rsna_sagittal_level_cl_spinal_v1 -f 0
```

## Architecture

### Training Workflows (3 stages)

**Stage 1: `run_pipeline.bat`** — Main 10-step pipeline. Trains preprocessing models (slice estimation, YOLO) then the base 5 classification models. Output: checkpoints + OOF CSVs in `results/<config_name>/`.

**Stage 2: `run_extra_training.bat`** — Trains 9 missing crop-ratio variants for multi-model ensemble. Each runs `train_one_fold.py` then `predict.py` to generate OOF CSVs.

**Stage 3: Noise reduction** — `find_noisy_label_local.py` (run from SpineSCAN_FYP/) reads all OOF CSVs, ensembles predictions, flags high-loss samples as noisy labels. Then `run_noise_reduction.bat` retrains with clean labels.

### Multi-Model Ensemble (inference_single.py)

`predict_study()` ensembles all available models per condition group, auto-skipping missing checkpoints via `_ckpt_exists()`. The 5 groups are defined as module-level lists:

| Config group constant | Condition | Max models |
|---|---|---|
| `SAG_SPINAL_CONFIGS` | Spinal canal stenosis (sagittal) | 4 |
| `SAG_NFN_CONFIGS` | Neural foraminal narrowing (sagittal) | 6 |
| `SAG_SS_CONFIGS` | Subarticular stenosis (sagittal) | 8 |
| `AXIAL_SPINAL_CONFIGS` | Spinal canal stenosis (axial) | 4 |
| `AXIAL_NFN_SS_CONFIGS` | NFN + SS combined (axial) | 6 |

Sagittal models are averaged within their group, then fused 50/50 with the axial group average per (condition, level). If only the original 5 models exist, it works identically to single-model inference.

### Config System

Configs are **Python classes** in `spine_model/src/configs.py`, not YAML. Each config class inherits from `Baseline`. The `-c` flag to `train_one_fold.py` is the class name (`eval(args.config)(fold)`).

GPU mode mapping in `train_one_fold.py`:
- `'big'` → 8 devices (A100 cluster)
- `'v100'` → 4 devices
- `'small'` → 1 device ← **our mode after patching**

### Classification Model Configs

| Group | Base config | Additional crop variants |
|---|---|---|
| Axial Spinal | `rsna_axial_spinal_dis3_crop_x1_y2` | `crop_x05_y6` |
| Axial NFN+SS | `rsna_axial_ss_nfn_x2_y2_center_pad0` | `x2_y6_center_pad0`, `x2_y8_center_pad10` |
| Sagittal Spinal | `rsna_saggital_mil_spinal_crop_x03_y07` | `crop_x03_y05` |
| Sagittal NFN | `rsna_saggital_mil_nfn_crop_x07_y1_v2` | `crop_x15_y1_v2`, `crop_x03_y1_v2` |
| Sagittal SS | `rsna_saggital_mil_ss_crop_x03_y07_96` | `crop_x03_y05_96`, `crop_x03_y2_96`, `crop_x1_y07_96` |

All above have `_reduce_noise` variants (same name + suffix). These inherit from the base class and filter `train_df` against noise CSVs in their `__init__`.

**Noise CSV mapping:** `_reduce_noise` configs for spinal (axial + sagittal) read `results/noisy_target_level_1016.csv`. Sagittal NFN/SS use `results/noisy_target_level_th08.csv`. Axial NFN+SS use `results/noisy_target_level_th09.csv`.

| Pipeline step | Config name(s) |
|---|---|
| Sagittal slice est. stage 1 | `rsna_sagittal_level_cl_spinal_v1`, `rsna_sagittal_level_cl_nfn_v1` |
| Sagittal slice est. stage 2 | `rsna_sagittal_cl` |
| Sagittal YOLO | `rsna_10classes_yolox_x` |
| Axial YOLO | **verify from `axial_yolo.sh`** (placeholder: `rsna_axial_yolox_x`) |

To discover all available config names:
```python
import sys, inspect
sys.path.insert(0, 'spine_model')
from src.configs import *
[print(n) for n, c in inspect.getmembers(sys.modules[__name__], inspect.isclass) if 'rsna' in n.lower()]
```

### RTX 4060 Batch Size Clamping

`train_one_fold.py` (lines 69–84) uses **image-size-aware** batch clamping for `gpu='small'`:
- `image_size ≤ 128` → `max_bs = 8` (sagittal MIL models, ~3-4 GB)
- `image_size ≤ 256` → `max_bs = 4` (~4-5 GB)
- `image_size > 256` → `max_bs = 2` (axial 384px models, ~5-6 GB)

When `cfg.batch_size > max_bs`, it scales down batch and proportionally increases `grad_accumulations` to preserve the effective batch size. Always sets `precision=16` (AMP FP16) and `gradient_clip_val=1.0`.

### Noise Detection (find_noisy_label_local.py)

Adapted from the upstream `find_noisy_label.py` which required external team predictions (Ian's, Bartley's). This version uses **only our own OOF predictions**:
1. Loads `results/<config>/oof_fold0.csv` from all available crop variants
2. Ensembles predictions per condition group (average across crop variants)
3. Merges axial + sagittal with weighted averaging (0.7/0.5/0.8 for spinal/NFN/SS)
4. Computes per-sample loss = `|true_label - predicted_prob|`
5. Flags samples exceeding threshold as noisy
6. Outputs `results/noisy_target_level_th08.csv`, `th09.csv`, and `1016.csv` (copy of th08)

Must be run **from SpineSCAN_FYP/ directory** (it `os.chdir`s into the repo).

### Wrapper Scripts (in `SpineSCAN_FYP/`)

| File | Role |
|---|---|
| `setup_check.py` | Validates CUDA, packages, repo, data files, disk space. |
| `patch_for_rtx4060.py` | Idempotent patcher; creates `.rtx4060_patch_applied` sentinel. |
| `run_pipeline.bat` | 10-step main pipeline; `pushd`s into repo dir; skips completed steps. |
| `run_extra_training.bat` | Trains 9 crop variants (6 sagittal + 3 axial) with predict after each. |
| `run_noise_reduction.bat` | Trains 10 `_reduce_noise` models (5 high-priority + 5 additional). |
| `find_noisy_label_local.py` | Noise detection from our own OOF predictions only. |
| `inference_single.py` | Multi-model ensemble inference. `predict_study()` auto-discovers checkpoints. |
| `gradcam_viz.py` | Grad-CAM using `CONDITION_MODEL_MAP` for condition→model routing. Class index = `cond_idx * n_levels * n_severity + level_idx * n_severity + 2`. |
| `app.py` | Streamlit app, 4 tabs: Select Study & Analyze / Results / Grad-CAM / Pipeline Guide. |
| `mri_classifier.py` | MobileNetV3-Small MRI type classifier (Sagittal T1 / Sagittal T2 / Axial T2). |
| `train_mri_classifier.py` | Trains MRI classifier from sagittal/axial PNGs → `checkpoints/mri_classifier.pth`. |

### Data Paths

The upstream repo reads/writes everything relative to `input/`. Data must live at:
```
spine_model/input/
  train.csv, train_label_coordinates.csv
  train_images/            ← competition DICOMs
  axial_all_images/        ← generated by dcm_to_png.py
  sagittal_all_images/     ← generated by dcm_to_png.py
  axial_df.csv, sagittal_df.csv
  train_with_fold.csv      ← generated by preprocess.py
  sagittal_spinal_range2_rolling5.csv       ← sagittal spinal input for inference
  sagittal_{left,right}_nfn_range2_rolling5.csv
  sagittal_{left,right}_ss_range2_rolling5.csv
  axial_classification.csv                   ← axial classification input for inference
```

### RTX 4060 Constraints

- Always train **fold 0 only** (`-f 0`).
- OOM fallback: reduce `image_size` from 384 → 256 in the affected config class, or reduce `batch_size` to 1 and increase `grad_accumulations` proportionally.
- `_reduce_noise` configs require completed OOF predictions from base configs + noise CSVs; if they fail, skip them.

## Important Known Issues

1. **Axial YOLO config name is a placeholder** in `run_pipeline.bat`. After cloning, verify from `axial_yolo.sh`.

2. **Windows multiprocessing**: `dcm_to_png.py` and `preprocess.py` use `multiprocessing.Pool`. The patch adds `if __name__ == '__main__'` guards required on Windows. Preserve those guards.

3. **Kaggle kernel dependency**: The axial level estimation step (`kaggle kernels output yujiariyasu/axial-level-estimation`) must be run from a Kaggle environment. Its output CSV goes into `input/`.

4. **`tee` on Windows**: Batch files use `tee` for logging. Add `C:\Program Files\Git\usr\bin` to PATH, or remove `| tee -a "%LOGFILE%"` from failing lines.

5. **Note the typo `saggital`** (double g) in all sagittal MIL config names — this matches upstream and must not be "corrected".
