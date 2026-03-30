"""
SpineSCAN FYP — Single-Study Inference
========================================
Runs the trained classification models on a single patient study
and returns severity predictions for all 25 conditions.

Usage:
    # Using a study_id from the training set (for demo):
    python inference_single.py --study_id 12345

    # Save results to JSON:
    python inference_single.py --study_id 12345 --output results.json

    # List available study IDs:
    python inference_single.py --list_studies

Conditions predicted (5 levels x 5 pathologies = 25 total):
    - Spinal Canal Stenosis
    - Left Neural Foraminal Narrowing
    - Right Neural Foraminal Narrowing
    - Left Subarticular Stenosis
    - Right Subarticular Stenosis

Severity classes: Normal/Mild, Moderate, Severe
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BASE_DIR / "spine_model"
sys.path.insert(0, str(REPO_DIR))

# Change working directory to repo so relative paths in configs work
os.chdir(str(REPO_DIR))

# Import all config classes at module level (import * not allowed inside functions)
from src.configs import *  # noqa: F401,F403

LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
LEVEL_NUM_MAP = {1: "L1/L2", 2: "L2/L3", 3: "L3/L4", 4: "L4/L5", 5: "L5/S1"}
CONDITIONS = [
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis",
]
SEVERITY = ["Normal/Mild", "Moderate", "Severe"]


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def parse_args():
    parser = argparse.ArgumentParser(description="SpineSCAN: single-study inference")
    parser.add_argument("--study_id", "-s", type=int, default=None,
                        help="Study ID from training set (for demo)")
    parser.add_argument("--dicom_dir", "-d", type=str, default=None,
                        help="Path to DICOM folder (for new studies)")
    parser.add_argument("--output", "-o", default=None,
                        help="Save JSON results to this path")
    parser.add_argument("--list_studies", action="store_true",
                        help="List available study IDs and exit")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_model_from_checkpoint(config_name, fold, device):
    """Load a trained model from its checkpoint."""
    import torch

    try:
        cfg = eval(config_name)(fold)
    except TypeError:
        cfg = eval(config_name)()

    if type(cfg.image_size) == int:
        cfg.image_size = (cfg.image_size, cfg.image_size)
    transform_dict = cfg.transform(cfg.image_size)
    cfg.transform = transform_dict['val']  # Use val transform (no augmentation)

    ckpt_path = REPO_DIR / "results" / config_name / f"fold_{fold}.ckpt"
    if not ckpt_path.exists():
        ckpt_path = REPO_DIR / "results" / config_name / f"last_fold{fold}.ckpt"

    state_dict = torch.load(str(ckpt_path), map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Strip 'model.model.' or 'model.' prefix
    clean_dict = {}
    has_model_model = any(k.startswith('model.model.') for k in state_dict)
    has_model = any(k.startswith('model.') for k in state_dict)
    for k, v in state_dict.items():
        if has_model_model:
            clean_dict[k[12:]] = v
        elif has_model:
            clean_dict[k[6:]] = v
        else:
            clean_dict[k] = v

    cfg.model.load_state_dict(clean_dict)
    cfg.model.to(device)
    cfg.model.eval()
    return cfg


def run_sagittal_mil_inference(cfg, study_df, device):
    """Run inference using SagittalMILDataset-style preprocessing."""
    import torch
    import cv2
    import math
    import albumentations as A

    def angle_of_line(x1, y1, x2, y2):
        return math.degrees(math.atan2(-(y2 - y1), x2 - x1))

    results = []
    for _, row in study_df.iterrows():
        path = row['path']
        paths = row['paths'].split(',')
        l_point = (row['l_x'], row['l_y'])
        r_point = (row['r_x'], row['r_y'])

        try:
            origin_img = cv2.imread(path)
            origin_size = origin_img.shape[:2]
        except:
            continue

        images = []
        for p in paths:
            if p == 'nan' or not os.path.exists(p):
                image = np.zeros((origin_size[0], origin_size[1], 3), dtype=np.uint8)
            else:
                image = cv2.imread(p)
                if image is None:
                    image = np.zeros((origin_size[0], origin_size[1], 3), dtype=np.uint8)
                else:
                    image = cv2.resize(image, (origin_size[1], origin_size[0]))

            # Rotate based on landmarks
            a, b = l_point, r_point
            rotate_angle = angle_of_line(a[0], a[1], b[0], b[1])
            transform = A.Compose(
                [A.Rotate(limit=(-rotate_angle, -rotate_angle), p=1.0)],
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
            )
            t = transform(image=image, keypoints=[a, b])
            image = t["image"]
            a, b = t["keypoints"]
            a = (max(0, a[0]), max(0, a[1]))
            b = (max(0, b[0]), max(0, b[1]))

            if cfg.box_crop:
                if hasattr(cfg, 'xy_center_point') and cfg.xy_center_point:
                    x = int((a[0] + b[0]) / 2)
                    y = int((a[1] + b[1]) / 2)
                else:
                    x = int(b[0])
                    y = int(b[1])
                w = abs(b[0] - a[0])
                h = image.shape[0] * 0.2
                crop_x = int(w * cfg.box_crop_x_ratio)
                crop_y = int(h * cfg.box_crop_y_ratio)
                x_min = max(x - crop_x, 0)
                y_min = max(y - crop_y, 0)
                image = image[y_min:y + crop_y, x_min:x + crop_x]
            else:
                # Simple crop between keypoints
                x1, y1 = int(min(a[0], b[0])), int(min(a[1], b[1]) - image.shape[0] * 0.1)
                x2, y2 = int(max(a[0], b[0])), int(max(a[1], b[1]) + image.shape[0] * 0.1)
                y1, y2 = max(0, y1), min(image.shape[0], y2)
                x1, x2 = max(0, x1), min(image.shape[1], x2)
                image = image[y1:y2, x1:x2]

            if image.size == 0:
                image = np.zeros((cfg.image_size[0], cfg.image_size[1], 3), dtype=np.uint8)

            image = cfg.transform(image=image.astype(np.uint8))['image']
            images.append(image)

        images = np.stack(images, 0)
        images = torch.tensor(images).float().unsqueeze(0).to(device)  # (1, N, C, H, W)

        with torch.no_grad():
            logits = cfg.model(images).cpu().numpy()  # (1, num_classes)

        results.append({
            'level': row['level'],
            'left_right': row.get('left_right', None),
            'logits': logits[0],
        })

    return results


def run_axial_inference(cfg, study_df, device):
    """Run inference using ClassificationDataset-style preprocessing."""
    import torch
    import cv2

    results = []
    for _, row in study_df.iterrows():
        path = row['path']
        image = cv2.imread(path)
        if image is None:
            continue
        image = image[:, :, ::-1]  # BGR → RGB

        if cfg.box_crop:
            box = row[['x_min', 'y_min', 'x_max', 'y_max']].values.astype(int)
            x_pad = (box[2] - box[0]) // 2 * cfg.box_crop_x_ratio
            y_pad = (box[3] - box[1]) // 2 * cfg.box_crop_y_ratio
            x_min = int(max(box[0] - x_pad, 0))
            y_min = int(max(box[1] - y_pad, 0))
            if hasattr(cfg, 'box_crop_y_upper_ratio'):
                y_upper_pad = (box[3] - box[1]) // 2 * cfg.box_crop_y_upper_ratio
                y_min = int(max(box[1] - y_upper_pad, 0))
            x_max = int(min(box[2] + x_pad, image.shape[1]))
            y_max = int(min(box[3] + y_pad, image.shape[0]))
            image = image[y_min:y_max, x_min:x_max]

        if image.size == 0:
            continue

        image = cfg.transform(image=image.astype(np.uint8))['image']
        image = torch.tensor(image).float().unsqueeze(0).to(device)  # (1, C, H, W)

        with torch.no_grad():
            logits = cfg.model(image).cpu().numpy()  # (1, num_classes)

        results.append({
            'pred_level': row.get('pred_level', None),
            'left_right': row.get('left_right', None),
            'logits': logits[0],
        })

    return results


def _ckpt_exists(config_name, fold):
    """Check if a checkpoint exists for this config."""
    ckpt1 = REPO_DIR / "results" / config_name / f"fold_{fold}.ckpt"
    ckpt2 = REPO_DIR / "results" / config_name / f"last_fold{fold}.ckpt"
    return ckpt1.exists() or ckpt2.exists()


# ── Multi-model ensemble config lists ─────────────────────────
# Each group lists all crop variants + reduce_noise variants.
# At inference time, only configs with existing checkpoints are used.

SAG_SPINAL_CONFIGS = [
    "rsna_saggital_mil_spinal_crop_x03_y07",
    "rsna_saggital_mil_spinal_crop_x03_y05",
    "rsna_saggital_mil_spinal_crop_x03_y07_reduce_noise",
    "rsna_saggital_mil_spinal_crop_x03_y05_reduce_noise",
]

SAG_NFN_CONFIGS = [
    "rsna_saggital_mil_nfn_crop_x07_y1_v2",
    "rsna_saggital_mil_nfn_crop_x15_y1_v2",
    "rsna_saggital_mil_nfn_crop_x03_y1_v2",
    "rsna_saggital_mil_nfn_crop_x07_y1_v2_reduce_noise",
    "rsna_saggital_mil_nfn_crop_x15_y1_v2_reduce_noise",
    "rsna_saggital_mil_nfn_crop_x03_y1_v2_reduce_noise",
]

SAG_SS_CONFIGS = [
    "rsna_saggital_mil_ss_crop_x03_y07_96",
    "rsna_saggital_mil_ss_crop_x03_y05_96",
    "rsna_saggital_mil_ss_crop_x03_y2_96",
    "rsna_saggital_mil_ss_crop_x1_y07_96",
    "rsna_saggital_mil_ss_crop_x03_y07_96_reduce_noise",
    "rsna_saggital_mil_ss_crop_x03_y05_96_reduce_noise",
    "rsna_saggital_mil_ss_crop_x03_y2_96_reduce_noise",
    "rsna_saggital_mil_ss_crop_x1_y07_96_reduce_noise",
]

AXIAL_SPINAL_CONFIGS = [
    "rsna_axial_spinal_dis3_crop_x1_y2",
    "rsna_axial_spinal_dis3_crop_x05_y6",
    "rsna_axial_spinal_dis3_crop_x1_y2_reduce_noise",
    "rsna_axial_spinal_dis3_crop_x05_y6_reduce_noise",
]

AXIAL_NFN_SS_CONFIGS = [
    "rsna_axial_ss_nfn_x2_y2_center_pad0",
    "rsna_axial_ss_nfn_x2_y6_center_pad0",
    "rsna_axial_ss_nfn_x2_y8_center_pad10",
    "rsna_axial_ss_nfn_x2_y2_center_pad0_reduce_noise",
    "rsna_axial_ss_nfn_x2_y6_center_pad0_reduce_noise",
    "rsna_axial_ss_nfn_x2_y8_center_pad10_reduce_noise",
]


def _run_sagittal_ensemble(config_list, study_id, csv_path_template, fold, device,
                           condition_key, predictions, sides=None):
    """Run inference across multiple sagittal configs and average predictions.

    Args:
        config_list: List of config names to try
        study_id: Study ID
        csv_path_template: CSV path (use {side} placeholder for sided conditions)
        fold: Fold number
        device: torch device
        condition_key: Key in predictions dict (or template with {side})
        predictions: predictions dict to update
        sides: None for center-only, or ['left', 'right'] for sided conditions
    """
    import torch
    import pandas as pd

    available = [c for c in config_list if _ckpt_exists(c, fold)]
    if not available:
        print(f"    [WARN] No checkpoints found for any of: {config_list[0]}...")
        return

    print(f"    Ensembling {len(available)} model(s)")

    if sides is None:
        # Center-only (e.g., spinal)
        df = pd.read_csv(csv_path_template)
        study_df = df[df.study_id == study_id]
        if len(study_df) == 0:
            print(f"    [WARN] No data for study {study_id}")
            return

        level_probs = {}  # level -> list of prob arrays
        for config_name in available:
            cfg = load_model_from_checkpoint(config_name, fold, device)
            results = run_sagittal_mil_inference(cfg, study_df, device)
            for r in results:
                level = r['level']
                probs = softmax(r['logits'].reshape(1, -1))[0]
                level_probs.setdefault(level, []).append(probs)
            del cfg
            torch.cuda.empty_cache()

        for level, prob_list in level_probs.items():
            avg = np.mean(prob_list, axis=0)
            predictions[condition_key][level] = avg
        print(f"    {condition_key}: {len(level_probs)} levels")
    else:
        # Sided (e.g., NFN left/right, SS left/right)
        for side in sides:
            csv_path = csv_path_template.format(side=side)
            df = pd.read_csv(csv_path)
            study_df = df[df.study_id == study_id]
            cond_name = condition_key.format(side=side)
            if len(study_df) == 0:
                print(f"    [WARN] No {side} data for study {study_id}")
                continue

            level_probs = {}
            for config_name in available:
                cfg = load_model_from_checkpoint(config_name, fold, device)
                results = run_sagittal_mil_inference(cfg, study_df, device)
                for r in results:
                    level = r['level']
                    probs = softmax(r['logits'].reshape(1, -1))[0]
                    level_probs.setdefault(level, []).append(probs)
                del cfg
                torch.cuda.empty_cache()

            for level, prob_list in level_probs.items():
                avg = np.mean(prob_list, axis=0)
                predictions[cond_name][level] = avg
            print(f"    {side} {condition_key.split('{')[0]}: {len(level_probs)} levels")


def _run_axial_ensemble(config_list, study_id, fold, device, predictions, mode):
    """Run inference across multiple axial configs and average predictions.

    Args:
        mode: 'spinal' or 'nfn_ss'
    """
    import torch
    import pandas as pd

    available = [c for c in config_list if _ckpt_exists(c, fold)]
    if not available:
        print(f"    [WARN] No checkpoints found for axial {mode}")
        return

    csv_path = REPO_DIR / "input" / "axial_classification.csv"
    df = pd.read_csv(csv_path)
    study_df = df[df.study_id == study_id]

    if len(study_df) == 0:
        return

    print(f"    Ensembling {len(available)} axial {mode} model(s)")

    if mode == 'spinal':
        # Collect probs per level across all configs
        level_probs = {}
        for config_name in available:
            cfg = load_model_from_checkpoint(config_name, fold, device)
            results = run_axial_inference(cfg, study_df, device)
            for r in results:
                level_num = r['pred_level']
                level = LEVEL_NUM_MAP.get(level_num)
                if level is None:
                    continue
                probs = softmax(r['logits'].reshape(1, -1))[0]
                level_probs.setdefault(level, []).append(probs)
            del cfg
            torch.cuda.empty_cache()

        # Average axial predictions, then fuse with existing sagittal
        for level, prob_list in level_probs.items():
            axial_avg = np.mean(prob_list, axis=0)
            if predictions["spinal_canal_stenosis"][level] is not None:
                predictions["spinal_canal_stenosis"][level] = (
                    predictions["spinal_canal_stenosis"][level] + axial_avg) / 2
            else:
                predictions["spinal_canal_stenosis"][level] = axial_avg
        print(f"    Axial spinal: {len(level_probs)} levels")

    elif mode == 'nfn_ss':
        # Collect NFN and SS probs per (level, side)
        nfn_level_probs = {}  # (side, level) -> list of prob arrays
        ss_level_probs = {}
        for config_name in available:
            cfg = load_model_from_checkpoint(config_name, fold, device)
            results = run_axial_inference(cfg, study_df, device)
            for r in results:
                level_num = r['pred_level']
                level = LEVEL_NUM_MAP.get(level_num)
                if level is None:
                    continue
                logits = r['logits']
                if len(logits) >= 6:
                    nfn_probs = softmax(logits[:3].reshape(1, -1))[0]
                    ss_probs = softmax(logits[3:6].reshape(1, -1))[0]
                    for side in ['left', 'right']:
                        nfn_level_probs.setdefault((side, level), []).append(nfn_probs)
                        ss_level_probs.setdefault((side, level), []).append(ss_probs)
            del cfg
            torch.cuda.empty_cache()

        # Average and fuse
        for (side, level), prob_list in nfn_level_probs.items():
            axial_avg = np.mean(prob_list, axis=0)
            nfn_key = f"{side}_neural_foraminal_narrowing"
            if predictions[nfn_key][level] is not None:
                predictions[nfn_key][level] = (predictions[nfn_key][level] + axial_avg) / 2
            else:
                predictions[nfn_key][level] = axial_avg

        for (side, level), prob_list in ss_level_probs.items():
            axial_avg = np.mean(prob_list, axis=0)
            ss_key = f"{side}_subarticular_stenosis"
            if predictions[ss_key][level] is not None:
                predictions[ss_key][level] = (predictions[ss_key][level] + axial_avg) / 2
            else:
                predictions[ss_key][level] = axial_avg

        n_nfn = len(set(l for _, l in nfn_level_probs))
        n_ss = len(set(l for _, l in ss_level_probs))
        print(f"    Axial NFN: {n_nfn} levels, SS: {n_ss} levels")


def predict_study(study_id, fold=0, device="cuda"):
    """Run all classification models for a single study_id.

    Automatically ensembles all available crop variants and _reduce_noise
    models. Gracefully degrades — if only the original 5 models exist,
    works exactly as the single-model version.
    """
    import torch
    import pandas as pd

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    predictions = {cond: {level: None for level in LEVELS} for cond in CONDITIONS}

    # ── 1. Sagittal Spinal Canal Stenosis (ensemble) ───────────
    print("  [1/5] Sagittal spinal ensemble...")
    _run_sagittal_ensemble(
        SAG_SPINAL_CONFIGS, study_id,
        csv_path_template=str(REPO_DIR / "input" / "sagittal_spinal_range2_rolling5.csv"),
        fold=fold, device=device,
        condition_key="spinal_canal_stenosis",
        predictions=predictions,
        sides=None,
    )

    # ── 2. Sagittal NFN (ensemble) ─────────────────────────────
    print("  [2/5] Sagittal NFN ensemble...")
    _run_sagittal_ensemble(
        SAG_NFN_CONFIGS, study_id,
        csv_path_template=str(REPO_DIR / "input" / "sagittal_{side}_nfn_range2_rolling5.csv"),
        fold=fold, device=device,
        condition_key="{side}_neural_foraminal_narrowing",
        predictions=predictions,
        sides=['left', 'right'],
    )

    # ── 3. Sagittal SS (ensemble) ──────────────────────────────
    print("  [3/5] Sagittal SS ensemble...")
    _run_sagittal_ensemble(
        SAG_SS_CONFIGS, study_id,
        csv_path_template=str(REPO_DIR / "input" / "sagittal_{side}_ss_range2_rolling5.csv"),
        fold=fold, device=device,
        condition_key="{side}_subarticular_stenosis",
        predictions=predictions,
        sides=['left', 'right'],
    )

    # ── 4. Axial Spinal (ensemble + fuse with sagittal) ───────
    print("  [4/5] Axial spinal ensemble...")
    _run_axial_ensemble(AXIAL_SPINAL_CONFIGS, study_id, fold, device, predictions, mode='spinal')

    # ── 5. Axial NFN + SS (ensemble + fuse with sagittal) ─────
    print("  [5/5] Axial NFN+SS ensemble...")
    _run_axial_ensemble(AXIAL_NFN_SS_CONFIGS, study_id, fold, device, predictions, mode='nfn_ss')

    return predictions


def format_predictions(predictions, study_id):
    """Convert numpy arrays to JSON-serializable format."""
    output = {
        "study_id": str(study_id),
        "model": "SpineSCAN Multi-Model Ensemble",
        "conditions": {}
    }

    for cond in CONDITIONS:
        output["conditions"][cond] = {}
        for level in LEVELS:
            probs = predictions[cond].get(level)
            if probs is not None:
                pred_idx = int(np.argmax(probs))
                output["conditions"][cond][level] = {
                    "Normal/Mild": round(float(probs[0]), 4),
                    "Moderate": round(float(probs[1]), 4),
                    "Severe": round(float(probs[2]), 4),
                    "predicted_class": SEVERITY[pred_idx],
                    "confidence": round(float(probs[pred_idx]), 4),
                }
            else:
                output["conditions"][cond][level] = {
                    "Normal/Mild": 0.0, "Moderate": 0.0, "Severe": 0.0,
                    "predicted_class": "N/A", "confidence": 0.0,
                }

    return output


def print_results_table(results):
    """Print human-readable results table."""
    print("\n" + "=" * 72)
    print(f"  RESULTS - Study: {results['study_id']}")
    print("=" * 72)
    print(f"{'Condition':<38} {'Level':<8} {'Predicted':>12} {'Conf':>6}")
    print("-" * 72)

    for cond, levels in results["conditions"].items():
        cond_label = cond.replace("_", " ").title()
        for level, preds in levels.items():
            pred = preds.get("predicted_class", "N/A")
            conf = preds.get("confidence", 0.0)
            if pred == "Severe":
                prefix = "  [!]"
            elif pred == "Moderate":
                prefix = "  [-]"
            else:
                prefix = "  [ ]"
            print(f"{prefix} {cond_label:<34} {level:<8} {pred:>12} {conf:>5.1%}")
        print()

    print("=" * 72)
    print("  Legend:  [ ] Normal/Mild   [-] Moderate   [!] Severe")
    print("=" * 72)


def list_available_studies():
    """List study IDs available in the preprocessed data."""
    import pandas as pd
    csv_path = REPO_DIR / "input" / "sagittal_spinal_range2_rolling5.csv"
    if not csv_path.exists():
        print("[ERROR] Preprocessed data not found. Run the training pipeline first.")
        return
    df = pd.read_csv(csv_path)
    study_ids = sorted(df.study_id.unique())
    print(f"\nAvailable study IDs ({len(study_ids)} studies):")
    print("-" * 50)
    # Print in columns
    cols = 5
    for i in range(0, len(study_ids), cols):
        row = study_ids[i:i + cols]
        print("  " + "  ".join(str(s) for s in row))
    print(f"\nUsage: python inference_single.py --study_id {study_ids[0]}")


def main():
    args = parse_args()

    if args.list_studies:
        list_available_studies()
        return

    if args.study_id is None and args.dicom_dir is None:
        print("[ERROR] Provide either --study_id or --dicom_dir")
        print("        Use --list_studies to see available study IDs")
        sys.exit(1)

    if args.dicom_dir:
        print("[INFO] DICOM-to-prediction pipeline for new studies is not yet implemented.")
        print("       Use --study_id with an existing training study for demo.")
        print("       Run: python inference_single.py --list_studies")
        sys.exit(0)

    study_id = args.study_id
    print(f"\nSpineSCAN Inference - Study: {study_id}")
    print("=" * 50)

    # Run inference
    predictions = predict_study(study_id, fold=args.fold, device=args.device)

    # Format and display
    results = format_predictions(predictions, study_id)
    print_results_table(results)

    # Save JSON
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {out_path}")

    return results


if __name__ == "__main__":
    main()
