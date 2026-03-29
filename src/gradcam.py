"""
SpineSCAN FYP — Grad-CAM Visualization
========================================
Generates Grad-CAM heatmaps for the classification models using the actual
model weights and preprocessing pipelines from inference_single.py.

Supports both sagittal MIL models (RSNA2ndModel) and axial single-image models.

Usage as CLI:
    python gradcam_viz.py --study_id 13317052 --condition spinal_canal_stenosis --level "L4/L5" --view sagittal

Usage as module (called from app.py):
    from gradcam_viz import run_gradcam_for_study
"""

import os
import sys
import math
import warnings
from pathlib import Path

import numpy as np
import cv2

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BASE_DIR / "spine_model"

LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
LEVEL_TO_NUM = {"L1/L2": 1, "L2/L3": 2, "L3/L4": 3, "L4/L5": 4, "L5/S1": 5}
CONDITIONS = [
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis",
]
SEVERITY = ["Normal/Mild", "Moderate", "Severe"]

# ── Condition → model config routing ─────────────────────────
CONDITION_MODEL_MAP = {
    "spinal_canal_stenosis": {
        "sagittal": "rsna_saggital_mil_spinal_crop_x03_y07",
        "axial":    "rsna_axial_spinal_dis3_crop_x1_y2",
    },
    "left_neural_foraminal_narrowing": {
        "sagittal": "rsna_saggital_mil_nfn_crop_x07_y1_v2",
        "axial":    "rsna_axial_ss_nfn_x2_y2_center_pad0",
    },
    "right_neural_foraminal_narrowing": {
        "sagittal": "rsna_saggital_mil_nfn_crop_x07_y1_v2",
        "axial":    "rsna_axial_ss_nfn_x2_y2_center_pad0",
    },
    "left_subarticular_stenosis": {
        "sagittal": "rsna_saggital_mil_ss_crop_x03_y07_96",
        "axial":    "rsna_axial_ss_nfn_x2_y2_center_pad0",
    },
    "right_subarticular_stenosis": {
        "sagittal": "rsna_saggital_mil_ss_crop_x03_y07_96",
        "axial":    "rsna_axial_ss_nfn_x2_y2_center_pad0",
    },
}

# ── Condition → sagittal CSV mapping ─────────────────────────
SAGITTAL_CSV_MAP = {
    "spinal_canal_stenosis":            "sagittal_spinal_range2_rolling5.csv",
    "left_neural_foraminal_narrowing":  "sagittal_left_nfn_range2_rolling5.csv",
    "right_neural_foraminal_narrowing": "sagittal_right_nfn_range2_rolling5.csv",
    "left_subarticular_stenosis":       "sagittal_left_ss_range2_rolling5.csv",
    "right_subarticular_stenosis":      "sagittal_right_ss_range2_rolling5.csv",
}


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def get_class_index(condition, view_type):
    """Return the target class index for Grad-CAM (targeting Severe)."""
    if view_type == "axial":
        config_name = CONDITION_MODEL_MAP[condition]["axial"]
        if config_name == "rsna_axial_ss_nfn_x2_y2_center_pad0":
            # 6-class model: [nfn_normal, nfn_moderate, nfn_severe, ss_normal, ss_moderate, ss_severe]
            if "subarticular_stenosis" in condition:
                return 5  # ss_severe
            else:
                return 2  # nfn_severe
    # 3-class models: [normal, moderate, severe]
    return 2  # severe


# ── Model loading ────────────────────────────────────────────

def _ensure_repo_imports():
    """Ensure repo is on sys.path and cwd is set for relative paths."""
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    os.chdir(str(REPO_DIR))


def load_model_for_condition(condition, view_type, fold=0, device="cuda"):
    """
    Load the correct model for a condition/view combination.
    Returns (model, cfg, config_name).
    """
    import torch
    import importlib

    _ensure_repo_imports()
    configs_module = importlib.import_module('src.configs')

    config_name = CONDITION_MODEL_MAP[condition][view_type]
    cfg_class = getattr(configs_module, config_name)

    try:
        cfg = cfg_class(fold)
    except TypeError:
        cfg = cfg_class()

    if type(cfg.image_size) == int:
        cfg.image_size = (cfg.image_size, cfg.image_size)
    transform_dict = cfg.transform(cfg.image_size)
    cfg.transform = transform_dict['val']

    ckpt_path = REPO_DIR / "results" / config_name / f"fold_{fold}.ckpt"
    if not ckpt_path.exists():
        ckpt_path = REPO_DIR / "results" / config_name / f"last_fold{fold}.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint for {config_name} at {ckpt_path}")

    state_dict = torch.load(str(ckpt_path), map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

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
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    cfg.model.to(dev)
    cfg.model.eval()
    return cfg.model, cfg, config_name


def find_target_layer(model):
    """Find the target layer for Grad-CAM."""
    # RSNA2ndModel (sagittal MIL) → encoder is the ConvNeXt backbone
    if hasattr(model, 'encoder'):
        enc = model.encoder
        if hasattr(enc, 'stages'):
            return enc.stages[-1]
    # Plain timm ConvNeXt
    if hasattr(model, 'stages'):
        return model.stages[-1]
    # Fallback: last Conv2d
    import torch.nn as nn
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("Could not find a convolutional layer for Grad-CAM")
    return last_conv


def overlay_heatmap(orig_bgr, cam, alpha=0.45):
    """Overlay a Grad-CAM heatmap on the original BGR image."""
    h, w = orig_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    cam_uint8 = (cam_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(orig_bgr, 1 - alpha, heatmap, alpha, 0)


# ── Axial Grad-CAM ──────────────────────────────────────────

def gradcam_axial(study_id, condition, level, fold=0, device="cuda"):
    """
    Generate Grad-CAM for an axial slice.
    Returns dict with 'overlay', 'original', 'logits' keys.
    """
    import torch
    import pandas as pd
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model, cfg, config_name = load_model_for_condition(condition, "axial", fold, device)
    dev = next(model.parameters()).device

    # Load CSV and filter
    csv_path = REPO_DIR / "input" / "axial_classification.csv"
    df = pd.read_csv(csv_path)
    level_num = LEVEL_TO_NUM[level]
    study_df = df[(df.study_id == study_id) & (df.pred_level == level_num)]

    if len(study_df) == 0:
        raise ValueError(f"No axial data for study {study_id} at level {level}")

    # Pick the row closest to center (by dis column if available, else first)
    if 'dis' in study_df.columns:
        row = study_df.iloc[study_df['dis'].abs().argmin()]
    else:
        row = study_df.iloc[0]

    # Load and crop image (same logic as inference_single.run_axial_inference)
    image = cv2.imread(row['path'])
    if image is None:
        raise ValueError(f"Could not read image: {row['path']}")
    image = image[:, :, ::-1]  # BGR → RGB

    if cfg.box_crop:
        box = np.array([row['x_min'], row['y_min'], row['x_max'], row['y_max']]).astype(int)
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
        raise ValueError("Cropped image is empty")

    # Save original for display (before normalization)
    original_bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Apply val transform
    transformed = cfg.transform(image=image.astype(np.uint8))['image']
    tensor = torch.tensor(transformed).float().unsqueeze(0).to(dev)  # (1, C, H, W)

    # Grad-CAM
    target_layer = find_target_layer(model)
    class_idx = get_class_index(condition, "axial")
    targets = [ClassifierOutputTarget(class_idx)]

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=targets)
    cam_map = grayscale_cam[0]

    # Get logits for probability display
    with torch.no_grad():
        logits = model(tensor).cpu().numpy()[0]

    overlay = overlay_heatmap(original_bgr, cam_map)
    label = f"{condition.replace('_', ' ').title()} | {level} | Axial"
    cv2.putText(overlay, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Extract relevant logits for this condition
    if "subarticular_stenosis" in condition:
        cond_logits = logits[3:6] if len(logits) >= 6 else logits[:3]
    elif "neural_foraminal_narrowing" in condition:
        cond_logits = logits[:3]
    else:
        cond_logits = logits[:3]

    del model
    torch.cuda.empty_cache()

    return {
        'overlay': overlay,
        'original': original_bgr,
        'logits': cond_logits,
    }


# ── Sagittal MIL Grad-CAM ───────────────────────────────────

class SagittalSliceWrapper:
    """
    Wraps RSNA2ndModel for single-slice Grad-CAM on MIL models.

    Pre-computes encoder features for all slices, then during forward:
    re-encodes the input slice and substitutes it into the feature array
    so gradients flow through the encoder for Grad-CAM.
    """

    def __init__(self, encoder, head, all_features, center_idx):
        """
        Args:
            encoder: model.encoder (the ConvNeXt backbone)
            head: model.head (AdaptiveConcatPool2d + FC)
            all_features: (N, C, H', W') pre-computed features for all slices
            center_idx: which slice index to re-encode for Grad-CAM
        """
        import torch.nn as nn

        # Build a simple nn.Module so GradCAM can hook into it
        class _Wrapper(nn.Module):
            def __init__(self, encoder, head, all_features, center_idx):
                super().__init__()
                self.encoder = encoder
                self.head = head
                self.register_buffer('_all_features', all_features)
                self.center_idx = center_idx

            def forward(self, x):
                import torch
                # x: (1, 3, H, W) — the center slice
                feat = self.encoder(x)  # (1, C', H', W')

                # Clone pre-computed features and substitute center slice
                all_feat = self._all_features.clone()
                all_feat[self.center_idx] = feat[0]

                # Reshape: (N, C', H', W') → (1, C', N*H', W') — same as RSNA2ndModel.forward
                n, c, h, w = all_feat.shape
                combined = all_feat.unsqueeze(0)           # (1, N, C', H', W')
                combined = combined.permute(0, 2, 1, 3, 4).contiguous()  # (1, C', N, H', W')
                combined = combined.view(1, c, n * h, w)   # (1, C', N*H', W')

                return self.head(combined)

        self.wrapper = _Wrapper(encoder, head, all_features, center_idx)

    def __call__(self, *args, **kwargs):
        return self.wrapper(*args, **kwargs)

    def __getattr__(self, name):
        if name in ('wrapper',):
            return object.__getattribute__(self, name)
        return getattr(self.wrapper, name)


def gradcam_sagittal(study_id, condition, level, fold=0, device="cuda"):
    """
    Generate Grad-CAM for a sagittal MIL model.
    Returns dict with 'overlay', 'original', 'logits', 'slice_idx' keys.
    """
    import torch
    import pandas as pd
    import albumentations as A
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model, cfg, config_name = load_model_for_condition(condition, "sagittal", fold, device)
    dev = next(model.parameters()).device

    def angle_of_line(x1, y1, x2, y2):
        return math.degrees(math.atan2(-(y2 - y1), x2 - x1))

    # Load CSV and filter by study_id + level
    csv_name = SAGITTAL_CSV_MAP[condition]
    csv_path = REPO_DIR / "input" / csv_name
    df = pd.read_csv(csv_path)
    study_df = df[(df.study_id == study_id) & (df.level == level)]

    if len(study_df) == 0:
        raise ValueError(f"No sagittal data for study {study_id} at level {level} in {csv_name}")

    row = study_df.iloc[0]
    path = row['path']
    paths = row['paths'].split(',')
    l_point = (row['l_x'], row['l_y'])
    r_point = (row['r_x'], row['r_y'])

    origin_img = cv2.imread(path)
    if origin_img is None:
        raise ValueError(f"Could not read image: {path}")
    origin_size = origin_img.shape[:2]

    # Preprocess all slices (same as inference_single.run_sagittal_mil_inference)
    images = []
    display_images = []  # Pre-transform images for display
    for p in paths:
        if p == 'nan' or not os.path.exists(p):
            image = np.zeros((origin_size[0], origin_size[1], 3), dtype=np.uint8)
        else:
            image = cv2.imread(p)
            if image is None:
                image = np.zeros((origin_size[0], origin_size[1], 3), dtype=np.uint8)
            else:
                image = cv2.resize(image, (origin_size[1], origin_size[0]))

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
            x1 = int(min(a[0], b[0]))
            y1 = int(min(a[1], b[1]) - image.shape[0] * 0.1)
            x2 = int(max(a[0], b[0]))
            y2 = int(max(a[1], b[1]) + image.shape[0] * 0.1)
            y1, y2 = max(0, y1), min(image.shape[0], y2)
            x1, x2 = max(0, x1), min(image.shape[1], x2)
            image = image[y1:y2, x1:x2]

        if image.size == 0:
            image = np.zeros((cfg.image_size[0], cfg.image_size[1], 3), dtype=np.uint8)

        display_images.append(image.copy())
        image = cfg.transform(image=image.astype(np.uint8))['image']
        images.append(image)

    # Stack and create tensor
    images_np = np.stack(images, 0)  # (N, C, H, W)
    images_tensor = torch.tensor(images_np).float().to(dev)

    # Choose center slice for Grad-CAM
    center_idx = len(images) // 2

    # Pre-compute encoder features for all slices
    with torch.no_grad():
        all_features = model.encoder(images_tensor)  # (N, C', H', W')

    # Build wrapper
    wrapper = SagittalSliceWrapper(model.encoder, model.head, all_features, center_idx)

    # The center slice input
    center_tensor = images_tensor[center_idx:center_idx + 1]  # (1, C, H, W)

    # Grad-CAM on wrapper
    target_layer = find_target_layer(wrapper)
    class_idx = get_class_index(condition, "sagittal")
    targets = [ClassifierOutputTarget(class_idx)]

    with GradCAM(model=wrapper.wrapper, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=center_tensor, targets=targets)
    cam_map = grayscale_cam[0]

    # Get logits for probability display
    with torch.no_grad():
        full_tensor = torch.tensor(images_np).float().unsqueeze(0).to(dev)  # (1, N, C, H, W)
        logits = model(full_tensor).cpu().numpy()[0]

    # Create overlay on the display image
    display_img = display_images[center_idx]
    original_bgr = display_img if display_img.shape[2] == 3 else cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    overlay = overlay_heatmap(original_bgr, cam_map)

    label = f"{condition.replace('_', ' ').title()} | {level} | Sagittal"
    cv2.putText(overlay, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    del model, wrapper
    torch.cuda.empty_cache()

    return {
        'overlay': overlay,
        'original': original_bgr,
        'logits': logits[:3],
        'slice_idx': center_idx,
    }


# ── Annotated Spine Report ───────────────────────────────────

def generate_spine_report(study_id, predictions):
    """
    Generate an annotated full sagittal MRI with severity markers at each disc level.

    Args:
        study_id: int, the study ID
        predictions: dict from format_predictions() with structure:
            {'study_id': ..., 'conditions': {cond_key: {level: {predicted_class, ...}}}}

    Returns:
        dict with 'annotated_image' (BGR), 'original_image' (BGR),
        'level_details' (per-level info), or None if data unavailable.
    """
    import pandas as pd

    coords_csv = REPO_DIR / "input" / "train_label_coordinates.csv"
    if not coords_csv.exists():
        return None

    df = pd.read_csv(coords_csv)
    scs = df[(df.study_id == study_id) & (df.condition == "Spinal Canal Stenosis")]
    if len(scs) == 0:
        return None

    # Identify mid-sagittal slice (mode of series/instance across the 5 levels)
    series_id = int(scs['series_id'].mode().iloc[0])
    instance_number = int(scs['instance_number'].mode().iloc[0])

    # Try both naming conventions produced by dcm_to_png.py
    img_dir = REPO_DIR / "input" / "sagittal_all_images"
    img_path = img_dir / f"{study_id}___{series_id}___{instance_number}.png"
    if not img_path.exists():
        img_path = img_dir / f"{study_id}___{instance_number}.png"
    if not img_path.exists():
        return None

    original = cv2.imread(str(img_path))
    if original is None:
        return None

    # Collect (x, y) per level from the CSV
    level_coords = {}
    for _, row in scs.iterrows():
        level_coords[row['level']] = (float(row['x']), float(row['y']))

    # Determine worst severity per level across all 5 conditions
    severity_rank = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    conditions_data = predictions.get("conditions", {})

    level_details = {}
    for level in LEVELS:
        worst_sev = "Normal/Mild"
        worst_cond = ""
        worst_rank = -1

        for cond_key in CONDITIONS:
            level_data = conditions_data.get(cond_key, {}).get(level, {})
            pred_class = level_data.get("predicted_class", "N/A")
            rank = severity_rank.get(pred_class, -1)
            if rank > worst_rank:
                worst_rank = rank
                worst_sev = pred_class
                worst_cond = cond_key

        xy = level_coords.get(level)
        level_details[level] = {
            'severity': worst_sev,
            'worst_condition': worst_cond.replace('_', ' ').title() if worst_cond else '',
            'x': xy[0] if xy else None,
            'y': xy[1] if xy else None,
        }

    # --- Draw the annotated image ---
    h, w = original.shape[:2]
    target_w = 800
    scale = target_w / w
    new_h = int(h * scale)
    margin_right = 350          # space for level labels
    margin_bottom = 80          # space for title + legend

    canvas = np.zeros((new_h + margin_bottom, target_w + margin_right, 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)    # dark-grey background

    img_scaled = cv2.resize(original, (target_w, new_h))
    canvas[0:new_h, 0:target_w] = img_scaled

    # Title
    cv2.putText(canvas, f"Study {study_id} - Spine Overview",
                (10, new_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)

    sev_colors_bgr = {
        "Normal/Mild": (69, 167, 40),    # green
        "Moderate":    (20, 126, 253),    # orange
        "Severe":      (53, 53, 220),     # red
    }
    label_x = target_w + 20

    for level in LEVELS:
        det = level_details[level]
        if det['x'] is None or det['y'] is None:
            continue

        cx = int(det['x'] * scale)
        cy = int(det['y'] * scale)
        sev = det['severity']
        color = sev_colors_bgr.get(sev, (125, 117, 108))

        # Circle with white outline
        cv2.circle(canvas, (cx, cy), 14, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 12, color, -1, cv2.LINE_AA)

        # Connector line to label column
        cv2.line(canvas, (cx + 14, cy), (label_x - 5, cy), color, 1, cv2.LINE_AA)

        # Level + severity text
        cv2.putText(canvas, f"{level} - {sev}", (label_x, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Condition name (only for Moderate / Severe)
        if sev in ("Moderate", "Severe") and det['worst_condition']:
            cv2.putText(canvas, det['worst_condition'], (label_x, cy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    # Legend bar
    legend_y = new_h + 60
    lx = 10
    for label, color in [("Normal/Mild", sev_colors_bgr["Normal/Mild"]),
                          ("Moderate",    sev_colors_bgr["Moderate"]),
                          ("Severe",      sev_colors_bgr["Severe"])]:
        cv2.circle(canvas, (lx + 8, legend_y), 6, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (lx + 20, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        lx += 140

    return {
        'annotated_image': canvas,
        'original_image': original,
        'level_details': level_details,
    }


# ── Top-level API ────────────────────────────────────────────

def run_gradcam_for_study(study_id, condition, level, view_type="sagittal",
                          fold=0, device="cuda"):
    """
    Generate Grad-CAM for a study/condition/level/view combination.

    Args:
        study_id: int, the study ID from training data
        condition: str, one of CONDITIONS
        level: str, e.g. "L4/L5"
        view_type: "sagittal" or "axial"
        fold: int, model fold (default 0)
        device: "cuda" or "cpu"

    Returns:
        dict with keys: overlay (BGR ndarray), original (BGR ndarray),
                        logits (numpy array of 3 values)
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}. Must be one of {CONDITIONS}")
    if level not in LEVELS:
        raise ValueError(f"Unknown level: {level}. Must be one of {LEVELS}")
    if view_type not in ("sagittal", "axial"):
        raise ValueError(f"view_type must be 'sagittal' or 'axial'")

    if view_type == "axial":
        return gradcam_axial(study_id, condition, level, fold, device)
    else:
        return gradcam_sagittal(study_id, condition, level, fold, device)


# ── CLI ──────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="SpineSCAN Grad-CAM visualization")
    p.add_argument("--study_id", "-s", type=int, required=True,
                   help="Study ID from training set")
    p.add_argument("--condition", "-c", default="spinal_canal_stenosis",
                   choices=CONDITIONS, help="Condition to visualize")
    p.add_argument("--level", "-l", default="L4/L5",
                   choices=LEVELS, help="Disc level")
    p.add_argument("--view", "-v", default="sagittal",
                   choices=["sagittal", "axial"], help="View type")
    p.add_argument("--output", "-o", default="gradcam_out.png",
                   help="Output image path")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"\nSpineSCAN Grad-CAM")
    print(f"  Study:     {args.study_id}")
    print(f"  Condition: {args.condition}")
    print(f"  Level:     {args.level}")
    print(f"  View:      {args.view}")
    print()

    try:
        result = run_gradcam_for_study(
            study_id=args.study_id,
            condition=args.condition,
            level=args.level,
            view_type=args.view,
            fold=args.fold,
            device=args.device,
        )
        cv2.imwrite(args.output, result['overlay'])
        probs = softmax(result['logits'].reshape(1, -1))[0]
        print(f"  Probabilities: Normal/Mild={probs[0]:.1%}  Moderate={probs[1]:.1%}  Severe={probs[2]:.1%}")
        print(f"  Saved: {args.output}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("  Install: pip install grad-cam")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
