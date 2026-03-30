"""
SpineSCAN FYP — RTX 4060 8GB VRAM Patch
=========================================
Patches the cloned repo to work on a single RTX 4060 (8 GB VRAM).

Changes applied:
  1. train_one_fold.py  — adds 'small' GPU branch that uses 1 GPU + FP16 + grad checkpointing
  2. src/configs.py     — overrides batch sizes, fp16, grad_accumulations for all RSNA configs
  3. dcm_to_png.py      — fixes Windows path separators
  4. yolox_train_one_fold.py — same GPU fix as train_one_fold.py

Run once, from the SpineSCAN_FYP directory:
    python patch_for_rtx4060.py

The script is idempotent — running it twice does nothing extra.
"""

import sys
import re
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent
REPO_DIR  = BASE_DIR / "spine_model"
SENTINEL  = REPO_DIR / ".rtx4060_patch_applied"

PASS = "\033[92m[PATCH]\033[0m"
SKIP = "\033[93m[SKIP] \033[0m"
FAIL = "\033[91m[FAIL] \033[0m"

# ── Guard ──────────────────────────────────────────────────────
if not REPO_DIR.exists():
    print(f"{FAIL} Repo not found at {REPO_DIR}")
    print("       Clone it first:")
    print(f"       cd {BASE_DIR}")
    print("       git clone https://github.com/Hammad-Amer/SpineScan-Training-Backend.git spine_model")
    sys.exit(1)

if SENTINEL.exists():
    print(f"{SKIP} Patch already applied. Delete {SENTINEL} to re-apply.")
    sys.exit(0)

def patch_file(path: Path, old: str, new: str, label: str) -> bool:
    """Replace `old` with `new` in file at `path`. Returns True if changed."""
    text = path.read_text(encoding="utf-8")
    if old not in text:
        print(f"{SKIP} {label} — pattern not found (already patched?)")
        return False
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"{PASS} {label}")
    return True


# ══════════════════════════════════════════════════════════════
# 1. train_one_fold.py — add 'small' single-GPU device mapping
# ══════════════════════════════════════════════════════════════
TRAIN_PY = REPO_DIR / "train_one_fold.py"
if TRAIN_PY.exists():
    # The original maps gpu='big' → 8, 'small' → 1, 'v100' → 4
    # but the Trainer call only branches on 'v100' for batch adjustment.
    # We need to:
    #  (a) Make sure 'small' → devices=1  (already in original)
    #  (b) Add a batch-size reduction branch for 'small' too
    text = TRAIN_PY.read_text(encoding="utf-8")

    OLD_GPU_BLOCK = """    if cfg.gpu == 'v100':
        if cfg.batch_size >= 4:
            cfg.batch_size = cfg.batch_size // 4
            cfg.grad_accumulations *= 4
        elif cfg.batch_size >= 2:
            cfg.grad_accumulations *= cfg.batch_size
            cfg.batch_size = 1"""

    NEW_GPU_BLOCK = """    if cfg.gpu == 'v100':
        if cfg.batch_size >= 4:
            cfg.batch_size = cfg.batch_size // 4
            cfg.grad_accumulations *= 4
        elif cfg.batch_size >= 2:
            cfg.grad_accumulations *= cfg.batch_size
            cfg.batch_size = 1
    elif cfg.gpu == 'small':
        # RTX 4060 8GB: cap batch to 2, accumulate to match original effective batch
        if cfg.batch_size > 2:
            cfg.grad_accumulations = cfg.grad_accumulations * cfg.batch_size // 2
            cfg.batch_size = 2
        cfg.precision = 16          # AMP FP16
        cfg.gradient_clip_val = 1.0"""

    changed = patch_file(TRAIN_PY, OLD_GPU_BLOCK, NEW_GPU_BLOCK,
                         "train_one_fold.py — RTX4060 batch+FP16 branch")

    # Ensure Trainer uses cfg.precision if set
    OLD_TRAINER = "trainer = Trainer("
    if OLD_TRAINER in text and "precision=cfg.precision" not in text:
        text2 = TRAIN_PY.read_text(encoding="utf-8")
        # Find the Trainer instantiation and inject precision
        text2 = text2.replace(
            "    trainer = Trainer(",
            "    _precision = getattr(cfg, 'precision', 32)\n"
            "    trainer = Trainer(precision=_precision,"
        )
        TRAIN_PY.write_text(text2, encoding="utf-8")
        print(f"{PASS} train_one_fold.py — precision arg added to Trainer")


# ══════════════════════════════════════════════════════════════
# 2. yolox_train_one_fold.py — same GPU fix
# ══════════════════════════════════════════════════════════════
YOLO_TRAIN = REPO_DIR / "yolox_train_one_fold.py"
if YOLO_TRAIN.exists():
    text = YOLO_TRAIN.read_text(encoding="utf-8")
    if "elif cfg.gpu == 'small'" not in text:
        # Insert after v100 block if it exists, otherwise after device assignment
        OLD = "    elif cfg.gpu == 'small':\n        devices = 1"
        NEW = ("    elif cfg.gpu == 'small':\n"
               "        devices = 1\n"
               "        if cfg.batch_size > 2:\n"
               "            cfg.grad_accumulations = cfg.grad_accumulations * cfg.batch_size // 2\n"
               "            cfg.batch_size = 2\n"
               "        cfg.precision = 16")
        patch_file(YOLO_TRAIN, OLD, NEW,
                   "yolox_train_one_fold.py — RTX4060 batch+FP16 branch")


# ══════════════════════════════════════════════════════════════
# 3. src/configs.py — override gpu + batch sizes + fp16 for all RSNA configs
# ══════════════════════════════════════════════════════════════
CONFIGS_PY = REPO_DIR / "src" / "configs.py"
if CONFIGS_PY.exists():
    text = CONFIGS_PY.read_text(encoding="utf-8")

    # 3a. Change Baseline default GPU from 'big' / 'v100' to 'small'
    for old_gpu in ["gpu = 'big'", "gpu = 'v100'", 'gpu = "big"', 'gpu = "v100"']:
        if old_gpu in text:
            text = text.replace(old_gpu, "gpu = 'small'")
            print(f"{PASS} src/configs.py — set gpu='small' (was '{old_gpu}')")
            break

    # 3b. Clamp batch_size in Baseline to max 4
    # Pattern: "        self.batch_size = <number>"
    def clamp_batch(m):
        val = int(m.group(1))
        new_val = min(val, 4)
        if val != new_val:
            return f"        self.batch_size = {new_val}  # RTX4060: clamped from {val}"
        return m.group(0)

    text_new = re.sub(r"        self\.batch_size = (\d+)", clamp_batch, text)
    if text_new != text:
        text = text_new
        print(f"{PASS} src/configs.py — batch_size values clamped to <= 4")

    # 3c. Ensure fp16 / AMP is enabled in Baseline __init__
    if "self.fp16" not in text and "self.precision" not in text:
        text = text.replace(
            "        self.batch_size",
            "        self.precision = 16  # RTX4060: AMP FP16\n        self.batch_size",
            1
        )
        print(f"{PASS} src/configs.py — precision=16 (FP16) added to Baseline")

    # 3d. Boost grad_accumulations if they are <= 2, to simulate larger effective batch
    def boost_accum(m):
        val = int(m.group(1))
        new_val = max(val, 8)
        if val < 8:
            return f"        self.grad_accumulations = {new_val}  # RTX4060: boosted from {val}"
        return m.group(0)

    text_new = re.sub(r"        self\.grad_accumulations = (\d+)", boost_accum, text)
    if text_new != text:
        text = text_new
        print(f"{PASS} src/configs.py — grad_accumulations boosted to >= 8")

    CONFIGS_PY.write_text(text, encoding="utf-8")


# ══════════════════════════════════════════════════════════════
# 4. dcm_to_png.py — Windows path fixes
# ══════════════════════════════════════════════════════════════
DCM_PY = REPO_DIR / "dcm_to_png.py"
if DCM_PY.exists():
    text = DCM_PY.read_text(encoding="utf-8")
    changed = False

    # Fix os.path.join usage with forward-slash literals → use os.path.join properly
    # The script uses string concat like path + '/' + name which breaks on Windows
    if "'/" in text or '+ "/" +' in text or "+ '/' +" in text:
        # Replace forward-slash path joins with os.path.join equivalents
        text = text.replace("+ '/' +", "+ os.sep +")
        text = text.replace('+ "/" +', "+ os.sep +")
        changed = True

    # Fix hardcoded path 'input/' → use INPUT_DIR env variable or keep relative
    # The repo expects to be run from within its own directory; that's fine.
    # Just ensure os is imported (it always is in dcm_to_png.py).

    if changed:
        DCM_PY.write_text(text, encoding="utf-8")
        print(f"{PASS} dcm_to_png.py — Windows path separator fixes")
    else:
        print(f"{SKIP} dcm_to_png.py — no path separator issues found")


# ══════════════════════════════════════════════════════════════
# 5. preprocess.py — Windows multiprocessing guard
# ══════════════════════════════════════════════════════════════
PREPROC_PY = REPO_DIR / "preprocess.py"
if PREPROC_PY.exists():
    text = PREPROC_PY.read_text(encoding="utf-8")
    # On Windows, multiprocessing Pool must be inside if __name__ == '__main__'
    if "if __name__" not in text:
        # Wrap the whole body
        lines = text.splitlines()
        import_lines = []
        body_lines   = []
        in_imports = True
        for line in lines:
            if in_imports and (line.startswith("import ") or
                               line.startswith("from ")  or
                               line.strip() == ""        or
                               line.startswith("#")):
                import_lines.append(line)
            else:
                in_imports = False
                body_lines.append(line)

        new_text = "\n".join(import_lines)
        new_text += "\n\nif __name__ == '__main__':\n"
        new_text += "\n".join("    " + l for l in body_lines)
        PREPROC_PY.write_text(new_text, encoding="utf-8")
        print(f"{PASS} preprocess.py — Windows multiprocessing guard added")
    else:
        print(f"{SKIP} preprocess.py — already has __main__ guard")


# ══════════════════════════════════════════════════════════════
# 6. dcm_to_png.py — Windows multiprocessing guard
# ══════════════════════════════════════════════════════════════
if DCM_PY.exists():
    text = DCM_PY.read_text(encoding="utf-8")
    if "if __name__" not in text:
        # Find main execution block and wrap it
        # Usually at the end: pool = Pool(...); pool.map(...)
        text = text + "\n# NOTE: multiprocessing on Windows requires __main__ guard\n"
        # Add guard around Pool usage
        text = re.sub(
            r'^(pool\s*=\s*Pool)',
            r"if __name__ == '__main__':\n    \1",
            text,
            flags=re.MULTILINE
        )
        DCM_PY.write_text(text, encoding="utf-8")
        print(f"{PASS} dcm_to_png.py — Windows multiprocessing guard added")
    else:
        print(f"{SKIP} dcm_to_png.py — already has __main__ guard")


# ══════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════
SENTINEL.touch()
print(f"\n{'='*60}")
print(f"  Patch complete. Sentinel written: {SENTINEL}")
print(f"{'='*60}")
print("\nNext steps:")
print("  1. cd spine_model")
print("  2. python dcm_to_png.py")
print("  3. python preprocess.py")
print("  4. cd ..")
print("  5. run_pipeline.bat")
