"""
SpineSCAN FYP — Local Noise Detection (no external team predictions)
=====================================================================
Adapted from the original find_noisy_label.py which required OOF predictions
from Ian's and Bartley's models. This version uses ONLY our own OOF predictions
from all trained crop variants.

Outputs:
    results/noisy_target_level_th08.csv   — threshold 0.8
    results/noisy_target_level_th09.csv   — threshold 0.9
    results/noisy_target_level_1016.csv   — copy of th08 (some configs reference this name)
    results/oof_ensemble.csv              — per-sample ensemble predictions + losses

Run from: SpineSCAN_FYP/ directory  (it will cd into the repo)
Prereqs:  All 14 classification models trained with predict.py (OOF CSVs exist)
          At minimum, the original 5 base models must have OOF CSVs.
"""

import os
import sys
import copy
import shutil
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit

warnings.simplefilter('ignore')
pd.set_option('display.max_columns', 500)

# ── Setup paths ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BASE_DIR / "spine_model"
os.chdir(str(REPO_DIR))

FOLD = 0

# ── Column definitions ───────────────────────────────────────────
label_features = [
    'spinal_canal_stenosis',
    'left_neural_foraminal_narrowing',
    'right_neural_foraminal_narrowing',
    'left_subarticular_stenosis',
    'right_subarticular_stenosis',
]
true_cols = []
pred_cols = []
for col in label_features:
    for level in ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']:
        for c in ['normal', 'moderate', 'severe']:
            true_cols.append(f'{col}_{level}_{c}')
            pred_cols.append(f'pred_{col}_{level}_{c}')


def load_oof_if_exists(config_name, fold=FOLD):
    """Load OOF CSV for a config, return None if not found."""
    path = Path(f'results/{config_name}/oof_fold{fold}.csv')
    if path.exists():
        df = pd.read_csv(path)
        print(f"  [OK] {config_name}: {len(df)} rows")
        return df
    else:
        print(f"  [--] {config_name}: not found, skipping")
        return None


def ensemble_axial_spinal():
    """Ensemble axial spinal configs → aggregated predictions per (study_id, pred_level)."""
    configs = [
        'rsna_axial_spinal_dis3_crop_x1_y2',
        'rsna_axial_spinal_dis3_crop_x05_y6',
    ]
    config_pred_cols = [
        'pred_spinal_canal_stenosis_normal',
        'pred_spinal_canal_stenosis_moderate',
        'pred_spinal_canal_stenosis_severe',
    ]
    config_cols = [c.replace('pred_', '') for c in config_pred_cols]

    # Load first available config for ground truth
    base_oof = None
    for config in configs:
        oof = load_oof_if_exists(config)
        if oof is not None and base_oof is None:
            base_oof = oof.copy()

    if base_oof is None:
        print("  [WARN] No axial spinal OOFs found")
        return None

    true = base_oof.groupby(['study_id', 'pred_level'])[config_cols + config_pred_cols].mean().reset_index().sort_values(['study_id', 'pred_level'])

    # Ensemble predictions
    dfs = []
    for config in configs:
        oof = load_oof_if_exists(config)
        if oof is not None:
            oof = oof.groupby(['study_id', 'pred_level'])[config_pred_cols].mean().reset_index().sort_values(['study_id', 'pred_level'])
            dfs.append(oof)

    if len(dfs) == 0:
        return None

    merged = pd.concat(dfs)
    merged = merged.groupby(['study_id', 'pred_level'])[config_pred_cols].mean().reset_index()
    merged[config_cols] = true[config_cols].values
    merged[config_cols] = merged[config_cols].astype(int)
    merged[['normal', 'moderate', 'severe']] = merged[config_cols].values
    merged[['pred_normal', 'pred_moderate', 'pred_severe']] = merged[config_pred_cols].values
    return merged


def ensemble_axial_nfn_ss(target_type):
    """Ensemble axial NFN or SS configs.
    target_type: 'nfn' or 'ss'
    """
    configs = [
        'rsna_axial_ss_nfn_x2_y2_center_pad0',
        'rsna_axial_ss_nfn_x2_y6_center_pad0',
        'rsna_axial_ss_nfn_x2_y8_center_pad10',
    ]
    axial_dis_th = 5

    if target_type == 'nfn':
        cols = [
            'neural_foraminal_narrowing_normal',
            'neural_foraminal_narrowing_moderate',
            'neural_foraminal_narrowing_severe',
        ]
    else:
        cols = [
            'subarticular_stenosis_normal',
            'subarticular_stenosis_moderate',
            'subarticular_stenosis_severe',
        ]
    config_pred_cols = ['pred_' + c for c in cols]
    config_cols = cols

    preds_list = []
    last_oof = None
    for config in configs:
        oof = load_oof_if_exists(config)
        if oof is not None:
            if 'dis' in oof.columns:
                oof.loc[oof.dis.isnull(), 'dis'] = oof.dis.mean()
                oof = oof[oof.dis < axial_dis_th]
            last_oof = oof.copy()
            preds_list.append(oof[config_pred_cols].values)

    if last_oof is None or len(preds_list) == 0:
        print(f"  [WARN] No axial {target_type} OOFs found")
        return None

    last_oof[config_pred_cols] = np.mean(preds_list, 0)
    last_oof[['normal', 'moderate', 'severe']] = last_oof[config_cols].values
    last_oof[['pred_normal', 'pred_moderate', 'pred_severe']] = last_oof[config_pred_cols].values
    return last_oof


def ensemble_sagittal_spinal():
    """Ensemble sagittal spinal configs."""
    configs = [
        'rsna_saggital_mil_spinal_crop_x03_y07',
        'rsna_saggital_mil_spinal_crop_x03_y05',
    ]
    config_cols = [
        'spinal_canal_stenosis_normal',
        'spinal_canal_stenosis_moderate',
        'spinal_canal_stenosis_severe',
    ]
    config_pred_cols = ['pred_' + c for c in config_cols]

    preds_list = []
    last_oof = None
    for config in configs:
        oof = load_oof_if_exists(config)
        if oof is not None:
            oof['pred_level'] = oof.level.map({
                'L1/L2': 1, 'L2/L3': 2, 'L3/L4': 3, 'L4/L5': 4, 'L5/S1': 5,
            })
            last_oof = oof.copy()
            preds_list.append(oof[config_pred_cols].values)

    if last_oof is None or len(preds_list) == 0:
        print("  [WARN] No sagittal spinal OOFs found")
        return None

    last_oof[config_pred_cols] = np.mean(preds_list, 0)
    last_oof[['normal', 'moderate', 'severe']] = last_oof[config_cols].values
    last_oof[['pred_normal', 'pred_moderate', 'pred_severe']] = last_oof[config_pred_cols].values
    return last_oof


def ensemble_sagittal_nfn():
    """Ensemble sagittal NFN configs."""
    configs = [
        'rsna_saggital_mil_nfn_crop_x07_y1_v2',
        'rsna_saggital_mil_nfn_crop_x15_y1_v2',
        'rsna_saggital_mil_nfn_crop_x03_y1_v2',
    ]
    config_cols = [
        'neural_foraminal_narrowing_normal',
        'neural_foraminal_narrowing_moderate',
        'neural_foraminal_narrowing_severe',
    ]
    config_pred_cols = ['pred_' + c for c in config_cols]

    preds_list = []
    last_oof = None
    for config in configs:
        oof = load_oof_if_exists(config)
        if oof is not None:
            oof['pred_level'] = oof.level.map({
                'L1/L2': 1, 'L2/L3': 2, 'L3/L4': 3, 'L4/L5': 4, 'L5/S1': 5,
            })
            if 'left_nfn' in list(oof):
                oof['left_right'] = 'right'
                oof.loc[oof.left_nfn == 1, 'left_right'] = 'left'
            last_oof = oof.copy()
            preds_list.append(oof[config_pred_cols].values)

    if last_oof is None or len(preds_list) == 0:
        print("  [WARN] No sagittal NFN OOFs found")
        return None

    last_oof[config_pred_cols] = np.mean(preds_list, 0)
    last_oof[['pred_normal', 'pred_moderate', 'pred_severe']] = last_oof[config_pred_cols].values
    last_oof[['normal', 'moderate', 'severe']] = last_oof[config_cols].values
    return last_oof


def ensemble_sagittal_ss():
    """Ensemble sagittal subarticular stenosis configs."""
    configs = [
        'rsna_saggital_mil_ss_crop_x03_y07_96',
        'rsna_saggital_mil_ss_crop_x03_y05_96',
        'rsna_saggital_mil_ss_crop_x03_y2_96',
        'rsna_saggital_mil_ss_crop_x1_y07_96',
    ]
    config_cols = [
        'subarticular_stenosis_normal',
        'subarticular_stenosis_moderate',
        'subarticular_stenosis_severe',
    ]
    config_pred_cols = ['pred_' + c for c in config_cols]

    preds_list = []
    last_oof = None
    for config in configs:
        oof = load_oof_if_exists(config)
        if oof is not None:
            if 'left_nfn' in list(oof):
                oof['left_right'] = 'right'
                oof.loc[oof.left_nfn == 1, 'left_right'] = 'left'
            oof = oof.groupby(['study_id', 'level', 'left_right'])[config_cols + config_pred_cols].mean().reset_index()
            oof['pred_level'] = oof.level.map({
                'L1/L2': 1, 'L2/L3': 2, 'L3/L4': 3, 'L4/L5': 4, 'L5/S1': 5,
            })
            last_oof = oof.copy()
            preds_list.append(oof[config_pred_cols].values)

    if last_oof is None or len(preds_list) == 0:
        print("  [WARN] No sagittal SS OOFs found")
        return None

    last_oof[config_pred_cols] = np.mean(preds_list, 0)
    last_oof[['normal', 'moderate', 'severe']] = last_oof[config_cols].values
    last_oof[['pred_normal', 'pred_moderate', 'pred_severe']] = last_oof[config_pred_cols].values
    return last_oof


def normalize_probabilities_to_one(tensor):
    """Normalize rows to sum to 1."""
    import torch
    row_totals = tensor.sum(dim=1, keepdim=True)
    row_totals = torch.where(row_totals == 0, torch.ones_like(row_totals), row_totals)
    return tensor / row_totals


def build_noise_csv(oof_ensemble_path, threshold):
    """Read oof_ensemble.csv and flag noisy labels at the given threshold."""
    oof = pd.read_csv(oof_ensemble_path)
    targets = [
        'spinal_canal_stenosis',
        'left_neural_foraminal_narrowing',
        'right_neural_foraminal_narrowing',
        'left_subarticular_stenosis',
        'right_subarticular_stenosis',
    ]
    levels_list = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']

    study_ids = []
    target_names = []
    level_names = []

    for target in targets:
        for level in levels_list:
            dfs = []
            for claz in ['normal', 'moderate', 'severe']:
                col = f'{target}_{level}_{claz}_loss'
                if col in oof.columns:
                    dfs.append(oof[oof[col] > threshold])
            if dfs:
                noisy_df = pd.concat(dfs)
                for sid in noisy_df.study_id.unique():
                    study_ids.append(sid)
                    target_names.append(target)
                    level_names.append(level)

    noise_df = pd.DataFrame({
        'study_id': study_ids,
        'target': target_names,
        'level': level_names,
    }).sort_values(['target', 'study_id', 'level']).drop_duplicates()
    noise_df['study_level'] = noise_df.study_id.astype(str) + '_' + noise_df.level
    return noise_df


if __name__ == '__main__':
    import torch

    print("=" * 60)
    print("  SpineSCAN — Local Noise Detection")
    print("=" * 60)

    # ── Load all OOF ensembles ────────────────────────────────
    print("\n[1/6] Axial Spinal...")
    axial_spinal = ensemble_axial_spinal()

    print("\n[2/6] Axial NFN...")
    axial_nfn = ensemble_axial_nfn_ss('nfn')

    print("\n[3/6] Axial SS...")
    axial_ss = ensemble_axial_nfn_ss('ss')

    print("\n[4/6] Sagittal Spinal...")
    sagittal_spinal = ensemble_sagittal_spinal()

    print("\n[5/6] Sagittal NFN...")
    sagittal_nfn = ensemble_sagittal_nfn()

    print("\n[6/6] Sagittal SS...")
    sagittal_ss = ensemble_sagittal_ss()

    # ── Gather all per-(study, level, condition) predictions ──
    print("\nBuilding unified prediction table...")
    study_ids = []
    targets = []
    levels = []
    is_axials = []
    lrs = []
    trues = []
    preds_all = []

    def gather(df, target_name, is_axial, group_cols, lr_val=None):
        if df is None:
            return
        for keys, idf in df.groupby(group_cols):
            if isinstance(keys, (int, float, np.integer)):
                keys = (keys,)
            trues.append(idf[['normal', 'moderate', 'severe']].mean(0).values)
            preds_all.append(idf[['pred_normal', 'pred_moderate', 'pred_severe']].mean(0).values)
            study_ids.append(keys[0])  # study_id
            targets.append(target_name)
            if 'pred_level' in group_cols:
                idx = group_cols.index('pred_level')
                levels.append(keys[idx])
            else:
                levels.append(keys[1] if len(keys) > 1 else 0)
            is_axials.append(is_axial)
            if 'left_right' in group_cols:
                idx = group_cols.index('left_right')
                lrs.append(keys[idx])
            elif lr_val is not None:
                lrs.append(lr_val)
            else:
                lrs.append('center')

    gather(axial_spinal, 'spinal', 1, ['study_id', 'pred_level'], lr_val='center')
    gather(axial_nfn, 'nfn', 1, ['study_id', 'pred_level', 'left_right'])
    gather(axial_ss, 'ss', 1, ['study_id', 'pred_level', 'left_right'])
    gather(sagittal_spinal, 'spinal', 0, ['study_id', 'pred_level'], lr_val='center')
    gather(sagittal_nfn, 'nfn', 0, ['study_id', 'pred_level', 'left_right'])
    gather(sagittal_ss, 'ss', 0, ['study_id', 'pred_level', 'left_right'])

    oof = pd.DataFrame({
        'study_id': study_ids,
        'target': targets,
        'level': levels,
        'is_axial': is_axials,
        'lr': lrs,
    })
    oof[['normal', 'moderate', 'severe']] = np.array(trues).astype(int)
    oof[['pred_normal', 'pred_moderate', 'pred_severe']] = np.array(preds_all)

    # ── Split axial/sagittal and merge ───────────────────────
    axial = oof[oof.is_axial == 1].copy()
    axial.columns = [
        'study_id', 'target', 'level', 'is_axial', 'lr',
        'normal', 'moderate', 'severe',
        'axial_pred_normal', 'axial_pred_moderate', 'axial_pred_severe',
    ]
    del axial['is_axial']

    sagittal = oof[oof.is_axial == 0].copy()
    sagittal.columns = [
        'study_id', 'target', 'level', 'is_axial', 'lr',
        'normal', 'moderate', 'severe',
        'sagittal_pred_normal', 'sagittal_pred_moderate', 'sagittal_pred_severe',
    ]
    for c in ['is_axial', 'normal', 'moderate', 'severe']:
        del sagittal[c]

    df = axial.merge(sagittal, on=['study_id', 'target', 'level', 'lr'], how='outer')

    # Fill missing modality with other
    for mod_from, mod_to in [('sagittal', 'axial'), ('axial', 'sagittal')]:
        for cond in ['normal', 'moderate', 'severe']:
            col_to = f'{mod_to}_pred_{cond}'
            col_from = f'{mod_from}_pred_{cond}'
            if col_to in df.columns and col_from in df.columns:
                df.loc[df[col_to].isnull(), col_to] = df.loc[df[col_to].isnull(), col_from]

    # Map target names to full condition names
    df.loc[(df.target == 'nfn') & (df.lr == 'left'), 'target'] = 'left_neural_foraminal_narrowing'
    df.loc[(df.target == 'nfn') & (df.lr == 'right'), 'target'] = 'right_neural_foraminal_narrowing'
    df.loc[(df.target == 'ss') & (df.lr == 'left'), 'target'] = 'left_subarticular_stenosis'
    df.loc[(df.target == 'ss') & (df.lr == 'right'), 'target'] = 'right_subarticular_stenosis'
    df.loc[df.target == 'spinal', 'target'] = 'spinal_canal_stenosis'
    df['level'] = df.level.map({1: 'l1_l2', 2: 'l2_l3', 3: 'l3_l4', 4: 'l4_l5', 5: 'l5_s1'})
    df = df.sort_values(['study_id', 'target'])

    # ── Pivot to per-study row format ────────────────────────
    all_targets = [
        'spinal_canal_stenosis',
        'left_neural_foraminal_narrowing', 'right_neural_foraminal_narrowing',
        'left_subarticular_stenosis', 'right_subarticular_stenosis',
    ]
    all_levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']

    m = {'study_id': []}
    meta_cols = []
    for axial_sagittal in ['sagittal', 'axial']:
        for target in all_targets:
            for level in all_levels:
                for condition in ['normal', 'moderate', 'severe']:
                    col = f'{axial_sagittal}_pred_{target}_{level}_{condition}'
                    m[col] = []
                    meta_cols.append(col)

    for sid, idf in df.groupby('study_id'):
        m['study_id'].append(sid)
        for level in all_levels:
            ldf = idf[idf.level == level]
            for target in all_targets:
                tdf = ldf[ldf.target == target]
                for condition in ['normal', 'moderate', 'severe']:
                    for axial_sagittal in ['sagittal', 'axial']:
                        col = f'{axial_sagittal}_pred_{target}_{level}_{condition}'
                        if len(tdf) == 0:
                            m[col].append(0)
                        else:
                            pred_col = f'{axial_sagittal}_pred_{condition}'
                            if pred_col in tdf.columns:
                                m[col].append(tdf[pred_col].values[0])
                            else:
                                m[col].append(0)
    df = pd.DataFrame(m)

    # ── Merge with ground truth ──────────────────────────────
    tr = pd.read_csv('input/train.csv')
    label_feats_25 = []
    gt_cols = []
    for col in label_features:
        for level in all_levels:
            feat = f'{col}_{level}'
            label_feats_25.append(feat)
            for claz_name, claz_col in [('normal', 'Normal/Mild'), ('moderate', 'Moderate'), ('severe', 'Severe')]:
                c = f'{feat}_{claz_name}'
                tr[c] = 0
                tr.loc[tr[feat] == claz_col, c] = 1
                tr.loc[tr[feat].isnull(), c] = np.nan
                gt_cols.append(c)

    tr = tr[['study_id'] + gt_cols]
    for c in gt_cols:
        if c in list(df):
            del df[c]
    oof = tr.merge(df, on='study_id')

    # Null out predictions where ground truth is missing
    for c in gt_cols:
        if c in oof.columns:
            oof.loc[oof[c].isnull(), 'axial_pred_' + c] = np.nan
            oof.loc[oof[c].isnull(), 'sagittal_pred_' + c] = np.nan

    # ── Combine axial + sagittal predictions ─────────────────
    # Use the same weighting as the original: 0.7 axial for spinal, 0.5 for NFN, 0.8 for SS
    ws = [0.7, 0.5, 0.8]
    for condition, w in zip(['spinal', 'neural_foraminal_narrowing', 'subarticular_stenosis'], ws):
        c_cols = [c for c in pred_cols if condition in c]
        for c in c_cols:
            axial_c = 'axial_' + c
            sagittal_c = 'sagittal_' + c
            if axial_c in oof.columns and sagittal_c in oof.columns:
                oof[c] = oof[axial_c] * w + oof[sagittal_c] * (1 - w)
                oof[c] = oof[c].fillna(oof[c].mean())
            elif sagittal_c in oof.columns:
                oof[c] = oof[sagittal_c].fillna(0)
            elif axial_c in oof.columns:
                oof[c] = oof[axial_c].fillna(0)
            else:
                oof[c] = 0

    # ── Apply severity upweighting (same as original) ────────
    preds_tensor = expit(oof[pred_cols].fillna(0).values)
    preds_tensor = torch.FloatTensor(preds_tensor)

    # Spinal (indices 0-4, each with 3 severity classes)
    for i in range(5):
        preds_tensor[:, i * 3 + 1] *= 1.8
        preds_tensor[:, i * 3 + 2] *= 5
        preds_tensor[:, i * 3:(i + 1) * 3] = normalize_probabilities_to_one(preds_tensor[:, i * 3:(i + 1) * 3])
    # NFN (indices 5-14)
    for i in range(5, 15):
        preds_tensor[:, i * 3 + 1] *= 2.2
        preds_tensor[:, i * 3 + 2] *= 5
        preds_tensor[:, i * 3:(i + 1) * 3] = normalize_probabilities_to_one(preds_tensor[:, i * 3:(i + 1) * 3])
    # SS (indices 15-24)
    for i in range(15, 25):
        preds_tensor[:, i * 3 + 1] *= 2.2
        preds_tensor[:, i * 3 + 2] *= 5.5
        preds_tensor[:, i * 3:(i + 1) * 3] = normalize_probabilities_to_one(preds_tensor[:, i * 3:(i + 1) * 3])

    oof[pred_cols] = preds_tensor.numpy()

    # ── Compute per-sample loss ──────────────────────────────
    for c in [c.replace('pred_', '') for c in pred_cols]:
        oof[f'{c}_loss'] = np.abs(oof[c].values - oof['pred_' + c].values)

    # Save ensemble OOF
    os.makedirs('results', exist_ok=True)
    save_cols = ['study_id'] + pred_cols + [c.replace('pred_', '') + '_loss' for c in pred_cols]
    oof[save_cols].to_csv('results/oof_ensemble.csv', index=False)
    print(f"\nSaved: results/oof_ensemble.csv ({len(oof)} studies)")

    # ── Generate noise CSVs ──────────────────────────────────
    for th, fname in [(0.8, 'noisy_target_level_th08.csv'), (0.9, 'noisy_target_level_th09.csv')]:
        noise_df = build_noise_csv('results/oof_ensemble.csv', th)
        noise_df.to_csv(f'results/{fname}', index=False)
        n_noisy = len(noise_df)
        print(f"  th={th}: {n_noisy} noisy (study, target, level) entries → results/{fname}")
        if len(noise_df) > 0:
            print(f"    Per condition: {dict(noise_df.target.value_counts())}")

    # Copy th08 as noisy_target_level_1016.csv (some configs reference this filename)
    shutil.copy2('results/noisy_target_level_th08.csv', 'results/noisy_target_level_1016.csv')
    print(f"  Copied th08 → results/noisy_target_level_1016.csv")

    print("\n" + "=" * 60)
    print("  Noise detection complete!")
    print("  Next: run run_noise_reduction.bat")
    print("=" * 60)
