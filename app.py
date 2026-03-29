"""
SpineSCAN FYP — Streamlit Web Application
==========================================
Lumbar Spine Degenerative Classification
Based on RSNA 2024 2nd-Place Solution (Yuji Ariyasu)

Launch:
    conda activate rsna
    cd C:\\Users\\hamma\\Desktop\\SpineSCAN_FYP
    streamlit run app.py
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import streamlit as st
import cv2

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="SpineSCAN — Lumbar Spine Analysis",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
REPO_DIR = BASE_DIR / "spine_model"

LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

CONDITIONS = {
    "Spinal Canal Stenosis":            "spinal_canal_stenosis",
    "Left Neural Foraminal Narrowing":  "left_neural_foraminal_narrowing",
    "Right Neural Foraminal Narrowing": "right_neural_foraminal_narrowing",
    "Left Subarticular Stenosis":       "left_subarticular_stenosis",
    "Right Subarticular Stenosis":      "right_subarticular_stenosis",
}

SEVERITY_COLORS = {
    "Normal/Mild": "#28a745",   # green
    "Moderate":    "#fd7e14",   # orange
    "Severe":      "#dc3545",   # red
    "N/A":         "#6c757d",   # grey
}

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a3a5c;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .severity-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        color: white;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a3a5c;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 4px;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .stProgress > div > div > div { height: 8px !important; }
    .info-box {
        background: #f0f4ff;
        border-left: 4px solid #4a6fa5;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def severity_badge(label: str) -> str:
    color = SEVERITY_COLORS.get(label, "#6c757d")
    return (f'<span class="severity-badge" '
            f'style="background-color:{color}">{label}</span>')


def prob_to_bar(prob: float, label: str) -> None:
    color = SEVERITY_COLORS.get(label, "#6c757d")
    pct = int(prob * 100)
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0">'
        f'  <span style="width:130px;font-size:0.82rem;color:#444">{label}</span>'
        f'  <div style="flex:1;background:#eee;border-radius:4px;height:10px">'
        f'    <div style="width:{pct}%;background:{color};height:10px;border-radius:4px"></div>'
        f'  </div>'
        f'  <span style="width:40px;text-align:right;font-size:0.82rem">{pct}%</span>'
        f'</div>',
        unsafe_allow_html=True
    )


@st.cache_data(show_spinner=False)
def load_study_ids():
    """Load available study IDs from the preprocessed CSVs."""
    import pandas as pd
    csv_path = REPO_DIR / "input" / "sagittal_spinal_range2_rolling5.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path, usecols=["study_id"])
    return sorted(df.study_id.unique().tolist())


@st.cache_data(show_spinner=False)
def get_study_preview_paths(study_id):
    """Get a few image paths for preview thumbnails."""
    import pandas as pd
    previews = {"sagittal": [], "axial": []}

    # Sagittal
    sag_csv = REPO_DIR / "input" / "sagittal_spinal_range2_rolling5.csv"
    if sag_csv.exists():
        df = pd.read_csv(sag_csv)
        sdf = df[df.study_id == study_id]
        if len(sdf) > 0:
            paths = sdf['path'].unique().tolist()
            # Pick up to 3 evenly spaced
            step = max(1, len(paths) // 3)
            previews["sagittal"] = [paths[i] for i in range(0, len(paths), step)][:3]

    # Axial
    ax_csv = REPO_DIR / "input" / "axial_classification.csv"
    if ax_csv.exists():
        df = pd.read_csv(ax_csv)
        sdf = df[df.study_id == study_id]
        if len(sdf) > 0:
            paths = sdf['path'].unique().tolist()
            step = max(1, len(paths) // 3)
            previews["axial"] = [paths[i] for i in range(0, len(paths), step)][:3]

    return previews


def load_preview_image(img_path):
    """Load a PNG image for preview display. Handles repo-relative paths."""
    full_path = REPO_DIR / img_path if not os.path.isabs(img_path) else Path(img_path)
    img = cv2.imread(str(full_path))
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")
    fold   = st.selectbox("Model fold", [0, 1, 2, 3, 4], index=0,
                          help="Which trained fold to use for inference")
    device = st.radio("Device", ["cuda", "cpu"], index=0)
    st.divider()

    st.markdown("### About")
    st.markdown(
        "**SpineSCAN** uses the RSNA 2024 Lumbar Spine "
        "Degenerative Classification 2nd-place solution "
        "(Yuji Ariyasu) to automatically classify:\n\n"
        "- Spinal Canal Stenosis\n"
        "- Neural Foraminal Narrowing (L/R)\n"
        "- Subarticular Stenosis (L/R)\n\n"
        "across 5 lumbar disc levels (L1-S1).\n\n"
        "Severity grades: **Normal/Mild | Moderate | Severe**"
    )
    st.divider()

    st.markdown("### Model Status")
    ckpt_dir = REPO_DIR / "results"
    if ckpt_dir.exists():
        from glob import glob
        ckpts = glob(str(ckpt_dir / "**" / "*.ckpt"), recursive=True)
        if ckpts:
            st.success(f"{len(ckpts)} checkpoint(s) found")
        else:
            st.error("No checkpoints - train first")
    else:
        st.warning("Repo not found - clone and train")


# ── Main page ─────────────────────────────────────────────────
st.markdown('<div class="main-header">SpineSCAN</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Lumbar Spine Degenerative Classification '
    '- RSNA 2024 2nd-Place Solution</div>',
    unsafe_allow_html=True
)

tabs = st.tabs(["Select Study & Analyze", "Results", "Grad-CAM", "Pipeline Guide"])

# ── TAB 1: Select Study & Analyze ────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-title">Select Study</div>', unsafe_allow_html=True)

    study_ids = load_study_ids()

    if not study_ids:
        st.markdown(
            '<div class="info-box">No preprocessed data found. '
            'Run the training pipeline first (see Pipeline Guide tab).</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(f"**{len(study_ids)}** studies available from fold 0 training data.")

        selected_study = st.selectbox(
            "Study ID",
            study_ids,
            index=min(0, len(study_ids) - 1),
            key="study_select",
        )

        # Preview images
        if selected_study:
            previews = get_study_preview_paths(selected_study)

            st.markdown('<div class="section-title">MRI Preview</div>', unsafe_allow_html=True)
            col_sag, col_ax = st.columns(2)

            with col_sag:
                st.markdown("**Sagittal**")
                if previews["sagittal"]:
                    img_cols = st.columns(len(previews["sagittal"]))
                    for i, p in enumerate(previews["sagittal"]):
                        img = load_preview_image(p)
                        if img is not None:
                            with img_cols[i]:
                                st.image(img, use_container_width=True)
                else:
                    st.caption("No sagittal images found")

            with col_ax:
                st.markdown("**Axial**")
                if previews["axial"]:
                    img_cols = st.columns(len(previews["axial"]))
                    for i, p in enumerate(previews["axial"]):
                        img = load_preview_image(p)
                        if img is not None:
                            with img_cols[i]:
                                st.image(img, use_container_width=True)
                else:
                    st.caption("No axial images found")

            # ── MRI Type Detection ────────────────────────────
            try:
                from src.mri_classifier import classify_mri

                st.markdown('<div class="section-title">MRI Type Detection</div>',
                            unsafe_allow_html=True)

                det_col_sag, det_col_ax = st.columns(2)

                with det_col_sag:
                    if previews["sagittal"]:
                        sag_img_path = REPO_DIR / previews["sagittal"][0]
                        sag_result = classify_mri(str(sag_img_path))
                        sag_cls = sag_result["class"]
                        sag_conf = sag_result["confidence"]
                        sag_color = "#28a745" if sag_result["is_mri"] else "#dc3545"
                        st.markdown(
                            f'**Sagittal:** <span class="severity-badge" '
                            f'style="background-color:{sag_color}">'
                            f'{sag_cls} ({sag_conf:.0%})</span>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.caption("No sagittal image to classify")

                with det_col_ax:
                    if previews["axial"]:
                        ax_img_path = REPO_DIR / previews["axial"][0]
                        ax_result = classify_mri(str(ax_img_path))
                        ax_cls = ax_result["class"]
                        ax_conf = ax_result["confidence"]
                        ax_color = "#28a745" if ax_result["is_mri"] else "#dc3545"
                        st.markdown(
                            f'**Axial:** <span class="severity-badge" '
                            f'style="background-color:{ax_color}">'
                            f'{ax_cls} ({ax_conf:.0%})</span>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.caption("No axial image to classify")

            except (ImportError, FileNotFoundError):
                # Classifier not trained yet — silently skip
                pass
            except Exception:
                pass

            st.divider()
            col_info, col_btn = st.columns([2, 1])
            with col_info:
                st.markdown(f"- Study ID: **{selected_study}**")
                st.markdown(f"- Device: **{device}** | Fold: **{fold}**")
            with col_btn:
                run_clicked = st.button("Run Analysis", type="primary",
                                        use_container_width=True)

            if run_clicked:
                with st.spinner("Running inference (~30s)..."):
                    try:
                        from src.inference import predict_study, format_predictions
                        predictions = predict_study(selected_study, fold=fold, device=device)
                        results = format_predictions(predictions, selected_study)
                        st.session_state["results"] = results
                        st.session_state["study_id"] = selected_study

                        # Generate annotated spine report
                        try:
                            from src.gradcam import generate_spine_report
                            report = generate_spine_report(selected_study, results)
                            st.session_state["spine_report"] = report  # may be None
                        except Exception:
                            st.session_state["spine_report"] = None

                        st.success("Analysis complete! View results in the **Results** tab.")
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())


# ── TAB 2: Results ────────────────────────────────────────────
with tabs[1]:
    if "results" not in st.session_state:
        st.markdown(
            '<div class="info-box">Select a study in the <b>Select Study & Analyze</b> '
            'tab and run analysis to see results here.</div>',
            unsafe_allow_html=True
        )
    else:
        results = st.session_state["results"]
        st.markdown(f'**Study ID:** `{results["study_id"]}`')

        # --- Two-column layout: Spine overview | Summary table ---
        spine_report = st.session_state.get("spine_report")

        if spine_report is not None:
            col_spine, col_table = st.columns([2, 3])
        else:
            col_spine, col_table = None, None

        # Left column: Annotated spine image
        if spine_report is not None:
            with col_spine:
                st.markdown('<div class="section-title">Spine Overview</div>',
                            unsafe_allow_html=True)
                ann_rgb = cv2.cvtColor(spine_report['annotated_image'], cv2.COLOR_BGR2RGB)
                st.image(ann_rgb, use_container_width=True)

                # Download annotated image
                _, img_enc = cv2.imencode(".png", spine_report['annotated_image'])
                st.download_button(
                    label="Download Spine Overview (PNG)",
                    data=img_enc.tobytes(),
                    file_name=f"spine_overview_{results['study_id']}.png",
                    mime="image/png",
                )

        # Right column (or full width if no spine report): Summary table
        table_ctx = col_table if col_table is not None else st  # noqa: F841

        def _render_table_and_probs(ctx):
            ctx.markdown('<div class="section-title">Summary Table</div>',
                         unsafe_allow_html=True)

            import pandas as pd
            rows = []
            for cond_label, cond_key in CONDITIONS.items():
                row = {"Condition": cond_label}
                for level in LEVELS:
                    p = results["conditions"].get(cond_key, {}).get(level, {})
                    pred = p.get("predicted_class", "N/A")
                    row[level] = pred
                rows.append(row)
            df = pd.DataFrame(rows).set_index("Condition")

            def color_severity(val):
                colors = {
                    "Normal/Mild": "background-color: #d4edda; color: #155724",
                    "Moderate":    "background-color: #fff3cd; color: #856404",
                    "Severe":      "background-color: #f8d7da; color: #721c24",
                    "N/A":         "background-color: #e2e3e5; color: #383d41",
                }
                return colors.get(val, "")

            styled = df.style.map(color_severity)
            ctx.dataframe(styled, use_container_width=True)

        if col_table is not None:
            with col_table:
                _render_table_and_probs(st)
        else:
            _render_table_and_probs(st)

        # Detailed probabilities (always full width)
        st.markdown('<div class="section-title">Detailed Probabilities</div>',
                    unsafe_allow_html=True)

        sel_cond = st.selectbox(
            "Select condition",
            list(CONDITIONS.keys()),
            key="results_condition"
        )
        cond_key = CONDITIONS[sel_cond]

        cols = st.columns(5)
        for i, level in enumerate(LEVELS):
            p = results["conditions"].get(cond_key, {}).get(level, {})
            pred = p.get("predicted_class", "N/A")
            conf = p.get("confidence", 0.0)
            with cols[i]:
                st.markdown(f"**{level}**")
                st.markdown(severity_badge(pred), unsafe_allow_html=True)
                st.markdown(f"<small>conf: {conf:.1%}</small>", unsafe_allow_html=True)
                for sev in ["Normal/Mild", "Moderate", "Severe"]:
                    prob_to_bar(p.get(sev, 0.0), sev)

        # Download JSON
        st.divider()
        st.download_button(
            label="Download Results (JSON)",
            data=json.dumps(results, indent=2),
            file_name=f"spinescan_{results['study_id']}.json",
            mime="application/json"
        )


# ── TAB 3: Grad-CAM ──────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-title">Grad-CAM Visualization</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Grad-CAM highlights the image regions that most influenced "
        "the model's severity prediction."
    )

    if "study_id" not in st.session_state:
        st.markdown(
            '<div class="info-box">Run analysis first (Select Study & Analyze tab) '
            'to enable Grad-CAM.</div>',
            unsafe_allow_html=True
        )
    else:
        study_id = st.session_state["study_id"]
        st.markdown(f"**Study ID:** `{study_id}`")

        col1, col2, col3 = st.columns(3)
        with col1:
            gcam_cond_label = st.selectbox("Condition", list(CONDITIONS.keys()),
                                           key="gcam_cond")
        with col2:
            gcam_level = st.selectbox("Level", LEVELS, index=3, key="gcam_level")
        with col3:
            gcam_view = st.selectbox("View", ["Sagittal", "Axial"], key="gcam_view")

        gcam_cond_key = CONDITIONS[gcam_cond_label]
        gcam_view_key = gcam_view.lower()

        if st.button("Generate Grad-CAM", type="primary"):
            with st.spinner(f"Computing {gcam_view} Grad-CAM..."):
                try:
                    from src.gradcam import run_gradcam_for_study
                    result = run_gradcam_for_study(
                        study_id=study_id,
                        condition=gcam_cond_key,
                        level=gcam_level,
                        view_type=gcam_view_key,
                        fold=fold,
                        device=device,
                    )
                    st.session_state["gcam_result"] = result
                    st.session_state["gcam_label"] = (gcam_cond_label, gcam_level, gcam_view)
                except Exception as e:
                    st.error(f"Grad-CAM failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # Display results if available
        if "gcam_result" in st.session_state:
            result = st.session_state["gcam_result"]
            label_info = st.session_state.get("gcam_label", ("", "", ""))

            st.markdown(f'<div class="section-title">'
                        f'{label_info[0]} | {label_info[1]} | {label_info[2]}</div>',
                        unsafe_allow_html=True)

            # Side-by-side: Original | Overlay
            col_orig, col_overlay = st.columns(2)
            with col_orig:
                st.markdown("**Original**")
                orig_rgb = cv2.cvtColor(result['original'], cv2.COLOR_BGR2RGB)
                st.image(orig_rgb, use_container_width=True)
            with col_overlay:
                st.markdown("**Grad-CAM Overlay**")
                overlay_rgb = cv2.cvtColor(result['overlay'], cv2.COLOR_BGR2RGB)
                st.image(overlay_rgb, use_container_width=True)

            # Probability bars
            logits = result['logits']
            probs = softmax(logits.reshape(1, -1))[0]
            st.markdown("**Model Confidence**")
            prob_cols = st.columns(3)
            for i, sev in enumerate(["Normal/Mild", "Moderate", "Severe"]):
                with prob_cols[i]:
                    prob_to_bar(float(probs[i]), sev)

            pred_idx = int(np.argmax(probs))
            pred_label = ["Normal/Mild", "Moderate", "Severe"][pred_idx]
            st.markdown(
                f"Prediction: {severity_badge(pred_label)} "
                f"({probs[pred_idx]:.1%})",
                unsafe_allow_html=True
            )

            # Download overlay
            _, img_encoded = cv2.imencode(".png", result['overlay'])
            st.download_button(
                label="Download Grad-CAM Image",
                data=img_encoded.tobytes(),
                file_name=f"gradcam_{gcam_cond_key}_{gcam_level}_{gcam_view_key}.png",
                mime="image/png"
            )

        st.divider()
        st.markdown("**How to interpret:**")
        st.markdown(
            "- **Red/warm regions**: high influence on the severity prediction\n"
            "- **Blue/cool regions**: low influence\n"
            "- The heatmap targets the **Severe** class to show what the model "
            "considers most indicative of pathology"
        )


# ── TAB 4: Pipeline Guide ─────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-title">Quick Start Guide</div>', unsafe_allow_html=True)

    steps = [
        ("1. Clone the repo",
         "```bash\ncd C:\\Users\\hamma\\Desktop\\SpineSCAN_FYP\n"
         "git clone https://github.com/yujiariyasu/spine_model.git\n```"),

        ("2. Apply RTX 4060 patch",
         "```bash\npython patch_for_rtx4060.py\n```\n"
         "This reduces batch sizes, enables FP16, and fixes Windows paths."),

        ("3. Download competition data",
         "```bash\n# Place your kaggle.json in ~/.kaggle/\n"
         "cd spine_model\\input\n"
         "kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification\n"
         "# Unzip: python -c \"import zipfile; zipfile.ZipFile('rsna-2024-lumbar-spine-degenerative-classification.zip').extractall('.')\"\n```"),

        ("4. Run axial level estimation (Kaggle kernel)",
         "```bash\n# Run this from kaggle.com/code/yujiariyasu/axial-level-estimation\n"
         "# Then download output:\n"
         "kaggle kernels output yujiariyasu/axial-level-estimation -p ./input/\n```"),

        ("5. Check environment",
         "```bash\npython setup_check.py\n```"),

        ("6. Run full pipeline",
         "```batch\nrun_pipeline.bat\n```\n"
         "Estimated ~24-40 hours on RTX 4060 (leave running overnight)."),

        ("7. Launch web app",
         "```bash\nstreamlit run app.py\n```"),
    ]

    for title, body in steps:
        with st.expander(title, expanded=False):
            st.markdown(body)

    st.divider()
    st.markdown("**Verifying CUDA:**")
    st.code('python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"')

    st.markdown("**Checking available config names (after cloning):**")
    st.code(
        "import sys; sys.path.insert(0, 'spine_model')\n"
        "from src.configs import *\n"
        "import inspect\n"
        "[print(n) for n, c in inspect.getmembers(sys.modules[__name__], inspect.isclass) if 'rsna' in n.lower()]"
    )

    st.markdown("**Manual training command (single fold):**")
    st.code(
        "cd spine_model\n"
        "python train_one_fold.py -c rsna_sagittal_level_cl_spinal_v1 -f 0"
    )
