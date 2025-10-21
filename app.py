from __future__ import annotations
import os
import pathlib
from typing import Optional

import pandas as pd
from PIL import Image
import streamlit as st

from utils import (
    SIZE, CLASS_NAMES,
    load_joblib_from_bytes, load_joblib_from_path,
    pil_list_to_features,
    run_binary, run_multiclass
)

st.set_page_config(page_title="Skin Lesion SVM Demo", layout="wide")
st.title("🧪 Skin Lesion Classification — SVM (Pretrained, No CNN)")

# --- Auto-load config ---
DEFAULTS = {
    "BIN_MODEL_PATH":   os.environ.get("BIN_MODEL_PATH",   "./models/svm_binary_model.joblib"),
    "BIN_SCALER_PATH":  os.environ.get("BIN_SCALER_PATH",  "./models/scaler_binary.joblib"),
    "MC_MODEL_PATH":    os.environ.get("MC_MODEL_PATH",    "./models/svm_multi_class_model.joblib"),
    "MC_SCALER_PATH":   os.environ.get("MC_SCALER_PATH",   "./models/scaler_multi_class.joblib"),
}
st.caption(f"🔎 Auto paths: {DEFAULTS}")

# Optional remote URLs via Streamlit secrets (e.g., GitHub Releases / HF / S3)
REMOTE_URLS = st.secrets.get("MODEL_URLS", {}) if hasattr(st, "secrets") else {}
CACHE_DIR = pathlib.Path(os.environ.get("MODEL_CACHE_DIR", ".model_cache"))
CACHE_DIR.mkdir(exist_ok=True)

@st.cache_resource(show_spinner=True)
def _download_if_needed(url: str, dst_name: str) -> Optional[pathlib.Path]:
    """Download a file to cache dir, return its path. Requires 'requests' in requirements.txt."""
    if not url:
        return None
    try:
        import requests
    except Exception:
        st.error("`requests` not installed; add it to requirements.txt to enable remote download.")
        return None
    dst = CACHE_DIR / dst_name
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dst.write_bytes(r.content)
        return dst
    except Exception as e:
        st.warning(f"Failed to download {url}: {e}")
        return None

@st.cache_resource(show_spinner=True)
def _auto_load_artifacts():
    """Try local paths, then remote URLs. Return (bin_model, bin_scaler, mc_model, mc_scaler, source)."""
    source = []

    # 1) Local repo/container paths
    bm = DEFAULTS["BIN_MODEL_PATH"]; bs = DEFAULTS["BIN_SCALER_PATH"]
    mm = DEFAULTS["MC_MODEL_PATH"];  ms = DEFAULTS["MC_SCALER_PATH"]

    bin_model = load_joblib_from_path(bm) if os.path.exists(bm) else None
    if bin_model: source.append(f"local:{bm}")
    bin_scaler = load_joblib_from_path(bs) if os.path.exists(bs) else None
    if bin_scaler: source.append(f"local:{bs}")
    mc_model  = load_joblib_from_path(mm) if os.path.exists(mm) else None
    if mc_model: source.append(f"local:{mm}")
    mc_scaler = load_joblib_from_path(ms) if os.path.exists(ms) else None
    if mc_scaler: source.append(f"local:{ms}")

    if (bin_model and bin_scaler) or (mc_model and mc_scaler):
        return bin_model, bin_scaler, mc_model, mc_scaler, " / ".join(source) or "local"

    # 2) Remote URLs from secrets
    if REMOTE_URLS:
        bm_url = REMOTE_URLS.get("bin_model")
        bs_url = REMOTE_URLS.get("bin_scaler")
        mm_url = REMOTE_URLS.get("mc_model")
        ms_url = REMOTE_URLS.get("mc_scaler")

        bm_p = _download_if_needed(bm_url, "svm_binary_model.joblib")
        bs_p = _download_if_needed(bs_url, "scaler_binary.joblib")
        mm_p = _download_if_needed(mm_url, "svm_multi_class_model.joblib")
        ms_p = _download_if_needed(ms_url, "scaler_multi_class.joblib")

        bin_model = load_joblib_from_path(str(bm_p)) if bm_p and bm_p.exists() else None
        bin_scaler = load_joblib_from_path(str(bs_p)) if bs_p and bs_p.exists() else None
        mc_model  = load_joblib_from_path(str(mm_p)) if mm_p and mm_p.exists() else None
        mc_scaler = load_joblib_from_path(str(ms_p)) if ms_p and ms_p.exists() else None

        if (bin_model and bin_scaler) or (mc_model and mc_scaler):
            return bin_model, bin_scaler, mc_model, mc_scaler, "remote"

    return None, None, None, None, "none"

with st.sidebar:
    st.header("Model source")
    use_auto = st.toggle("Use bundled/remote models if available", value=True, help="Load from ./models or secrets URLs on startup")

    st.markdown("---")
    st.subheader("Manual upload (fallback)")
    bin_model_file = st.file_uploader("Binary SVM (svm_binary_model.joblib)", type=['joblib','pkl','sav'], key='bin_model')
    bin_scaler_file = st.file_uploader("Binary Scaler (scaler_binary.joblib)", type=['joblib','pkl','sav'], key='bin_scaler')
    mc_model_file = st.file_uploader("Multiclass SVM (svm_multi_class_model.joblib)", type=['joblib','pkl','sav'], key='mc_model')
    mc_scaler_file = st.file_uploader("Multiclass Scaler (scaler_multi_class.joblib)", type=['joblib','pkl','sav'], key='mc_scaler')

@st.cache_resource(show_spinner=False)
def _load_uploaded(bin_m, bin_s, mc_m, mc_s):
    bin_model = load_joblib_from_bytes(bin_m.getvalue()) if bin_m else None
    bin_scaler = load_joblib_from_bytes(bin_s.getvalue()) if bin_s else None
    mc_model  = load_joblib_from_bytes(mc_m.getvalue()) if mc_m else None
    mc_scaler = load_joblib_from_bytes(mc_s.getvalue()) if mc_s else None
    return bin_model, bin_scaler, mc_model, mc_scaler

def get_artifacts():
    if use_auto:
        bm, bs, mm, ms, src = _auto_load_artifacts()
        if src != "none":
            st.caption(f"✅ Loaded models from {src}")
            return bm, bs, mm, ms
    return _load_uploaded(bin_model_file, bin_scaler_file, mc_model_file, mc_scaler_file)

tabs = st.tabs(["🔍 Single Image", "📦 Batch Inference", "ℹ️ About / Notes"])

with tabs[0]:
    st.subheader("Single Image Prediction")
    up = st.file_uploader("Upload a dermoscopic image (JPG/PNG)", type=['jpg','jpeg','png'], key='single')
    if up is not None:
        img = Image.open(up).convert('RGB')
        st.image(img, caption="Uploaded image", use_column_width=True)

        bin_model, bin_scaler, mc_model, mc_scaler = get_artifacts()
        if (bin_model is None or bin_scaler is None) and (mc_model is None or mc_scaler is None):
            st.info("No models loaded yet. Bundle them under ./models or set secrets URLs, or upload in the sidebar.")
        else:
            feats = pil_list_to_features([img], size=SIZE)

            if bin_model and bin_scaler:
                pred_bin, pB, is_prob = run_binary(feats, bin_model, bin_scaler)
                label = "B (mel/bcc/akiec/vasc)" if int(pred_bin[0]) == 1 else "M (nv/df/bkl)"
                st.success(f"**Binary** → {label}  |  {'P(B)=' if is_prob else 'score≈'}**{float(pB[0]):.3f}**")

            if mc_model and mc_scaler:
                top1_idx, top1_score, proba_like, has_prob = run_multiclass(feats, mc_model, mc_scaler, CLASS_NAMES)
                top = int(top1_idx[0])
                st.write("**Multiclass (7 classes)**")
                st.success(f"Top-1: **{CLASS_NAMES[top]}**  |  {'p≈' if has_prob else 'score≈'}{float(top1_score[0]):.3f}")
                order = proba_like[0].argsort()[::-1][:3]
                df_top = pd.DataFrame({
                    "rank":[1,2,3],
                    "class":[CLASS_NAMES[i] for i in order],
                    "score":[float(proba_like[0][i]) for i in order]
                })
                st.dataframe(df_top, hide_index=True, use_container_width=True)

with tabs[1]:
    st.subheader("Batch Inference on Multiple Images")
    ups = st.file_uploader("Upload one or more images", type=['jpg','jpeg','png'], accept_multiple_files=True, key='batch')
    if ups:
        bin_model, bin_scaler, mc_model, mc_scaler = get_artifacts()
        if (bin_model is None or bin_scaler is None) and (mc_model is None or mc_scaler is None):
            st.info("No models loaded yet. Bundle them under ./models or set secrets URLs, or upload in the sidebar.")
        else:
            imgs, names = [], []
            for f in ups:
                try:
                    im = Image.open(f).convert('RGB')
                    imgs.append(im)
                    names.append(os.path.basename(f.name))
                except Exception as e:
                    st.warning(f"Skipping {f.name}: {e}")

            feats = pil_list_to_features(imgs, size=SIZE)
            rows = []

            # Binary
            if bin_model and bin_scaler:
                pred_bin, pB, has_bin_prob = run_binary(feats, bin_model, bin_scaler)
            else:
                pred_bin = pB = has_bin_prob = None

            # Multiclass
            if mc_model and mc_scaler:
                top1_idx, top1_score, proba_like, has_mc_prob = run_multiclass(feats, mc_model, mc_scaler, CLASS_NAMES)
            else:
                top1_idx = top1_score = proba_like = has_mc_prob = None

            for i, name in enumerate(names):
                row = {"image": name}
                if pred_bin is not None:
                    row["binary_pred"] = int(pred_bin[i])
                    row["binary_label"] = "B" if row["binary_pred"] == 1 else "M"
                    row["P(B)≈" if has_bin_prob else "score≈"] = float(pB[i])
                if top1_idx is not None:
                    row["multiclass_top1"] = CLASS_NAMES[int(top1_idx[i])]
                    row["score≈"] = float(top1_score[i])
                rows.append(row)

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download results as CSV", csv, file_name="svm_predictions.csv", mime="text/csv")