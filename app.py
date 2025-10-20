from __future__ import annotations
import os
import pandas as pd
from PIL import Image

import streamlit as st

from utils import (
    SIZE, CLASS_NAMES,
    load_joblib_from_bytes,
    pil_list_to_features,
    run_binary, run_multiclass
)

st.set_page_config(page_title="Skin Lesion SVM Demo", layout="wide")
st.title("🧪 Skin Lesion Classification — SVM (Pretrained, No CNN)")

with st.sidebar:
    st.header("🔧 Load pretrained artifacts")
    st.write("Upload the **trained joblib models** and **scalers**. No training here.")
    bin_model_file = st.file_uploader("Binary SVM (svm_binary_model.joblib)", type=['joblib','pkl','sav'], key='bin_model')
    bin_scaler_file = st.file_uploader("Binary Scaler (scaler_binary.joblib)", type=['joblib','pkl','sav'], key='bin_scaler')
    mc_model_file = st.file_uploader("Multiclass SVM (svm_multi_class_model.joblib)", type=['joblib','pkl','sav'], key='mc_model')
    mc_scaler_file = st.file_uploader("Multiclass Scaler (scaler_multi_class.joblib)", type=['joblib','pkl','sav'], key='mc_scaler')

@st.cache_resource
def _load_artifacts(bin_m, bin_s, mc_m, mc_s):
    bin_model = load_joblib_from_bytes(bin_m.getvalue()) if bin_m else None
    bin_scaler = load_joblib_from_bytes(bin_s.getvalue()) if bin_s else None
    mc_model  = load_joblib_from_bytes(mc_m.getvalue()) if mc_m else None
    mc_scaler = load_joblib_from_bytes(mc_s.getvalue()) if mc_s else None
    return bin_model, bin_scaler, mc_model, mc_scaler

tabs = st.tabs(["🔍 Single Image", "📦 Batch Inference", "ℹ️ About / Notes"])

with tabs[0]:
    st.subheader("Single Image Prediction")
    up = st.file_uploader("Upload a dermoscopic image (JPG/PNG)", type=['jpg','jpeg','png'], key='single')

    if up is not None:
        img = Image.open(up).convert('RGB')
        col1, col2 = st.columns([1,1])
        with col1:
            st.image(img, caption="Uploaded image", use_column_width=True)

        # Load artifacts
        bin_model, bin_scaler, mc_model, mc_scaler = _load_artifacts(bin_model_file, bin_scaler_file, mc_model_file, mc_scaler_file)
        if (bin_model is None or bin_scaler is None) and (mc_model is None or mc_scaler is None):
            st.info("Please upload at least a **binary pair** or a **multiclass pair** of model + scaler in the sidebar.")
        else:
            # Extract features
            feats = pil_list_to_features([img], size=SIZE)

            # Binary
            if bin_model and bin_scaler:
                pred_bin, pB, is_prob = run_binary(feats, bin_model, bin_scaler)
                label = "B (mel/bcc/akiec/vasc)" if int(pred_bin[0]) == 1 else "M (nv/df/bkl)"
                if is_prob:
                    st.success(f"**Binary** → {label}  |  P(B)=**{float(pB[0]):.3f}**")
                else:
                    st.success(f"**Binary** → {label}  |  score≈**{float(pB[0]):.3f}** (sigmoid of margin)")
            else:
                st.info("Upload binary SVM **and** its scaler to enable binary prediction.")

            # Multiclass
            if mc_model and mc_scaler:
                top1_idx, top1_score, proba_like, has_prob = run_multiclass(feats, mc_model, mc_scaler, CLASS_NAMES)
                top = int(top1_idx[0])
                st.write("**Multiclass (7 classes)**")
                if has_prob:
                    st.success(f"Top‑1: **{CLASS_NAMES[top]}**  |  p≈{float(top1_score[0]):.3f}")
                else:
                    st.success(f"Top‑1: **{CLASS_NAMES[top]}**  |  score≈{float(top1_score[0]):.3f} (softmax of margins)")

                # Show Top‑3
                order = proba_like[0].argsort()[::-1][:3]
                import pandas as pd
                df_top = pd.DataFrame({
                    "rank":[1,2,3],
                    "class":[CLASS_NAMES[i] for i in order],
                    "score":[float(proba_like[0][i]) for i in order]
                })
                st.dataframe(df_top, hide_index=True, use_container_width=True)
            else:
                st.info("Upload multi‑class SVM **and** its scaler to enable 7‑class prediction.")

with tabs[1]:
    st.subheader("Batch Inference on Multiple Images")
    ups = st.file_uploader("Upload one or more images", type=['jpg','jpeg','png'], accept_multiple_files=True, key='batch')
    if ups:
        bin_model, bin_scaler, mc_model, mc_scaler = _load_artifacts(bin_model_file, bin_scaler_file, mc_model_file, mc_scaler_file)
        if (bin_model is None or bin_scaler is None) and (mc_model is None or mc_scaler is None):
            st.info("Please upload at least a **binary pair** or a **multiclass pair** of model + scaler in the sidebar.")
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
            pred_bin = pB = None
            has_bin_prob = False
            if bin_model and bin_scaler:
                pred_bin, pB, has_bin_prob = run_binary(feats, bin_model, bin_scaler)

            # Multiclass
            top1_idx = top1_score = proba_like = None
            has_mc_prob = False
            if mc_model and mc_scaler:
                top1_idx, top1_score, proba_like, has_mc_prob = run_multiclass(feats, mc_model, mc_scaler, CLASS_NAMES)

            import pandas as pd
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
