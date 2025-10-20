from __future__ import annotations
import io
import math
import warnings
from typing import Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from PIL import Image

import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from skimage.filters import gabor

# -----------------------------
# Constants & label semantics
# -----------------------------
SIZE: int = 32  # must match training
# If your LabelEncoder followed alphabetical order on 'dx' from HAM10000:
CLASS_NAMES: List[str] = ['akiec','bcc','bkl','df','mel','nv','vasc']

# Binary mapping used in the notebook:
# B = 1: {mel, bcc, akiec, vasc}; M = 0: {nv, df, bkl}
BINARY_POSITIVE = {'mel','bcc','akiec','vasc'}  # label 1
BINARY_NEGATIVE = {'nv','df','bkl'}             # label 0

# -----------------------------
# Loading utilities
# -----------------------------
def load_joblib_from_bytes(bytes_obj) -> object:
    """Load a joblib object from raw bytes (e.g., Streamlit file_uploader.getvalue())."""
    if bytes_obj is None:
        return None
    return joblib.load(io.BytesIO(bytes_obj))

# -----------------------------
# Feature extractors (match the notebook pipeline)
# -----------------------------
def _lbp_hist(gray: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
    lbp = local_binary_pattern(gray, P=P, R=R, method='uniform')
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins), density=True)
    return hist.astype(np.float32)  # 10 dims

def _glcm_props(gray: np.ndarray,
                distances: Sequence[int] = (1, 2, 3),
                angles: Sequence[float] = (0, np.pi/4, np.pi/2, 3*np.pi/4)) -> np.ndarray:
    g8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    M = graycomatrix(g8, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    props = ['contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity']
    feats = [graycoprops(M, p).ravel() for p in props]  # each len = len(distances)*len(angles)
    return np.concatenate(feats).astype(np.float32)     # 5 * (3*4) = 60

def _hu_moments(gray: np.ndarray) -> np.ndarray:
    m = cv2.moments(gray)
    hu = cv2.HuMoments(m).flatten()
    return (-np.sign(hu) * np.log10(np.abs(hu) + 1e-12)).astype(np.float32)  # 7

def _gabor_bank(gray: np.ndarray,
                thetas: Sequence[float] = (0, np.pi/4, np.pi/2, 3*np.pi/4),
                freqs: Sequence[float]  = (0.1, 0.2, 0.3)) -> np.ndarray:
    feats = []
    g = gray.astype(np.float32) / 255.0
    for t in thetas:
        for f in freqs:
            real, imag = gabor(g, frequency=f, theta=t)
            mag = np.sqrt(real**2 + imag**2)
            feats.append(mag.mean())
            feats.append(mag.std())
    return np.array(feats, dtype=np.float32)  # 24

def _hog_features(gray: np.ndarray,
                  orientations: int = 9,
                  pixels_per_cell: Tuple[int, int] = (8, 8),
                  cells_per_block: Tuple[int, int] = (2, 2)) -> np.ndarray:
    g = gray.astype(np.float32) / 255.0
    h = hog(
        g,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        visualize=False
    )
    return h.astype(np.float32)  # ~324 dims on 32x32

def _intensity_stats(gray: np.ndarray) -> np.ndarray:
    return np.array([gray.mean(), gray.std()], dtype=np.float32)  # 2

def image_to_flat_gray(img: Image.Image, size: int = SIZE) -> np.ndarray:
    g = img.convert('L').resize((size, size))
    return np.array(g, dtype=np.uint8).reshape(-1)

def extract_features_one(flat_gray: np.ndarray, size: int = SIZE) -> np.ndarray:
    g = flat_gray.reshape(size, size)
    f_lbp  = _lbp_hist(g)              # 10
    f_glcm = _glcm_props(g)            # 60
    f_hu   = _hu_moments(g)            # 7
    f_gab  = _gabor_bank(g)            # 24
    f_hog  = _hog_features(g)          # ~324
    f_int  = _intensity_stats(g)       # 2
    return np.concatenate([f_lbp, f_glcm, f_hu, f_gab, f_hog, f_int]).astype(np.float32)

def extract_features_batch(flat_grays: List[np.ndarray], size: int = SIZE) -> np.ndarray:
    sample = flat_grays[0].reshape(size, size)
    n_hog = len(_hog_features(sample))
    n_feats = 10 + 60 + 7 + 24 + n_hog + 2
    out = np.zeros((len(flat_grays), n_feats), dtype=np.float32)
    for i, arr in enumerate(flat_grays):
        try:
            out[i] = extract_features_one(arr, size=size)
        except Exception:
            out[i] = np.zeros(n_feats, dtype=np.float32)
    return out

def pil_list_to_features(images: List[Image.Image], size: int = SIZE) -> np.ndarray:
    flats = [image_to_flat_gray(im, size=size) for im in images]
    return extract_features_batch(flats, size=size)

# -----------------------------
# Inference helpers
# -----------------------------
def build_model_input(X_img_feats: np.ndarray, scaler) -> np.ndarray:
    """Pad or truncate to match scaler.n_features_in_ if necessary."""
    n_expected = getattr(scaler, 'n_features_in_', X_img_feats.shape[1])
    if n_expected == X_img_feats.shape[1]:
        return X_img_feats
    elif n_expected > X_img_feats.shape[1]:
        pad = np.zeros((X_img_feats.shape[0], n_expected - X_img_feats.shape[1]), dtype=np.float32)
        return np.hstack([X_img_feats, pad])
    else:
        warnings.warn(f"Image features ({X_img_feats.shape[1]}) exceed scaler.n_features_in_ ({n_expected}). Truncating.")
        return X_img_feats[:, :n_expected]

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def ovr_softmax(decision_scores: np.ndarray) -> np.ndarray:
    Z = decision_scores if decision_scores.ndim == 2 else decision_scores.reshape(1, -1)
    eZ = np.exp(Z - Z.max(axis=1, keepdims=True))
    return eZ / eZ.sum(axis=1, keepdims=True)

def run_binary(X_img_feats: np.ndarray, model, scaler):
    """Return (pred_labels, score_or_prob), where score_or_prob is P(B) if probas available,
    otherwise a logistic of margin for display."""
    Xb = build_model_input(X_img_feats, scaler)
    Xb_sc = scaler.transform(Xb)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xb_sc)  # shape (N, 2) with columns [P(M=0), P(B=1)]
        pred = np.argmax(proba, axis=1).astype(int)
        pB = proba[:, 1].astype(float)
        return pred, pB, True
    else:
        margins = model.decision_function(Xb_sc).astype(np.float32)  # positive → class 1
        pred = model.predict(Xb_sc).astype(int)
        pB = sigmoid(margins)  # pseudo-probability for display
        return pred, pB, False

def run_multiclass(X_img_feats: np.ndarray, model, scaler, class_names: Sequence[str] = CLASS_NAMES):
    Xm = build_model_input(X_img_feats, scaler)
    Xm_sc = scaler.transform(Xm)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xm_sc)  # (N, C)
        top1_idx = np.argmax(proba, axis=1).astype(int)
        top1_score = proba[np.arange(len(top1_idx)), top1_idx]
        return top1_idx, top1_score.astype(float), proba.astype(float), True
    else:
        scores = model.decision_function(Xm_sc)
        scores = scores if scores.ndim == 2 else scores.reshape(-1, len(class_names))
        proba_like = ovr_softmax(scores)
        top1_idx = np.argmax(proba_like, axis=1).astype(int)
        top1_score = proba_like[np.arange(len(top1_idx)), top1_idx]
        return top1_idx, top1_score.astype(float), proba_like.astype(float), False
