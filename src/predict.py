"""
predict.py — Reusable prediction module using the full improved pipeline.
"""
import os, sys
import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    MODEL_PATH,
    SCALER_PATH,
    SELECTOR_PATH,
    ORIGINAL_FEATURES,
)
from preprocessing import engineer_features


def _load_artifacts():
    for p in [MODEL_PATH, SCALER_PATH, SELECTOR_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}. Run 'python src/train.py' first.")
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH), joblib.load(SELECTOR_PATH)


def _prepare(feature_dict: dict):
    """Apply full pipeline: engineer -> scale -> select."""
    model, scaler, selector = _load_artifacts()
    df = pd.DataFrame([feature_dict])[ORIGINAL_FEATURES]
    df_eng = engineer_features(df)
    df_sc  = scaler.transform(df_eng)
    df_sel = selector.transform(df_sc)
    return model, df_sel


def predict_single(feature_dict: dict) -> dict:
    """
    Predict for one patient. All 22 MDVP features must be provided.

    Returns dict with: prediction, label, confidence, pd_proba, healthy_proba
    """
    missing = [f for f in ORIGINAL_FEATURES if f not in feature_dict]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    model, X_sel = _prepare(feature_dict)
    label = int(model.predict(X_sel)[0])
    proba = model.predict_proba(X_sel)[0]

    return {
        "prediction"    : "Parkinson's" if label == 1 else "Healthy",
        "label"         : label,
        "confidence"    : float(proba[label]),
        "pd_proba"      : float(proba[1]),
        "healthy_proba" : float(proba[0]),
    }


def predict_batch(filepath_or_df) -> pd.DataFrame:
    """
    Predict for multiple patients from a CSV file or DataFrame.
    Columns 'name' and 'status' are ignored if present.
    """
    if isinstance(filepath_or_df, str):
        df = pd.read_csv(filepath_or_df)
    else:
        df = filepath_or_df.copy()

    for col in ["name", "status"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    missing = [f for f in ORIGINAL_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    model, scaler, selector = _load_artifacts()
    X     = df[ORIGINAL_FEATURES]
    X_eng = engineer_features(X)
    X_sc  = scaler.transform(X_eng)
    X_sel = selector.transform(X_sc)

    labels = model.predict(X_sel)
    probas = model.predict_proba(X_sel)

    result = df.copy()
    result["Prediction"]    = ["Parkinson's" if l == 1 else "Healthy" for l in labels]
    result["PD_Risk_%"]     = (probas[:, 1] * 100).round(2)
    result["Confidence_%"]  = [round(probas[i, l] * 100, 2) for i, l in enumerate(labels)]
    result["Risk_Level"]    = pd.cut(probas[:, 1],
                                      bins=[0, 0.3, 0.6, 1.0],
                                      labels=["Low", "Medium", "High"]).astype(str)
    return result


def predict_from_voice(raw_features: dict) -> dict:
    """
    End-to-end voice-to-prediction integration for the Streamlit UI.

    Accepts the 22-key dictionary produced by src/extractor.extract_voice_features()
    and runs the full ML inference pipeline:

        ┌─────────────────────────────────────────────────────────────┐
        │  raw_features (22 keys from extractor)                     │
        │       │                                                    │
        │       ▼                                                    │
        │  Step 1 — Validate: confirm all 22 ORIGINAL_FEATURES exist │
        │       │                                                    │
        │       ▼                                                    │
        │  Step 2 — DataFrame: single-row df in canonical column     │
        │          order from config.ORIGINAL_FEATURES               │
        │       │                                                    │
        │       ▼                                                    │
        │  Step 3 — Engineer: preprocessing.engineer_features()      │
        │          adds 7 derived features → 29 columns              │
        │       │                                                    │
        │       ▼                                                    │
        │  Step 4 — Scale: scaler.pkl (MinMax [-1, 1])               │
        │          fitted on training data, no data leakage          │
        │       │                                                    │
        │       ▼                                                    │
        │  Step 5 — Select: selector.pkl (Mutual Information)        │
        │          reduces 29 → 18 most informative features         │
        │       │                                                    │
        │       ▼                                                    │
        │  Step 6 — Predict: model.pkl (VotingClassifier)            │
        │          soft-voting across GB + RF + SVM + ExtraTrees     │
        │       │                                                    │
        │       ▼                                                    │
        │  Return: dict with label, probabilities, risk level        │
        └─────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    raw_features : dict
        Exactly 22 keys matching config.ORIGINAL_FEATURES.
        Typically produced by extractor.extract_voice_features(wav_path).

    Returns
    -------
    dict
        prediction     : str   — "Parkinson's" or "Healthy"
        label          : int   — 1 (PD) or 0 (Healthy)
        confidence     : float — model confidence in its own prediction [0–1]
        pd_probability : float — P(Parkinson's) from soft voting [0–1]
        healthy_probability : float — P(Healthy) [0–1]
        risk_level     : str   — "Low" / "Medium" / "High"
        features_used  : dict  — the validated 22 input features (for UI display)

    Raises
    ------
    ValueError
        If any of the 22 required features are missing from the input.
    FileNotFoundError
        If model/scaler/selector .pkl files are not found on disk.
    """

    # ── Step 1: Validate that all 22 features are present ─────────────
    missing = [f for f in ORIGINAL_FEATURES if f not in raw_features]
    if missing:
        raise ValueError(
            f"Voice extraction incomplete — missing {len(missing)} features: "
            f"{missing}. Please re-record a longer, clearer sample."
        )

    # ── Step 2: Build a single-row DataFrame in canonical column order ─
    df = pd.DataFrame([raw_features])[ORIGINAL_FEATURES]

    # ── Step 3: Engineer 7 derived features (29 total columns) ────────
    df_engineered = engineer_features(df)

    # ── Step 4 & 5: Load artifacts, scale, and select ─────────────────
    model, scaler, selector = _load_artifacts()

    df_scaled   = scaler.transform(df_engineered)    # MinMax [-1, 1]
    df_selected = selector.transform(df_scaled)      # 29 → 18 features

    # ── Step 6: Soft-voting ensemble inference ────────────────────────
    label = int(model.predict(df_selected)[0])
    probas = model.predict_proba(df_selected)[0]

    pd_prob = float(probas[1])
    risk = "High" if pd_prob >= 0.6 else ("Medium" if pd_prob >= 0.3 else "Low")

    return {
        "prediction"          : "Parkinson's" if label == 1 else "Healthy",
        "label"               : label,
        "confidence"          : float(probas[label]),
        "pd_probability"      : pd_prob,
        "healthy_probability" : float(probas[0]),
        "risk_level"          : risk,
        "features_used"       : {k: raw_features[k] for k in ORIGINAL_FEATURES},
    }


if __name__ == "__main__":
    sample = {
        "MDVP:Fo(Hz)": 119.992, "MDVP:Fhi(Hz)": 157.302, "MDVP:Flo(Hz)": 74.997,
        "MDVP:Jitter(%)": 0.00784, "MDVP:Jitter(Abs)": 0.00007,
        "MDVP:RAP": 0.0037, "MDVP:PPQ": 0.00554, "Jitter:DDP": 0.01109,
        "MDVP:Shimmer": 0.04374, "MDVP:Shimmer(dB)": 0.426,
        "Shimmer:APQ3": 0.02182, "Shimmer:APQ5": 0.0313,
        "MDVP:APQ": 0.02971, "Shimmer:DDA": 0.06545,
        "NHR": 0.02211, "HNR": 21.033,
        "RPDE": 0.414783, "DFA": 0.815285,
        "spread1": -4.813031, "spread2": 0.266482,
        "D2": 2.301442, "PPE": 0.284654
    }
    print(predict_single(sample))
