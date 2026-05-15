"""
predict.py — Reusable prediction module using the full improved pipeline.
"""
import os, sys
import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (MODEL_PATH, SCALER_PATH, SELECTOR_PATH,
                    ORIGINAL_FEATURES)
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
