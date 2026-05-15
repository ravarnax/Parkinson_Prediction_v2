"""
preprocessing.py — Feature engineering, selection, and scaling pipeline.

Improvements over v1:
  - stratify=y in train_test_split (fixes biased test set)
  - 7 engineered features derived from domain knowledge
  - Mutual information feature selection (22+7 -> 18 best features)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import ORIGINAL_FEATURES, ENGINEERED_FEATURES


def load_data(filepath):
    """Load the Parkinson's dataset from a CSV/data file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded: {filepath}  |  Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add 7 derived features based on clinical domain knowledge.

    New features created:
      PPE_RPDE_sum       — combined entropy signal (both measure neural irregularity)
      spread_range       — nonlinear frequency spread range
      Fo_range           — fundamental frequency bandwidth
      Jitter_total       — aggregate jitter across all measures
      Shimmer_total      — aggregate shimmer across all measures
      nonlinear_composite— product of 3 nonlinear chaos measures
      HNR_NHR_diff       — signal clarity proxy (high = cleaner voice)
    """
    X = X.copy()
    X['PPE_RPDE_sum']        = X['PPE'] + X['RPDE']
    X['spread_range']        = X['spread2'] - X['spread1']
    X['Fo_range']            = X['MDVP:Fhi(Hz)'] - X['MDVP:Flo(Hz)']
    X['Jitter_total']        = X['MDVP:Jitter(%)'] + X['MDVP:RAP'] + X['MDVP:PPQ']
    X['Shimmer_total']       = X['MDVP:Shimmer'] + X['Shimmer:APQ3'] + X['Shimmer:APQ5']
    X['nonlinear_composite'] = X['PPE'] * X['RPDE'] * X['DFA']
    X['HNR_NHR_diff']        = X['HNR'] - X['NHR'] * 10
    return X


def preprocess_data(df, test_size=0.2, random_state=42, n_features=18):
    """
    Full preprocessing pipeline:
      1. Drop name column
      2. Engineer 7 new features
      3. Stratified 80/20 train-test split
      4. MinMax scale to [-1, 1]
      5. Mutual information feature selection (best n_features)

    Returns:
        X_train, X_test, y_train, y_test, scaler, selector
    """
    if "name" in df.columns:
        df = df.drop(["name"], axis=1)

    X = df.drop(["status"], axis=1)[ORIGINAL_FEATURES]
    y = df["status"]

    print(f"Original features  : {X.shape[1]}")
    print(f"Class balance      : PD={y.sum()} ({y.mean()*100:.1f}%)  Healthy={len(y)-y.sum()}")

    # Step 1: Feature engineering
    X_eng = engineer_features(X)
    print(f"After engineering  : {X_eng.shape[1]} features")

    # Step 2: Stratified split (fix: preserves 75:25 ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X_eng, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

    # Step 3: Scale (fit on train only — no data leakage)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Step 4: Feature selection (fit on train only)
    selector = SelectKBest(mutual_info_classif, k=n_features)
    X_train_sel = selector.fit_transform(X_train_sc, y_train)
    X_test_sel  = selector.transform(X_test_sc)

    selected = [list(X_eng.columns)[i] for i in selector.get_support(indices=True)]
    print(f"After selection    : {len(selected)} features -> {selected}")

    return X_train_sel, X_test_sel, y_train, y_test, scaler, selector
