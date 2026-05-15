"""
config.py — Central configuration for Parkinson's Detection project.
"""
import os

ROOT_DIR           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR       = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
DATA_FILE          = os.path.join(DATA_RAW_DIR, "parkinsons.data")
MODELS_DIR         = os.path.join(ROOT_DIR, "models")
MODEL_PATH         = os.path.join(MODELS_DIR, "model.pkl")
SCALER_PATH        = os.path.join(MODELS_DIR, "scaler.pkl")
SELECTOR_PATH      = os.path.join(MODELS_DIR, "selector.pkl")
META_PATH          = os.path.join(MODELS_DIR, "model_meta.json")

# Original 22 MDVP features (in dataset column order)
ORIGINAL_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

# All 29 features after engineering (22 original + 7 derived)
ENGINEERED_FEATURES = ORIGINAL_FEATURES + [
    "PPE_RPDE_sum",         # PPE + RPDE combined entropy signal
    "spread_range",         # spread2 - spread1  (nonlinear range)
    "Fo_range",             # Fhi - Flo  (fundamental frequency range)
    "Jitter_total",         # Jitter(%) + RAP + PPQ
    "Shimmer_total",        # Shimmer + APQ3 + APQ5
    "nonlinear_composite",  # PPE × RPDE × DFA
    "HNR_NHR_diff",         # HNR - NHR×10 (signal quality proxy)
]

# 18 features selected by mutual information after engineering
SELECTED_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:Shimmer(dB)",
    "Shimmer:APQ5", "MDVP:APQ", "NHR", "HNR",
    "spread1", "spread2", "PPE",
    "PPE_RPDE_sum", "spread_range", "Shimmer_total",
    "nonlinear_composite", "HNR_NHR_diff"
]

# Full metadata for all 22 original features (min, max, mean, description, step)
FEATURE_META = {
    "MDVP:Fo(Hz)":       (88.33,  260.11, 154.23, "Average Vocal Fundamental Frequency (Hz)",      0.1),
    "MDVP:Fhi(Hz)":      (102.15, 592.03, 197.10, "Maximum Vocal Fundamental Frequency (Hz)",      0.1),
    "MDVP:Flo(Hz)":      (65.48,  239.17, 116.32, "Minimum Vocal Fundamental Frequency (Hz)",      0.1),
    "MDVP:Jitter(%)":    (0.00168,0.03316,0.00622,"Jitter — Cycle-to-cycle pitch variation (%)",   0.0001),
    "MDVP:Jitter(Abs)":  (7e-6,   0.00026,4.4e-5, "Jitter — Absolute pitch period variation",      1e-6),
    "MDVP:RAP":          (0.00068,0.02144,0.00331,"Relative Average Perturbation",                  0.0001),
    "MDVP:PPQ":          (0.00092,0.01958,0.00345,"5-point Period Perturbation Quotient",           0.0001),
    "Jitter:DDP":        (0.00204,0.06433,0.00992,"Average absolute difference of jitter periods",  0.0001),
    "MDVP:Shimmer":      (0.00954,0.11908,0.02971,"Shimmer — Amplitude variation",                  0.0001),
    "MDVP:Shimmer(dB)":  (0.085,  1.302,  0.282,  "Shimmer in decibels",                           0.001),
    "Shimmer:APQ3":      (0.00455,0.05647,0.01566,"3-point Amplitude Perturbation Quotient",        0.0001),
    "Shimmer:APQ5":      (0.0057, 0.0794, 0.01788,"5-point Amplitude Perturbation Quotient",        0.0001),
    "MDVP:APQ":          (0.00719,0.13778,0.02408,"11-point Amplitude Perturbation Quotient",       0.0001),
    "Shimmer:DDA":       (0.01364,0.16942,0.04699,"Average absolute shimmer differences",           0.0001),
    "NHR":               (0.00065,0.31482,0.02485,"Noise-to-Harmonics Ratio",                      0.0001),
    "HNR":               (8.441,  33.047, 21.886, "Harmonics-to-Noise Ratio",                      0.001),
    "RPDE":              (0.25657,0.68515,0.49854,"Recurrence Period Density Entropy",              0.0001),
    "DFA":               (0.57428,0.82529,0.71810,"Detrended Fluctuation Analysis",                 0.0001),
    "spread1":           (-7.965, -2.434, -5.684, "Nonlinear fundamental freq variation (spread1)", 0.001),
    "spread2":           (0.00627,0.45049,0.22651,"Nonlinear fundamental freq variation (spread2)", 0.0001),
    "D2":                (1.423,  3.671,  2.382,  "Correlation Dimension",                          0.001),
    "PPE":               (0.04454,0.52737,0.20655,"Pitch Period Entropy ★ (most discriminative)",   0.0001),
}
