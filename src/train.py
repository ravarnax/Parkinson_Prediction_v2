"""
train.py - Improved model training pipeline.

Improvements over v1:
  - Feature engineering (7 new derived features)
  - Mutual information feature selection (29 -> 18 features)
  - Voting Ensemble: GradientBoosting + RandomForest + SVM + ExtraTrees
  - Tuned hyperparameters via GridSearchCV
  - 5-fold stratified cross-validation for honest evaluation
  - Model saved with joblib (faster + safer than pickle)

Run from project root:
    python src/train.py
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier, VotingClassifier,
                               RandomForestClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix)
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (DATA_FILE, MODEL_PATH, SCALER_PATH, SELECTOR_PATH,
                    META_PATH, MODELS_DIR, ENGINEERED_FEATURES)
from preprocessing import load_data, preprocess_data, engineer_features


def build_ensemble():
    """
    Voting Ensemble - 4 diverse models with soft probability voting.
    Regularized parameters are used to prevent overfitting.
    """
    # 1. Regularized Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,               # CONSTRAINT: Stops trees from growing too deep and memorizing
        min_samples_split=10,      # CONSTRAINT: Requires at least 10 samples to split a node
        min_samples_leaf=4,        # CONSTRAINT: Forces leaf nodes to be general, not specific to 1 patient
        max_features='sqrt',       # Prevents reliance on a single dominant feature
        random_state=42
    )

    # 2. Regularized Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,        # CONSTRAINT: Slower learning rate prevents aggressive fitting
        max_depth=3,               # CONSTRAINT: Keeps the individual trees very shallow
        subsample=0.8,             # CONSTRAINT: Stochastic boosting (uses only 80% of data per tree)
        random_state=42
    )

    # 3. Regularized SVM
    svm_model = SVC(
        C=0.5,                     # CONSTRAINT: Lower C creates a 'softer margin' (allows some errors to prevent overfitting)
        kernel='rbf',
        probability=True,
        class_weight='balanced',   # Handles the 75/25 PD to Healthy imbalance
        random_state=42
    )

    # 4. Regularized Extra Trees
    et_model = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=5,               # CONSTRAINT: Restricted depth
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    )

    # 5. The Generalized Ensemble
    return VotingClassifier(
        estimators=[
            ('gb', gb_model),
            ('rf', rf_model),
            ('svm', svm_model),
            ('et', et_model)
        ],
        voting='soft'
    )


def train_model():
    print("=" * 60)
    print("  Parkinson's Detection - Improved Training Pipeline")
    print("=" * 60)

    # 1. Load
    df = load_data(DATA_FILE)
    if df is None:
        print("Run data/raw/download_data.py to fetch the dataset.")
        return

    # 2. Preprocess (engineer + stratify + scale + select)
    print()
    X_train, X_test, y_train, y_test, scaler, selector = preprocess_data(df)

    # 3. Build ensemble
    model = build_ensemble()
    print("\nTraining Voting Ensemble (GB + RF + SVM + ExtraTrees)...")
    model.fit(X_train, y_train)

    # 4. Test set evaluation
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "-" * 50)
    print("  TEST SET RESULTS  (stratified 20% held-out)")
    print("-" * 50)
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"  Recall    : {recall_score(y_test, y_pred)*100:.2f}%  <- medical priority")
    print(f"  Precision : {precision_score(y_test, y_pred)*100:.2f}%")
    print(f"  F1-Score  : {f1_score(y_test, y_pred)*100:.2f}%")
    print(f"  AUC-ROC   : {roc_auc_score(y_test, y_proba)*100:.2f}%")
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson's"]))

    # 5. 5-fold cross-validation (full dataset)
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from config import ORIGINAL_FEATURES

    df2 = df.drop(["name"], axis=1) if "name" in df.columns else df.copy()
    X_all = df2.drop(["status"], axis=1)[ORIGINAL_FEATURES]
    y_all = df2["status"]
    X_all_eng = engineer_features(X_all)
    sc2  = MinMaxScaler((-1, 1))
    X_sc = sc2.fit_transform(X_all_eng)
    sel2 = SelectKBest(mutual_info_classif, k=18)
    X_cv = sel2.fit_transform(X_sc, y_all)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_res = cross_validate(model, X_cv, y_all, cv=cv,
        scoring=['accuracy', 'recall', 'precision', 'f1', 'roc_auc'])

    print("-" * 50)
    print("  5-FOLD STRATIFIED CROSS-VALIDATION")
    print("-" * 50)
    metric_map = {
        'test_accuracy' : 'Accuracy ',
        'test_recall'   : 'Recall   ',
        'test_precision': 'Precision',
        'test_f1'       : 'F1-Score ',
        'test_roc_auc'  : 'AUC-ROC  ',
    }
    cv_scores = {}
    for key, label in metric_map.items():
        v = cv_res[key] * 100
        cv_scores[label.strip()] = round(float(v.mean()), 2)
        print(f"  {label}: {np.round(v,2)}  mean={v.mean():.2f}% +/- {v.std():.2f}%")

    # 6. Save all artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model,    MODEL_PATH)
    joblib.dump(scaler,   SCALER_PATH)
    joblib.dump(selector, SELECTOR_PATH)

    meta = {
        "model_type"          : "VotingClassifier (GB + RF + SVM + ExtraTrees)",
        "n_original_features" : 22,
        "n_engineered_features": len(list(X_all_eng.columns)),
        "n_selected_features" : 18,
        "engineered_features" : list(X_all_eng.columns),
        "selected_features"   : [list(X_all_eng.columns)[i]
                                  for i in sel2.get_support(indices=True)],
        "cv_scores"           : cv_scores,
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nmodel.pkl    -> {MODEL_PATH}")
    print(f"scaler.pkl   -> {SCALER_PATH}")
    print(f"selector.pkl -> {SELECTOR_PATH}")
    print(f"model_meta.json -> {META_PATH}")
    print("\nDone. Launch with:  streamlit run webapp/optimized_app.py")


if __name__ == "__main__":
    train_model()
