"""
train.py - Leak-Free Model Training Pipeline.
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier, VotingClassifier,
                               RandomForestClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix)
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (DATA_FILE, MODEL_PATH, SCALER_PATH, SELECTOR_PATH,
                    META_PATH, MODELS_DIR, ORIGINAL_FEATURES)
from preprocessing import load_data, engineer_features

def build_ensemble():
    # 1. Regularized Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=150, max_depth=5, min_samples_split=10, 
        min_samples_leaf=4, max_features='sqrt', random_state=42
    )
    # 2. Regularized Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3, 
        subsample=0.8, random_state=42
    )
    # 3. Regularized SVM
    svm_model = SVC(
        C=0.5, kernel='rbf', probability=True, 
        class_weight='balanced', random_state=42
    )
    # 4. Regularized Extra Trees
    et_model = ExtraTreesClassifier(
        n_estimators=100, max_depth=5, min_samples_split=10, 
        min_samples_leaf=4, random_state=42
    )

    return VotingClassifier(
        estimators=[('gb', gb_model), ('rf', rf_model), ('svm', svm_model), ('et', et_model)],
        voting='soft'
    )

def train_model():
    print("=" * 60)
    print("  Parkinson's Detection - Unified Leak-Free Pipeline")
    print("=" * 60)

    # 1. Load & Engineer Features (Row-wise math does not leak data)
    df = load_data(DATA_FILE)
    if df is None:
        return
    
    df2 = df.drop(["name"], axis=1) if "name" in df.columns else df.copy()
    X_all = df2.drop(["status"], axis=1)[ORIGINAL_FEATURES]
    y_all = df2["status"]
    
    X_engineered = engineer_features(X_all)

    # 2. Split for final holdout test
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    # 3. Define the Master Pipeline (Matches Dissertation Exactly)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(score_func=mutual_info_classif, k=18)),
        ('classifier', build_ensemble())
    ])

    # 4. 5-Fold Stratified Cross-Validation
    print("\nExecuting 5-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_res = cross_validate(
        pipeline, X_engineered, y_all, cv=cv, 
        scoring=['accuracy', 'recall', 'precision', 'f1', 'roc_auc']
    )

    cv_scores = {}
    print("-" * 50)
    for key, label in zip(['test_accuracy', 'test_recall', 'test_precision', 'test_f1', 'test_roc_auc'], 
                          ['Accuracy ', 'Recall   ', 'Precision', 'F1-Score ', 'AUC-ROC  ']):
        v = cv_res[key] * 100
        cv_scores[label.strip()] = round(float(v.mean()), 2)
        print(f"  {label}: mean={v.mean():.2f}% +/- {v.std():.2f}%")
    print("-" * 50)

    # 5. Train Final Pipeline on Train Set & Evaluate on Test Set
    print("\nTraining Final Pipeline on 80% Train Split...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n  TEST SET RESULTS (20% held-out)")
    print("-" * 50)
    print(f"  Recall    : {recall_score(y_test, y_pred)*100:.2f}%")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson's"]))

    # 6. Extract Fitted Components for Streamlit Export
    fitted_scaler = pipeline.named_steps['scaler']
    fitted_selector = pipeline.named_steps['selector']
    fitted_model = pipeline.named_steps['classifier']

    # 7. Save Artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(fitted_model, MODEL_PATH)
    joblib.dump(fitted_scaler, SCALER_PATH)
    joblib.dump(fitted_selector, SELECTOR_PATH)

    selected_feature_names = [list(X_engineered.columns)[i] for i in fitted_selector.get_support(indices=True)]

    meta = {
        "model_type": "VotingClassifier (GB + RF + SVM + ExtraTrees)",
        "n_original_features": 22,
        "n_engineered_features": len(X_engineered.columns),
        "n_selected_features": 18,
        "engineered_features": list(X_engineered.columns),
        "selected_features": selected_feature_names,
        "cv_scores": cv_scores,
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    print("\nArtifacts successfully exported. Pipeline integrity secured.")

if __name__ == "__main__":
    train_model()







    