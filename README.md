# Early Detection of Parkinson's Disease
### Using Voice Telemonitoring & Machine Learning

**MCA Final Year Project** — Shivam Kothiyal | Roll No: 241347080031  
Department of CS & IT, H.N.B. Garhwal University (A Central University), Srinagar, Uttarakhand  
Under the guidance of **Prof. Y.P. Raiwani**

---

## 📌 Project Overview

This project builds an AI-powered, non-invasive screening tool for **early detection of Parkinson's Disease** using biomedical voice measurements. The model extracts 22 MDVP (Multi-Dimensional Voice Program) features from voice recordings and predicts whether a patient is at risk.

> 90% of Parkinson's patients develop voice changes *years* before physical symptoms appear — making voice the earliest available biomarker.

---

## 🏗️ Project Structure

```
Parkinson_Prediction/
├── data/
│   ├── raw/
│   │   ├── parkinsons.data        # UCI Oxford dataset (195 samples)
│   │   └── download_data.py       # Script to re-download the dataset
│   └── processed/                 # Saved after preprocessing
├── models/
│   ├── model.pkl                  # Trained GradientBoosting model
│   └── scaler.pkl                 # Fitted MinMaxScaler (-1 to 1)
├── notebooks/
│   └── EDA.ipynb                  # Exploratory Data Analysis notebook
├── src/
│   ├── config.py                  # Central path configuration
│   ├── preprocessing.py           # Data loading & preprocessing
│   ├── train.py                   # Model training + cross-validation
│   └── predict.py                 # Reusable prediction module
├── webapp/
│   └── optimized_app.py           # Streamlit web application
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone / unzip the project
```bash
cd Parkinson_Prediction
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Train the model
```bash
python src/train.py
```
This will:
- Load and preprocess the dataset
- Apply stratified 80/20 train-test split
- Train a GradientBoostingClassifier with tuned hyperparameters
- Run 5-fold stratified cross-validation
- Save `models/model.pkl` and `models/scaler.pkl`

### Launch the web application
```bash
streamlit run webapp/optimized_app.py
```
Open your browser at `http://localhost:8501`

### Run predictions from code
```python
from src.predict import predict_single, predict_batch
import pandas as pd

# Single patient prediction
features = {
    'MDVP:Fo(Hz)': 119.992, 'MDVP:Fhi(Hz)': 157.302,
    # ... all 22 features
}
result = predict_single(features)
print(result)  # {'prediction': 'Parkinson\'s', 'confidence': 0.94}

# Batch from CSV
results = predict_batch('path/to/patients.csv')
print(results)
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | UCI ML Repository — Oxford University (Max Little et al.) |
| Recordings | 195 voice samples from 31 subjects |
| PD Patients | 147 (75.4%) |
| Healthy | 48 (24.6%) |
| Features | 22 MDVP biomedical voice measurements |
| Target | `status` — 1 = Parkinson's, 0 = Healthy |

---

## 🤖 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 94.87% |
| Recall (PD) | 96.55% |
| Precision (PD) | 96.55% |
| F1-Score | 96.55% |
| 5-Fold CV Recall | 96.57% ± 3.8% |

> **Why Recall is the key metric:** A False Negative (missing a PD patient) is medically catastrophic. Recall measures how many actual PD cases are correctly detected.

---

## 📚 References

1. M.A. Little et al. — *Suitability of Dysphonia Measurements for Telemonitoring of Parkinson's Disease*, IEEE Trans. Biomed. Eng., 2009
2. UCI ML Repository — Parkinson's Disease Data Set
3. Scikit-learn: Gradient Boosting Classifier documentation

---

*Connect on LinkedIn: [linkedin.com/in/shivam-kothiyal-a07201195](https://www.linkedin.com/in/shivam-kothiyal-a07201195)*

---

## 🚀 v2 Improvements (Accuracy Upgrades)

| Improvement | Details |
|---|---|
| Feature Engineering | 7 new derived features (PPE_RPDE_sum, spread_range, nonlinear_composite, etc.) |
| Feature Selection | Mutual Information SelectKBest: 29 → 18 best features |
| Voting Ensemble | GradientBoosting (×2) + RandomForest + SVM + ExtraTrees |
| Hyperparameter Tuning | GridSearchCV on GradientBoosting (best: depth=5, lr=0.03, n=200) |
| Class Handling | Stratified split + balanced weights across ensemble |

### Performance: v1 vs v2

| Metric | v1 (single model) | v2 (ensemble + engineering) |
|---|---|---|
| CV Accuracy | 92.31% | 92.31% |
| CV Recall | 96.57% | 97.29% |
| CV F1-Score | 94.98% | 95.08% |
| CV AUC-ROC | 96.49% | 97.17% |
