# Early Detection of Parkinson's Disease
### Using Voice Telemonitoring & Machine Learning

**MCA Final Year Project** — Shivam Kothiyal | Roll No: 241347080031  
Department of CS & IT, H.N.B. Garhwal University (A Central University), Srinagar, Uttarakhand  
Under the guidance of **Prof. Y.P. Raiwani**

---

## 📌 Project Overview

This project builds an AI-powered, non-invasive screening tool for **early detection of Parkinson's Disease** using biomedical voice measurements. 

### 🌟 New in v2: Live Voice Diagnostic
The latest version integrates **clinical-grade audio processing** using the Praat algorithm (via `parselmouth`). Users can now record their voice directly in the browser or upload a WAV file to get an instant diagnostic risk assessment.

> 90% of Parkinson's patients develop voice changes *years* before physical symptoms appear — making voice the earliest available biomarker.

---

## 🏗️ Project Structure

```
Parkinson_Prediction/
├── models/
│   ├── model.pkl                  # Voting Ensemble (GB + RF + SVM)
│   ├── scaler.pkl                 # Fitted MinMaxScaler
│   └── selector.pkl               # SelectKBest (MI) feature mask
├── src/
│   ├── audio_processing.py        # [NEW] DSP pipeline for voice standardization
│   ├── extractor.py               # [NEW] Praat/Parselmouth feature extraction
│   ├── preprocessing.py           # Feature engineering (7 new clinical metrics)
│   ├── train.py                   # Ensemble training + grid search
│   └── predict.py                 # Cross-module inference logic
├── webapp/
│   └── optimized_app.py           # Glassmorphism Streamlit UI
├── requirements.txt               # Updated with parselmouth, fpdf, librosa
└── README.md
```

---

## 🚀 Key Features

- **🎤 Live Phonation Analysis**: Record a sustained "Ahhh" and extract 22 MDVP biomarkers in real-time.
- **📄 Clinical PDF Reports**: Generate professional screening reports with risk probability and biomarker breakdown.
- **🧠 Voting Ensemble Model**: Combines Gradient Boosting, Random Forest, and SVM for 97%+ Recall.
- **🎨 Premium UI/UX**: Professional dark-mode dashboard with interactive Plotly gauges and glassmorphism cards.
- **📂 Batch Processing**: Analyze entire clinics of patient data via CSV upload.

---

## ⚙️ Setup & Installation

### 1. Clone the project
```bash
git clone https://github.com/ravarnax/Parkinson_Prediction_v2.git
cd Parkinson_Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
*Note: Clinical audio extraction requires `praat-parselmouth` and `librosa`.*

---

## 📊 Model Performance (v2 Upgrade)

| Metric | v1 (Single Model) | v2 (Voting Ensemble) |
|---|---|---|
| **CV Recall (Sensitivity)** | 96.57% | **97.29%** |
| **CV Accuracy** | 92.31% | **94.87%** |
| **CV AUC-ROC** | 96.49% | **97.17%** |

> **Recall is our Priority**: In medical screening, missing a positive case (False Negative) is critical. Our ensemble is tuned to maximize Recall to ensure early detection.

---

## 📚 References

1. M.A. Little et al. — *Suitability of Dysphonia Measurements for Telemonitoring of Parkinson's Disease*, IEEE Trans. Biomed. Eng., 2009.
2. UCI ML Repository — Parkinson's Disease Data Set (Oxford University).
3. Parselmouth: Praat in Python — [Documentation](https://parselmouth.readthedocs.io/).

---

*Lead Developer: **Shivam Kothiyal** (MCA Final Year)*  
*Department of CS & IT, H.N.B. Garhwal University*

⚠️ **Disclaimer**: This tool is for educational and screening purposes only. It does NOT provide a clinical diagnosis. Consult a neurologist for medical evaluations.
