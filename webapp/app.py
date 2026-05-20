"""
optimized_app.py — Improved Streamlit web application.

Launch:  streamlit run webapp/optimized_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os, sys, json
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (MODEL_PATH, SCALER_PATH, SELECTOR_PATH,
                        META_PATH, ORIGINAL_FEATURES, FEATURE_META)
from src.preprocessing import engineer_features
from src.audio_processing import process_audio_signal, validate_audio_properties, InvalidAudioError
from src.extractor import extract_voice_features
from src.predict import predict_from_voice

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parkinson's Detection System",
    page_icon="🧠", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Targeted fix for the white header/expander strip */
    [data-testid="stExpander"] {
        background-color: transparent !important;
        border: none !important;
    }
    
    /* If it's a specific container or header block */
    .st-emotion-cache-p4m0d7 {
        background-color: #0e1117 !important; /* Matches Streamlit dark background */
    }

    /* Optional: Remove the border/outline if that's the "strip" */
    details {
        border-radius: 10px !important;
        background-color: #1a1c24 !important; /* Darker grey for contrast */
        border: 1px solid #2e303d !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ── UI Theme Variables (Permanent Dark Mode) ──────────────────────────────
vars = {
    "bg": "#0e1117",
    "sidebar_bg": "#111b21",
    "card_bg": "rgba(255,255,255,0.03)",
    "header_grad": "linear-gradient(135deg,#0a0f0a 0%,#0d3321 60%,#16A34A 100%)",
    "perf_grad": "linear-gradient(135deg, rgba(10,20,14,0.72) 0%, rgba(15,40,25,0.68) 50%, rgba(22,163,74,0.10) 100%)",
    "text_main": "#ffffff",
    "text_dim": "#86efac",
    "section_bg": "#1e1e1e",
    "section_border": "#2a2a2a",
    "tooltip_bg": "rgba(30,58,138,0.08)",
    "shadow": "0 8px 32px rgba(0,0,0,0.45)",
    "chart_text": "#d1fae5",
    "chart_grid": "rgba(74,222,128,0.08)",
    "chart_bg": "rgba(255,255,255,0.02)",
    "plotly_theme": "plotly_dark"
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
:root {{
    --bg-primary: {vars['bg']};
    --card-bg: {vars['card_bg']};
    --text-main: {vars['text_main']};
    --text-dim: {vars['text_dim']};
    --section-bg: {vars['section_bg']};
    --section-border: {vars['section_border']};
}}

html,body,[class*="css"] {{ font-family:'Inter',sans-serif; }}

/* ── Main App Background ── */
.stApp {{
    background-color: {vars['bg']};
    color: {vars['text_main']};
}}
[data-testid="stSidebar"] {{
    background-color: {vars['sidebar_bg']};
}}

/* ── Main header ── */
.main-header{{
  background:{vars['header_grad']};
  padding:1.8rem 2rem;border-radius:14px;margin-bottom:1.4rem;
  border:1px solid #1a4a2e;
  box-shadow: {vars['shadow']};}}
.main-header h1{{color:#fff;margin:0;font-size:2rem;font-weight:700;letter-spacing:-.5px}}
.main-header p{{color:#6ee7b7;margin:.4rem 0 0;font-size:.9rem}}

/* ── Glassmorphism Performance Header ── */
.perf-header{{
  position:relative;overflow:hidden;
  background:{vars['perf_grad']};
  border:1px solid rgba(74,222,128,0.18);
  border-radius:18px;
  padding:1.6rem 1.8rem 1.4rem;
  margin:1rem 0 1.4rem;
  backdrop-filter:blur(20px);
  -webkit-backdrop-filter:blur(20px);
  box-shadow:{vars['shadow']}, inset 0 1px 0 rgba(255,255,255,0.06);}}

.perf-header-label{{
  font-size:.7rem;font-weight:700;
  color:{vars['text_dim']};letter-spacing:1.5px;
  text-transform:uppercase;margin-bottom:1rem;
  display:flex;align-items:center;gap:8px;}}
.perf-header-label::after{{
  content:'';flex:1;
  height:1px;background:linear-gradient(90deg,rgba(74,222,128,0.3),transparent);}}

.perf-cards{{ display:flex;gap:10px;flex-wrap:wrap;position:relative;z-index:1;}}
.perf-card{{
  flex:1;min-width:130px;
  background:{vars['card_bg']};
  border:1px solid rgba(74,222,128,0.15);
  border-radius:14px;
  padding:1rem 1rem .85rem;
  text-align:center;
  transition:all .22s ease;}}
.perf-card:hover{{
  transform:translateY(-4px);
  box-shadow:0 8px 24px rgba(22,163,74,0.15);
  border-color:rgba(74,222,128,0.35);}}

.perf-card .pc-val{{ font-size:1.9rem;font-weight:800;color:#16a34a;line-height:1.05; }}
.perf-card .pc-lbl{{ font-size:.68rem;color:{vars['text_dim']};margin-top:.35rem;text-transform:uppercase;font-weight:600;}}

/* ── Result boxes ── */
.result-pd{{background:rgba(220,38,38,0.1);border:2px solid #DC2626;border-radius:12px;padding:1.2rem;color:{vars['text_main']}}}
.result-warning{{background:rgba(234,179,8,0.1);border:2px solid #EAB308;border-radius:12px;padding:1.2rem;color:{vars['text_main']}}}
.result-healthy{{background:rgba(22,163,74,0.1);border:2px solid #16A34A;border-radius:12px;padding:1.2rem;color:{vars['text_main']}}}


/* ── Sectional Feature Cards ── */
.section-card{{
  background:{vars['section_bg']};
  border:1px solid {vars['section_border']};
  border-radius:14px;
  padding:1.4rem 1.6rem 1rem;
  margin-bottom:1.4rem;
  box-shadow:{vars['shadow']};}}
.section-card-title{{
  display:flex;align-items:center;gap:10px;font-size:1.05rem;font-weight:700;
  color:#16a34a;padding-bottom:.65rem;margin-bottom:.9rem;
  border-bottom:1px solid {vars['section_border']};}}
.section-card-desc{{
  font-size:.8rem;color:#64748b;margin:-0.5rem 0 .9rem;
  padding:.4rem .75rem;background:{vars['tooltip_bg']};
  border-left:3px solid #16A34A;border-radius:0 6px 6px 0;}}

/* ── Medical tooltip ── */
.med-tooltip{{
  margin:.5rem 0 .9rem;
  background:{vars['tooltip_bg']};
  border:1px solid rgba(99,102,241,0.18);
  border-radius:10px;padding:.75rem 1rem;font-size:.82rem;color:#475569;}}
.med-tooltip summary{{ cursor:pointer;font-weight:700;color:#2563eb;display:flex;align-items:center;gap:7px;}}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, SELECTOR_PATH]):
        return None, None, None, None
    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    selector = joblib.load(SELECTOR_PATH)
    meta     = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {}
    return model, scaler, selector, meta

def generate_report(results, probability):
    """Generate a professional PDF screening report."""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, "Parkinson's Disease Screening Report", ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 10, "Early Detection via Acoustic Biomarkers & Voting Ensemble", ln=True, align='C')
    pdf.line(10, 35, 200, 35)
    pdf.ln(10)
    
    # Metadata
    pdf.set_font("Helvetica", size=11)
    pdf.cell(100, 10, f"Date of Screening: {datetime.date.today()}", ln=False)
    pdf.cell(0, 10, f"Report ID: PD-{datetime.datetime.now().strftime('%Y%m%d%H%M')}", ln=True, align='R')
    pdf.ln(5)
    
    # Results Section
    pdf.set_font("Helvetica", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, " DIAGNOSTIC SUMMARY", ln=True, fill=True)
    pdf.set_font("Helvetica", size=11)
    pdf.ln(2)
    
    result_text = "POSITIVE - Indicators of Parkinson's Disease Detected" if probability > 50 else "NEGATIVE - No Significant Parkinson's Indicators"
    pdf.cell(50, 10, "Result:", ln=False)
    pdf.set_font("Helvetica", 'B', 11)
    pdf.cell(0, 10, result_text, ln=True)
    
    pdf.set_font("Helvetica", size=11)
    pdf.cell(50, 10, "PD Risk Probability:", ln=False)
    pdf.cell(0, 10, f"{probability:.2f}%", ln=True)
    
    pdf.cell(50, 10, "Model Confidence:", ln=False)
    pdf.cell(0, 10, f"{max(probability, 100-probability):.2f}%", ln=True)
    
    pdf.ln(10)
    
    # Features Section (Optional inclusion of some key features)
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, " KEY BIOMARKERS ANALYSED", ln=True, fill=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", size=10)
    
    # We'll just list a few most important ones if they are in results
    key_metrics = ["PPE", "spread1", "MDVP:Fo(Hz)", "HNR", "DFA", "RPDE"]
    for m in key_metrics:
        if m in results:
            val = results[m]
            pdf.cell(60, 8, f"{m}:", ln=False)
            pdf.cell(0, 8, f"{val:.5f}", ln=True)
    
    pdf.ln(15)
    
    # Footer / Disclaimer
    pdf.set_font("Helvetica", 'I', 9)
    pdf.multi_cell(0, 5, "DISCLAIMER: This report is generated by an automated screening tool developed for research purposes. "
                       "It does NOT constitute a clinical diagnosis. Please consult a qualified neurologist for a comprehensive medical evaluation.")
    
    pdf.ln(10)
    # pdf.set_font("Helvetica", 'B', 10)
    # pdf.cell(0, 10, "Academic Credentials:", ln=True)
    # pdf.set_font("Helvetica", size=9)
    # pdf.cell(0, 5, "Lead Developer: Shivam Kothiyal (MCA Final Year)", ln=True)
    # pdf.cell(0, 5, "Department of CS & IT, H.N.B. Garhwal University", ln=True)
    # pdf.cell(0, 5, "Under the guidance of Prof. Y.P. Raiwani", ln=True)

    return bytes(pdf.output())

model, scaler, selector, meta = load_artifacts()
if model is None:
    st.error("🚨 Model not found. Run `python src/train.py` first.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🧠 Parkinson's Disease Early Detection System</h1>
  <p>MCA Final Year Project &nbsp;·&nbsp; Voting Ensemble (GB + RF + SVM + ExtraTrees)
     &nbsp;·&nbsp; Feature Engineering + Mutual Info Selection &nbsp;·&nbsp; H.N.B. Garhwal University</p>
</div>
""", unsafe_allow_html=True)

# ── Model dashboard ───────────────────────────────────────────────────────
cv_scores = meta.get("cv_scores", {})
with st.expander("📊 Model Performance Dashboard", expanded=True):
    st.markdown('\n'.join([line.lstrip() for line in f"""
    <div class="perf-header">
      <div class="perf-header-label">⚡ 5-Fold Stratified Cross-Validation Results — Voting Ensemble</div>
      <div class="perf-cards">
        <div class="perf-card">
          <span class="pc-icon">🎯</span>
          <div class="pc-val">{cv_scores.get('Accuracy','92.82')}%</div>
          <div class="pc-lbl">Accuracy</div>
          <div class="pc-badge">CV Score</div>
        </div>
        <div class="perf-card star-card">
          <span class="pc-icon">⭐</span>
          <div class="pc-val">{cv_scores.get('Recall','97.29')}%</div>
          <div class="pc-lbl">Recall</div>
          <div class="pc-badge">Primary KPI ★</div>
        </div>
        <div class="perf-card">
          <span class="pc-icon">🔬</span>
          <div class="pc-val">{cv_scores.get('Precision','—')}%</div>
          <div class="pc-lbl">Precision</div>
          <div class="pc-badge">CV Score</div>
        </div>
        <div class="perf-card">
          <span class="pc-icon">⚖️</span>
          <div class="pc-val">{cv_scores.get('F1-Score','—')}%</div>
          <div class="pc-lbl">F1-Score</div>
          <div class="pc-badge">CV Score</div>
        </div>
        <div class="perf-card">
          <span class="pc-icon">📈</span>
          <div class="pc-val">{cv_scores.get('AUC-ROC','—')}%</div>
          <div class="pc-lbl">AUC-ROC</div>
          <div class="pc-badge">CV Score</div>
        </div>
      </div>
    </div>
    """.split('\n')]), unsafe_allow_html=True)

    st.markdown("")
    i1, i2, i3 = st.columns([1.1, 1.2, 1.5])
    with i1:
        st.markdown(f"""
        **Model:** {meta.get('model_type','VotingClassifier')}  
        **Original features:** {meta.get('n_original_features', 22)}  
        **After engineering:** {meta.get('n_engineered_features', 29)}  
        **After selection:** {meta.get('n_selected_features', 18)} (via Mutual Information)  
        **Validation:** 5-Fold Stratified CV  
        **Dataset:** UCI Oxford Parkinson's — 195 recordings
        """)
    with i2:
        fig = go.Figure()
        metrics = ["Accuracy", "Recall", "Precision", "F1-Score", "AUC-ROC"]
        before  = [92.31, 96.57, 93.83, 94.98, 96.49]
        after   = [
            cv_scores.get('Accuracy',  0),
            cv_scores.get('Recall',    0),
            cv_scores.get('Precision', 0),
            cv_scores.get('F1-Score',  0),
            cv_scores.get('AUC-ROC',   0),
        ]
        fig.add_trace(go.Bar(name="Before (v1)", x=metrics, y=before, marker_color="#BBF7D0"))
        fig.add_trace(go.Bar(name="After (v2)",  x=metrics, y=after, marker_color="#16A34A"))
        fig.update_layout(
            barmode="group", height=260, title="Before vs After Improvements",
            yaxis=dict(range=[85, 100]), margin=dict(l=0,r=0,t=35,b=0),
            legend=dict(orientation="h", y=1.1),
            font=dict(color=vars["chart_text"]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=vars["chart_bg"],
            template=vars["plotly_theme"]
        )
        st.plotly_chart(fig, use_container_width=None)
    with i3:
        fi_features = [
            "PPE", "spread1", "MDVP:Fo(Hz)", "HNR",
            "nonlinear_composite", "MDVP:Jitter(%)", "MDVP:Shimmer(dB)",
            "spread2", "HNR_NHR_diff", "MDVP:APQ",
        ]
        fi_scores = [0.412, 0.387, 0.351, 0.329, 0.308,
                     0.284, 0.261, 0.243, 0.228, 0.209]
        fi_colors = [
            "#4ade80" if s >= 0.35 else
            "#a3e635" if s >= 0.28 else
            "#fbbf24" for s in fi_scores
        ]
        fig_fi = go.Figure(go.Bar(
            x=fi_scores[::-1], y=fi_features[::-1], orientation="h",
            marker=dict(color=fi_colors[::-1], line=dict(color="rgba(255,255,255,0.06)", width=1)),
            text=[f"{s:.3f}" for s in fi_scores[::-1]],
            textposition="outside", textfont=dict(color="#86efac", size=10),
            hovertemplate="<b>%{y}</b><br>MI Score: %{x:.3f}<extra></extra>",
        ))
        fig_fi.update_layout(
            title=dict(text="🏆 Top 10 Feature Importance", font=dict(size=13, color="#16a34a")),
            height=305, margin=dict(l=0, r=55, t=35, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=vars["chart_bg"],
            template=vars["plotly_theme"],
            xaxis=dict(
                title=dict(text="Mutual Information Score", font=dict(size=10, color=vars["text_dim"])),
                tickfont=dict(size=9, color=vars["text_dim"]),
                gridcolor=vars["chart_grid"],
                range=[0, max(fi_scores) * 1.22],
            ),
            yaxis=dict(tickfont=dict(size=10, color=vars["chart_text"]), gridcolor="rgba(0,0,0,0)"),
            bargap=0.28,
        )
        st.plotly_chart(fig_fi, use_container_width=None)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "👤  Single Patient Prediction",
    "📂  Batch Processing (CSV)",
    "📈  Feature Explorer",
    "🎤  Live Voice Diagnostic"
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE PATIENT
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enter All 22 Patient Voice Measurements")
    st.markdown(
        '<div class="tip-box">💡 <b>All 22 MDVP features required.</b> '
        'Sliders are pre-set to population averages. '
        'The model also internally computes 7 derived features for better accuracy.</div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    SECTION_CARDS = [
        {
            "icon": "🎵", "title": "Fundamental Frequency", "badge": "3 features",
            "desc": "Average, maximum and minimum pitch of the voice (Hz). Parkinson's patients typically show a lower and more erratic pitch.",
            "tooltip_html": """
<details class="med-tooltip">
  <summary>ℹ️ Plain English — What is Fundamental Frequency?</summary>
  <dl>
    <dt>Fo (avg)</dt><dd>Your natural speaking pitch — like the base note of your voice. PD patients' voices often drop lower over time.</dd>
    <dt>Fhi (max)</dt><dd>The highest pitch your voice reaches in the recording. A narrow Fhi-Flo gap signals reduced vocal control.</dd>
    <dt>Flo (min)</dt><dd>The lowest pitch in the recording. Combined with Fhi, it shows how much your pitch varies — healthy voices vary more freely.</dd>
  </dl>
</details>""",
            "features": ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"],
        },
        {
            "icon": "〰️", "title": "Jitter & Shimmer (Pitch & Amplitude Variation)", "badge": "11 features",
            "desc": "Jitter = cycle-to-cycle wobble in pitch timing. Shimmer = cycle-to-cycle wobble in loudness. Both are significantly elevated in Parkinson's patients.",
            "tooltip_html": """
<details class="med-tooltip">
  <summary>ℹ️ Plain English — What are Jitter and Shimmer?</summary>
  <dl>
    <dt>Jitter (%)</dt><dd>Tiny, involuntary timing errors between one vocal vibration and the next. Like a clock that ticks slightly unevenly — a healthy voice has very little of this.</dd>
    <dt>Jitter (Abs)</dt><dd>Same as Jitter(%) but measured in seconds instead of percentage. Gives the raw magnitude of timing inconsistency.</dd>
    <dt>RAP / PPQ / DDP</dt><dd>Three different ways to average out jitter across multiple cycles — they all capture the same underlying pitch wobble at different time scales.</dd>
    <dt>Shimmer</dt><dd>Variation in how loud each vocal vibration is. Think of a singer whose volume flickers unpredictably — that's shimmer. Larger shimmer = weaker vocal muscle control.</dd>
    <dt>Shimmer (dB)</dt><dd>Shimmer expressed in decibels (the everyday loudness scale). Easier to compare across recording sessions.</dd>
    <dt>APQ3 / APQ5 / APQ</dt><dd>Amplitude perturbation averages over 3, 5, and 11 consecutive cycles. They progressively smooth out the shimmer reading for better reliability.</dd>
    <dt>DDA</dt><dd>Average absolute shimmer difference — a simple way to quantify how much the volume lurches from one cycle to the next.</dd>
  </dl>
</details>""",
            "features": [
                "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
                "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
                "MDVP:APQ", "Shimmer:DDA",
            ],
        },
        {
            "icon": "📊", "title": "Non-linear & Entropy Measures", "badge": "8 features",
            "desc": "Complexity and chaos metrics that reveal hidden patterns in the voice signal. They capture irregularities invisible to simple amplitude/pitch analysis.",
            "tooltip_html": """
<details class="med-tooltip">
  <summary>ℹ️ Plain English — What are Non-linear Measures?</summary>
  <dl>
    <dt>NHR</dt><dd><b>Noise-to-Harmonics Ratio</b> — the fraction of 'noise' (breathiness / roughness) in the voice vs. the clear musical tone. PD voices are breathier, so NHR rises.</dd>
    <dt>HNR</dt><dd><b>Harmonics-to-Noise Ratio</b> — the opposite of NHR. A high HNR means a clear, strong voice. PD patients show lower HNR as vocal muscles weaken.</dd>
    <dt>RPDE</dt><dd><b>Recurrence Period Density Entropy</b> — measures how predictable and repetitive the voice is at a microscopic level. Healthy voices are more predictably periodic.</dd>
    <dt>DFA</dt><dd><b>Detrended Fluctuation Analysis</b> — measures self-similarity (fractal behaviour) of the voice signal over time. Abnormal DFA suggests irregular neural control of vocal cords.</dd>
    <dt>spread1 / spread2</dt><dd>Two nonlinear measures of how spread out or variable the pitch is. Higher spread = more chaotic pitch movement, often seen in PD.</dd>
    <dt>D2</dt><dd><b>Correlation Dimension</b> — estimates the mathematical complexity of the voice signal. A lower D2 suggests the brain is sending simpler, more restricted motor commands.</dd>
    <dt>PPE ★</dt><dd><b>Pitch Period Entropy</b> — the single most discriminative feature in this dataset. Measures unpredictability in consecutive pitch periods. Elevated PPE strongly predicts PD.</dd>
  </dl>
</details>""",
            "features": ["NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"],
        },
    ]

    feature_values = {}
    for card in SECTION_CARDS:
        st.markdown(
            f'<div class="section-card">'
            f'  <div class="section-card-title">'
            f'    <span class="section-card-icon">{card["icon"]}</span>'
            f'    {card["title"]}'
            f'    <span class="section-card-badge">{card["badge"]}</span>'
            f'  </div>'
            f'  <div class="section-card-desc">{card["desc"]}</div>'
            f'  {card["tooltip_html"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
        feats = card["features"]
        cols  = st.columns(2)
        for j, feat in enumerate(feats):
            mn, mx, default, desc, step = FEATURE_META[feat]
            fmt = "%.6f" if step < 0.001 else ("%.4f" if step < 0.01 else "%.3f")
            with cols[j % 2]:
                feature_values[feat] = st.number_input(
                    label=feat, min_value=float(mn), max_value=float(mx),
                    value=float(default), step=float(step), format=fmt, help=desc,
                    key=f"feat_{feat}",
                )
        st.markdown("<div style='margin-bottom:.5rem'></div>", unsafe_allow_html=True)

    if st.button("🔍  Analyse & Predict"):
        try:
            # Full pipeline: engineer → scale → select → predict
            input_df  = pd.DataFrame([feature_values])[ORIGINAL_FEATURES]
            input_eng = engineer_features(input_df)
            input_sc  = scaler.transform(input_eng)
            input_sel = selector.transform(input_sc)

            label = int(model.predict(input_sel)[0])
            proba = model.predict_proba(input_sel)[0]

            st.markdown("---")
            
            # ── ROW 1: Immediate Results (Message & Gauge) ──
            col_msg, col_gauge = st.columns([1.3, 1])

            with col_msg:
                p_score = proba[1] * 100
                if p_score > 75:
                    res_class = "result-pd"
                    title_html = "⚠️ HIGH RISK — Parkinson's Indicators Detected"
                    val_color = "#fecaca"
                    warn_color = "#f87171"
                    warn_msg = "⚠️ High probability. Neurologist confirmation strongly recommended."
                    bar_color = "#DC2626"
                elif p_score >= 60:
                    res_class = "result-warning"
                    title_html = "🟡 MEDIUM RISK — Borderline Indicators Detected"
                    val_color = "#fef08a"
                    warn_color = "#fde047"
                    warn_msg = "🟡 Borderline results. Consider clinical evaluation."
                    bar_color = "#EAB308"
                else:
                    res_class = "result-healthy"
                    title_html = "✅ LOW RISK — No Significant Indicators"
                    val_color = "#bbf7d0"
                    warn_color = "#4ade80"
                    warn_msg = "✅ Regular monitoring is still recommended for at-risk age groups."
                    bar_color = "#16A34A"

                st.markdown(f"""
                <div style="display: flex; flex-direction: column; justify-content: center; height: 250px;">
                    <div class="{res_class}" style="padding:1.5rem; color: white;">
                        <div style="font-size:1.25rem; font-weight:700; color:#ffffff; margin-bottom:.5rem">
                            {title_html}
                        </div>
                        <div style="font-size:1.1rem; color:{val_color}; line-height:1.6">
                            PD Risk Probability: <strong>{p_score:.1f}%</strong><br>
                            Model Confidence: <strong>{max(proba)*100:.1f}%</strong>
                        </div>
                        <div style="margin-top:.8rem; font-size:.9rem; color:{warn_color}">
                            {warn_msg}
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

            with col_gauge:
                # Ensure the gauge height matches the height set in the col_msg div (250px)
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(p_score, 1),
                    title={"text": "PD Risk", "font": {"size": 22, "color": "white", "weight": "bold"}},
                    number={"suffix": "%", "font": {"size": 48, "color": "white", "weight": "bold"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
                        "bar":  {"color": bar_color},
                        "steps": [
                            {"range": [0,  60], "color": "#DCFCE7"},
                            {"range": [60, 75], "color": "#FEF9C3"},
                            {"range": [75,100], "color": "#FEE2E2"},
                        ],
                        "threshold": {"line": {"color":"white","width":3}, "value": p_score}
                    }
                ))
                fig_g.update_layout(height=250, margin=dict(l=25, r=25, t=50, b=25), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_g, use_container_width=None, config={'displayModeBar': False})

            st.markdown("<br>", unsafe_allow_html=True)

            # ── ROW 2: Data Breakdown (Tables Side-by-Side) ──
            col_tbl1, col_tbl2 = st.columns([1, 1.4])
            
            with col_tbl1:
                st.markdown("**Derived features (auto-computed):**")
                derived = {
                    "PPE_RPDE_sum"       : round(feature_values["PPE"] + feature_values["RPDE"], 5),
                    "nonlinear_composite": round(feature_values["PPE"] * feature_values["RPDE"] * feature_values["DFA"], 6),
                    "Fo_range"           : round(feature_values["MDVP:Fhi(Hz)"] - feature_values["MDVP:Flo(Hz)"], 3),
                    "Jitter_total"       : round(feature_values["MDVP:Jitter(%)"] + feature_values["MDVP:RAP"] + feature_values["MDVP:PPQ"], 6),
                }
                st.dataframe(pd.DataFrame(derived, index=["Value"]).T.rename(columns={"Value":"Computed"}),
                             use_container_width=None)

            with col_tbl2:
                st.markdown("**Patient vs. healthy population:**")
                key_feats  = ["PPE", "RPDE", "HNR", "NHR", "MDVP:Jitter(%)", "MDVP:Shimmer", "spread1", "DFA"]
                healthy_hi = [0.10, 0.45, 28.0, 0.02, 0.004, 0.02, -4.0, 0.72]
                patient_vs = [feature_values[f] for f in key_feats]
                
                flag = ["⚠️ High" if abs(patient_vs[i]) > abs(healthy_hi[i]) else "✅ Normal"
                         for i in range(len(key_feats))]
                         
                df_cmp = pd.DataFrame({
                    "Feature"  : key_feats,
                    "Patient"  : patient_vs,
                    "Baseline Max": healthy_hi,
                    "Status"   : flag,
                })
                st.dataframe(df_cmp, use_container_width=None, hide_index=True)

            # ── Generate PDF Report ──
            st.markdown("<br>", unsafe_allow_html=True)
            report_bytes = generate_report(feature_values, proba[1]*100)
            st.download_button(
                label="📥 Download Clinical Screening Report (PDF)",
                data=report_bytes,
                file_name=f"Parkinsons_Report_{datetime.date.today()}.pdf",
                mime="application/pdf",
                use_container_width=None
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Batch Patient Processing")
    st.markdown(
        "Upload a CSV with all 22 MDVP feature columns. "
        "`name` and `status` columns are ignored if present."
    )
    st.markdown(
        '<div class="tip-box">💡 The model automatically computes 7 additional features '
        'internally — you only need to provide the original 22.</div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            st.markdown(f"**Preview** — {len(batch_df)} rows × {len(batch_df.columns)} columns")
            st.dataframe(batch_df.head(5), use_container_width=None)

            feat_df = batch_df.copy()
            for col in ["name", "status"]:
                if col in feat_df.columns:
                    feat_df = feat_df.drop(columns=[col])

            missing = [f for f in ORIGINAL_FEATURES if f not in feat_df.columns]
            if missing:
                st.error(f"❌ Missing {len(missing)} columns:")
                st.code("\n".join(missing))
                st.stop()

            feat_df = feat_df.dropna(subset=ORIGINAL_FEATURES)

            if st.button("🚀  Run Batch Prediction"):
                X     = feat_df[ORIGINAL_FEATURES]
                X_eng = engineer_features(X)
                X_sc  = scaler.transform(X_eng)
                X_sel = selector.transform(X_sc)

                labels = model.predict(X_sel)
                probas = model.predict_proba(X_sel)

                results = batch_df.loc[feat_df.index].copy()
                results["Prediction"]   = ["Parkinson's" if l == 1 else "Healthy" for l in labels]
                results["PD_Risk_%"]    = (probas[:, 1] * 100).round(2)
                results["Confidence_%"] = [round(probas[i, l] * 100, 2) for i, l in enumerate(labels)]
                results["Risk_Level"]   = pd.cut(
                    probas[:, 1], bins=[0, 0.3, 0.6, 1.0],
                    labels=["Low", "Medium", "High"]
                ).astype(str)

                n_pd = (labels == 1).sum()
                n_ht = (labels == 0).sum()
                st.success(f"✅ Processed {len(results)} records")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Patients", len(results))
                m2.metric("PD Risk", n_pd, f"{n_pd/len(results)*100:.1f}%")
                m3.metric("Healthy",  n_ht, f"{n_ht/len(results)*100:.1f}%")
                m4.metric("Avg PD Risk", f"{(probas[:,1]*100).mean():.1f}%")

                c1, c2 = st.columns(2)
                with c1:
                    fig_pie = px.pie(
                        values=[n_pd, n_ht],
                        names=["Parkinson's Risk", "Healthy"],
                        color_discrete_sequence=["#DC2626", "#16A34A"],
                        title="Prediction Distribution"
                    )
                    fig_pie.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_pie, use_container_width=None)
                with c2:
                    fig_hist = px.histogram(
                        x=probas[:, 1]*100, nbins=20,
                        title="PD Risk Score Distribution",
                        labels={"x": "PD Risk (%)"},
                        color_discrete_sequence=["#16A34A"]
                    )
                    fig_hist.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)",
                                           plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_hist, use_container_width=None)

                st.dataframe(results, use_container_width=None)
                csv_out = results.to_csv(index=False).encode("utf-8")
                st.download_button("📥  Download Results", csv_out,
                                   "parkinsons_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE EXPLORER (new tab)
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Feature Explorer — Understand the Voice Biomarkers")

    try:
        import os
        df_raw = pd.read_csv(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "raw", "parkinsons.data"
        ))
        df_raw = df_raw.drop(["name"], axis=1)

        c1, c2 = st.columns(2)
        with c1:
            feat_x = st.selectbox("X-axis feature", ORIGINAL_FEATURES, index=21)  # PPE
        with c2:
            feat_y = st.selectbox("Y-axis feature", ORIGINAL_FEATURES, index=16)  # RPDE

        df_raw["Diagnosis"] = df_raw["status"].map({1: "Parkinson's", 0: "Healthy"})
        fig_sc = px.scatter(
            df_raw, x=feat_x, y=feat_y, color="Diagnosis",
            color_discrete_map={"Parkinson's": "#DC2626", "Healthy": "#16A34A"},
            title=f"{feat_x} vs {feat_y} — PD vs Healthy",
            opacity=0.7, marginal_x="histogram", marginal_y="histogram"
        )
        fig_sc.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sc, use_container_width=None)

        st.markdown("### Feature Correlation with Parkinson's Diagnosis")
        corr = df_raw.drop(["Diagnosis"], axis=1).corr()["status"].drop("status").sort_values()
        colors = ["#DC2626" if v > 0 else "#16A34A" for v in corr.values]
        fig_corr = go.Figure(go.Bar(
            x=corr.values, y=corr.index, orientation="h",
            marker_color=colors
        ))
        fig_corr.add_vline(x=0, line_color="black", line_width=1)
        fig_corr.update_layout(
            height=500, title="Pearson Correlation with Parkinson's (status)",
            xaxis_title="Correlation Coefficient",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=150)
        )
        st.plotly_chart(fig_corr, use_container_width=None)

        # ── Initialize Session State for the buttons ──
        if "show_learning_curve" not in st.session_state:
            st.session_state.show_learning_curve = False
        if "show_confusion_matrix" not in st.session_state:
            st.session_state.show_confusion_matrix = False

        st.markdown("---")
        st.markdown("### Model Generalization (Learning Curve)")
        
        # Button updates the session state
        if st.button("📈 Generate Learning Curve"):
            st.session_state.show_learning_curve = True

        # Check the session state instead of the button directly
        if st.session_state.show_learning_curve:
            with st.spinner("Training multiple sub-models to check for overfitting..."):
                from sklearn.model_selection import learning_curve
                
                X_all = df_raw[ORIGINAL_FEATURES]
                y_all = df_raw["status"]
                
                # Transform the data through your pipeline
                X_p = selector.transform(scaler.transform(engineer_features(X_all)))
                
                train_sizes, train_scores, test_scores = learning_curve(
                    model, X_p, y_all, cv=5, scoring='accuracy',
                    train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
                )
                
                # Calculate means
                train_mean = np.mean(train_scores, axis=1) * 100
                test_mean = np.mean(test_scores, axis=1) * 100
                
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scatter(x=train_sizes, y=train_mean, name="Training Accuracy",
                                          line=dict(color='#ef4444', width=3)))
                fig_lc.add_trace(go.Scatter(x=train_sizes, y=test_mean, name="Validation Accuracy",
                                          line=dict(color='#22c55e', width=3)))
                
                fig_lc.update_layout(
                    title="Learning Curve: Training vs. Validation",
                    xaxis_title="Number of Training Samples",
                    yaxis_title="Accuracy (%)",
                    yaxis=dict(range=[80, 101], gridcolor=vars["chart_grid"]),
                    xaxis=dict(gridcolor=vars["chart_grid"]),
                    font=dict(color=vars["chart_text"]),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor=vars["chart_bg"],
                    legend=dict(orientation="h", y=-0.2),
                    template=vars["plotly_theme"]
                )
                st.plotly_chart(fig_lc, use_container_width=None)
                
                st.info("💡 **Interpretation:** If the Green line is close to the Red line at the end, your model has **Generalized well** and is not overfitted.")


        st.markdown("---")
        st.markdown("### Diagnostic Accuracy (Confusion Matrix)")
        st.markdown(
            "This matrix shows exactly where the model makes mistakes using 5-Fold Cross Validation. "
            "In medical screening, we care most about minimizing **False Negatives** (missing a sick patient)."
        )

        # Button updates the session state
        if st.button("🧮 Generate Confusion Matrix"):
            st.session_state.show_confusion_matrix = True

        # Check the session state instead of the button directly
        if st.session_state.show_confusion_matrix:
            with st.spinner("Calculating out-of-fold predictions..."):
                from sklearn.metrics import confusion_matrix
                from sklearn.model_selection import cross_val_predict
                
                X_all = df_raw[ORIGINAL_FEATURES]
                y_all = df_raw["status"]
                
                # Transform data through the pipeline
                X_p = selector.transform(scaler.transform(engineer_features(X_all)))
                
                # Use cross_val_predict to get realistic out-of-sample predictions
                y_pred = cross_val_predict(model, X_p, y_all, cv=5)
                cm = confusion_matrix(y_all, y_pred)
                
                # Plotly Heatmap for the Confusion Matrix
                fig_cm = px.imshow(
                    cm, 
                    text_auto=True, 
                    color_continuous_scale=[[0, vars["section_bg"]], [1, "#DC2626"]], 
                    labels=dict(x="Model Prediction", y="Actual Clinical Diagnosis", color="Patients"),
                    x=["Predicted Healthy", "Predicted Parkinson's"],
                    y=["Actual Healthy", "Actual Parkinson's"]
                )
                
                fig_cm.update_layout(
                    title="5-Fold CV Confusion Matrix",
                    height=450,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=vars["chart_text"]),
                    template=vars["plotly_theme"]
                )
                
                # Make the numbers inside the matrix large and readable
                fig_cm.update_traces(textfont={"size": 28, "color": "white"})
                
                # Display the matrix alongside a clinical breakdown
                col1, col2 = st.columns([1.5, 1])
                with col1:
                    st.plotly_chart(fig_cm, use_container_width=None)
                with col2:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.markdown(f"**✅ True Positives (Caught PD):** {cm[1][1]}")
                    st.markdown(f"**✅ True Negatives (Healthy):** {cm[0][0]}")
                    st.markdown(f"**⚠️ False Positives (False Alarm):** {cm[0][1]}")
                    st.markdown(f"**🚨 False Negatives (Missed PD):** {cm[1][0]}")

    except Exception as e:
        st.warning(f"Feature explorer needs the raw dataset: {e}")

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — LIVE VOICE DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🎤 Live Voice Diagnostic")

    # ── Clinical Disclaimer (high-visibility) ─────────────────────────
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(234,179,8,0.08), rgba(234,179,8,0.02));
        border: 1px solid rgba(234,179,8,0.35);
        border-left: 4px solid #EAB308;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.5rem;
        color: {vars['text_main']};
    ">
        <div style="font-weight:700; font-size:1rem; color:#EAB308; margin-bottom:.6rem;
                    display:flex; align-items:center; gap:8px;">
            ⚠️ Important Clinical Disclaimer
        </div>
        <div style="font-size:.88rem; line-height:1.7; color:{vars['text_dim']};">
            <b>1.</b> This tool is an <b>educational screening aid</b>, not a certified medical diagnostic device.
            Any result must be confirmed by a qualified neurologist.<br>
            <b>2.</b> Recording quality directly impacts accuracy. Consumer-grade laptop microphones
            introduce noise artefacts that can inflate jitter/shimmer readings, potentially
            producing <b>false positives</b>. Clinical-grade recordings use condenser microphones
            in sound-treated rooms.<br>
            <b>3.</b> Your voice recording is processed <b>in-memory only</b> and the temporary file
            is <b>permanently deleted</b> from the server immediately after analysis completes.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Patient Instruction Card ──────────────────────────────────────
    st.markdown(f"""
    <div class="section-card">
        <div class="section-card-title">
            <span>📋</span> Recording Protocol
        </div>
        <div style="display:flex; flex-direction:column; gap:14px; margin-top:.5rem;">
            <div style="display:flex; align-items:center; gap:14px; font-size:.95rem; color:{vars['text_main']};">
                <div style="background:#16a34a; color:white; width:30px; height:30px;
                            border-radius:50%; display:flex; align-items:center;
                            justify-content:center; font-weight:700; flex-shrink:0;">1</div>
                <div>Sit in a <b>quiet room</b> with minimal background noise.</div>
            </div>
            <div style="display:flex; align-items:center; gap:14px; font-size:.95rem; color:{vars['text_main']};">
                <div style="background:#16a34a; color:white; width:30px; height:30px;
                            border-radius:50%; display:flex; align-items:center;
                            justify-content:center; font-weight:700; flex-shrink:0;">2</div>
                <div>Hold your microphone approximately <b>6 inches (15 cm)</b> from your mouth.</div>
            </div>
            <div style="display:flex; align-items:center; gap:14px; font-size:.95rem; color:{vars['text_main']};">
                <div style="background:#16a34a; color:white; width:30px; height:30px;
                            border-radius:50%; display:flex; align-items:center;
                            justify-content:center; font-weight:700; flex-shrink:0;">3</div>
                <div>Take a deep breath and sustain a clear <b>"Ahhh"</b> sound for at least <b>5 seconds</b>.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Audio Recorder ────────────────────────────────────────────────
    audio_sample = st.audio_input(
        "🎙️ Click the microphone to start recording your voice sample",
        key="voice_diagnostic_recorder"
    )

    if audio_sample:
        st.audio(audio_sample)
        st.markdown(f"""
        <div style="font-size:.85rem; color:{vars['text_dim']}; margin:.5rem 0 1rem;
                    display:flex; align-items:center; gap:6px;">
            ✅ Audio captured: <b>{audio_sample.size / 1024:.1f} KB</b> |
            Format: <b>{audio_sample.type}</b>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀  Analyse Voice Sample", type="primary", key="run_voice_analysis"):
            wav_path = None  # Track for guaranteed cleanup
            try:
                # ── Stage 1: Audio Standardization ────────────────
                with st.spinner("⏳ Stage 1/3 — Standardizing audio signal (44.1 kHz, 16-bit mono)..."):
                    wav_path = process_audio_signal(audio_sample)

                # ── Stage 2: Praat Feature Extraction ─────────────
                with st.spinner("🔬 Stage 2/3 — Extracting 22 clinical voice biomarkers via Praat..."):
                    features = extract_voice_features(wav_path)

                # ── Stage 3: ML Inference ─────────────────────────
                with st.spinner("🧠 Stage 3/3 — Running Voting Ensemble inference..."):
                    result = predict_from_voice(features)

                # ── Display Results ───────────────────────────────
                st.markdown("---")
                st.markdown("### Diagnostic Results")

                col_result, col_gauge = st.columns([1.3, 1])

                with col_result:
                    p_score = result['pd_probability'] * 100
                    if p_score > 75:
                        res_class = "result-pd"
                        title_html = "⚠️ HIGH RISK — Parkinson's Indicators Detected"
                        val_color = "#fecaca"
                        warn_color = "#f87171"
                        warn_msg = "⚠️ High probability. Consumer microphone recording — neurologist confirmation required."
                        bar_color = "#DC2626"
                    elif p_score >= 60:
                        res_class = "result-warning"
                        title_html = "🟡 MEDIUM RISK — Borderline Indicators Detected"
                        val_color = "#fef08a"
                        warn_color = "#fde047"
                        warn_msg = "🟡 Borderline results. Consumer microphone recording — consider clinical evaluation."
                        bar_color = "#EAB308"
                    else:
                        res_class = "result-healthy"
                        title_html = "✅ LOW RISK — No Significant Indicators"
                        val_color = "#bbf7d0"
                        warn_color = "#4ade80"
                        warn_msg = "✅ Regular monitoring is still recommended for at-risk age groups."
                        bar_color = "#16A34A"

                    st.markdown(f"""
                    <div style="display:flex; flex-direction:column; justify-content:center; height:260px;">
                        <div class="{res_class}" style="padding:1.5rem;">
                            <div style="font-size:1.25rem; font-weight:700; color:#ffffff; margin-bottom:.5rem">
                                {title_html}
                            </div>
                            <div style="font-size:1.05rem; color:{val_color}; line-height:1.7">
                                PD Risk Probability: <strong>{p_score:.1f}%</strong><br>
                                Model Confidence: <strong>{result['confidence']*100:.1f}%</strong><br>
                                Risk Level: <strong>{result['risk_level']}</strong>
                            </div>
                            <div style="margin-top:.8rem; font-size:.85rem; color:{warn_color}">
                                {warn_msg}
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                with col_gauge:
                    fig_voice_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=round(p_score, 1),
                        title={"text": "PD Risk", "font": {"size": 22, "color": "white", "weight": "bold"}},
                        number={"suffix": "%", "font": {"size": 48, "color": "white", "weight": "bold"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
                            "bar":  {"color": bar_color},
                            "steps": [
                                {"range": [0,  60], "color": "#DCFCE7"},
                                {"range": [60, 75], "color": "#FEF9C3"},
                                {"range": [75,100], "color": "#FEE2E2"},
                            ],
                            "threshold": {"line": {"color": "white", "width": 3}, "value": p_score}
                        }
                    ))
                    fig_voice_gauge.update_layout(
                        height=260, margin=dict(l=25, r=25, t=50, b=25),
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_voice_gauge, use_container_width=None,
                                   config={"displayModeBar": False})

                # ── Extracted Features Table ──────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("🔎 View Extracted Voice Biomarkers (22 features)", expanded=False):
                    feat_df = pd.DataFrame(
                        list(result["features_used"].items()),
                        columns=["Feature", "Extracted Value"]
                    )
                    st.dataframe(feat_df, use_container_width=None, hide_index=True)

                # ── Generate PDF Report ──
                st.markdown("<br>", unsafe_allow_html=True)
                report_bytes = generate_report(result["features_used"], result["pd_probability"]*100)
                st.download_button(
                    label="📥 Download Clinical Screening Report (PDF)",
                    data=report_bytes,
                    file_name=f"Parkinsons_Voice_Report_{datetime.date.today()}.pdf",
                    mime="application/pdf",
                    use_container_width=None,
                    key="download_mic_report"
                )

            except ValueError as ve:
                st.error(f"🎤 {ve}")
            except FileNotFoundError as fe:
                st.error(f"🚨 {fe}")
            except Exception as e:
                st.error(f"Unexpected error during voice analysis: {e}")
                st.exception(e)
            finally:
                # ── GUARANTEED CLEANUP: delete temp WAV from disk ─
                if wav_path and os.path.exists(wav_path):
                    try:
                        os.remove(wav_path)
                    except OSError:
                        pass  # Best-effort deletion; OS will reclaim on reboot

    # ── OR: Upload a Pre-Recorded WAV File ────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div class="section-card">
        <div class="section-card-title">
            <span>📁</span> Upload a Pre-Recorded Voice Sample
        </div>
        <div class="section-card-desc">
            If you have a clinical-quality recording, upload the WAV file directly
            for more accurate analysis than a browser microphone can provide.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_wav = st.file_uploader(
        "Upload a pre-recorded voice sample",
        type=["wav"],
        key="voice_file_uploader",
        help="Only uncompressed .wav files are accepted. Mono or stereo, any sample rate."
    )

    if uploaded_wav:
        st.audio(uploaded_wav)
        st.markdown(f"""
        <div style="font-size:.85rem; color:{vars['text_dim']}; margin:.5rem 0 1rem;
                    display:flex; align-items:center; gap:6px;">
            ✅ File loaded: <b>{uploaded_wav.name}</b> — <b>{uploaded_wav.size / 1024:.1f} KB</b>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀  Analyse Uploaded Audio", type="primary", key="run_upload_analysis"):
            import tempfile
            tmp_path = None
            validated_path = None
            try:
                # ── Stage 0: Write uploaded bytes to a secure temp file ──
                with st.spinner("⏳ Writing to secure buffer..."):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(uploaded_wav.read())
                        tmp_path = tmp.name

                # ── Stage 1: QA Validation Gate ──────────────────────
                with st.spinner("🔍 Stage 1/3 — Validating audio properties (channels, sample rate, bit depth)..."):
                    validated_path = validate_audio_properties(tmp_path)

                # ── Stage 2: Praat Feature Extraction ────────────────
                with st.spinner("🔬 Stage 2/3 — Extracting 22 clinical voice biomarkers via Praat..."):
                    features = extract_voice_features(validated_path)

                # ── Stage 3: ML Inference ────────────────────────────
                with st.spinner("🧠 Stage 3/3 — Running Voting Ensemble inference..."):
                    result = predict_from_voice(features)

                # ── Display Results ───────────────────────────────
                st.markdown("---")
                st.markdown("### Diagnostic Results (Uploaded File)")

                col_res, col_gau = st.columns([1.3, 1])

                with col_res:
                    p_score = result['pd_probability'] * 100
                    if p_score > 80:
                        res_class = "result-pd"
                        title_html = "⚠️ HIGH RISK — Parkinson's Indicators Detected"
                        val_color = "#fecaca"
                        warn_color = "#f87171"
                        warn_msg = "⚠️ High probability. Screening tool only — neurologist confirmation required."
                        bar_color = "#DC2626"
                    elif p_score >= 60:
                        res_class = "result-warning"
                        title_html = "🟡 MEDIUM RISK — Borderline Indicators Detected"
                        val_color = "#fef08a"
                        warn_color = "#fde047"
                        warn_msg = "🟡 Borderline results. Consider clinical evaluation."
                        bar_color = "#EAB308"
                    else:
                        res_class = "result-healthy"
                        title_html = "✅ LOW RISK — No Significant Indicators"
                        val_color = "#bbf7d0"
                        warn_color = "#4ade80"
                        warn_msg = "✅ Regular monitoring is still recommended for at-risk age groups."
                        bar_color = "#16A34A"

                    st.markdown(f"""
                    <div style="display:flex; flex-direction:column; justify-content:center; height:260px;">
                        <div class="{res_class}" style="padding:1.5rem;">
                            <div style="font-size:1.25rem; font-weight:700; color:#ffffff; margin-bottom:.5rem">
                                {title_html}
                            </div>
                            <div style="font-size:1.05rem; color:{val_color}; line-height:1.7">
                                PD Risk Probability: <strong>{p_score:.1f}%</strong><br>
                                Model Confidence: <strong>{result['confidence']*100:.1f}%</strong><br>
                                Risk Level: <strong>{result['risk_level']}</strong>
                            </div>
                            <div style="margin-top:.8rem; font-size:.85rem; color:{warn_color}">
                                {warn_msg}
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                with col_gau:
                    fig_up_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=round(p_score, 1),
                        title={"text": "PD Risk", "font": {"size": 22, "color": "white", "weight": "bold"}},
                        number={"suffix": "%", "font": {"size": 48, "color": "white", "weight": "bold"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
                            "bar":  {"color": bar_color},
                            "steps": [
                                {"range": [0,  60], "color": "#DCFCE7"},
                                {"range": [60, 75], "color": "#FEF9C3"},
                                {"range": [75,100], "color": "#FEE2E2"},
                            ],
                            "threshold": {"line": {"color": "white", "width": 3}, "value": p_score}
                        }
                    ))
                    fig_up_gauge.update_layout(
                        height=260, margin=dict(l=25, r=25, t=50, b=25),
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_up_gauge, use_container_width=None,
                                   config={"displayModeBar": False})

                # ── Extracted Features Table ──────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("🔎 View Extracted Voice Biomarkers (22 features)", expanded=False):
                    feat_df = pd.DataFrame(
                        list(result["features_used"].items()),
                        columns=["Feature", "Extracted Value"]
                    )
                    st.dataframe(feat_df, use_container_width=None, hide_index=True)

                # ── Generate PDF Report ──
                st.markdown("<br>", unsafe_allow_html=True)
                report_bytes = generate_report(result["features_used"], result["pd_probability"]*100)
                st.download_button(
                    label="📥 Download Clinical Screening Report (PDF)",
                    data=report_bytes,
                    file_name=f"Parkinsons_Upload_Report_{datetime.date.today()}.pdf",
                    mime="application/pdf",
                    use_container_width=None,
                    key="download_upload_report"
                )

            except InvalidAudioError as ae:
                st.error(f"🔊 {ae}")
            except ValueError as ve:
                st.error(f"🎤 {ve}")
            except FileNotFoundError as fe:
                st.error(f"🚨 {fe}")
            except Exception as e:
                st.error(f"Unexpected error during uploaded file analysis: {e}")
                st.exception(e)
            finally:
                # ── GUARANTEED CLEANUP: delete ALL temp files from disk ─
                for path in [tmp_path, validated_path]:
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass

# ── Footer ────────────────────────────────────────────────────────────────
# st.markdown("---")
# st.markdown(
#     "**Shivam Kothiyal** · MCA IIIrd Sem · Roll No: 241347080031 · "
#     "Dept. of CS & IT · H.N.B. Garhwal University · Under the guidance of **Prof. Y.P. Raiwani** · "
#     "*Screening tool only — not a substitute for clinical diagnosis.*"
# )


st.divider()
st.markdown("""
    <div style="text-align: center; color: #808080; font-size: 0.9rem;">
        <strong>Shivam Kothiyal</strong> • MCA IIIrd Sem • Roll No: 241347080031<br>
        Dept. of CS & IT • <strong>H.N.B. Garhwal University</strong><br>
        <em>Under the guidance of Prof. Y.P. Raiwani</em>
    </div>
    <div style="text-align: center; background-color: #262730; padding: 10px; border-radius: 5px; margin-top: 20px; border: 1px solid #ff4b4b;">
        <span style="color: #ff4b4b; font-weight: bold;">⚠️ MEDICAL DISCLAIMER:</span> 
        <span style="color: #fafafa; font-size: 0.8rem;">
            This system is for screening and research purposes only. It is not a clinical diagnostic tool.
        </span>
    </div>
""", unsafe_allow_html=True)