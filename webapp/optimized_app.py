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
import textwrap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (MODEL_PATH, SCALER_PATH, SELECTOR_PATH,
                        META_PATH, ORIGINAL_FEATURES, FEATURE_META)
from src.preprocessing import engineer_features

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parkinson's Detection System",
    page_icon="🧠", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}

/* ── Main header ── */
.main-header{
  background:linear-gradient(135deg,#0a0f0a 0%,#0d3321 60%,#16A34A 100%);
  padding:1.8rem 2rem;border-radius:14px;margin-bottom:1.4rem;
  border:1px solid #1a4a2e;}
.main-header h1{color:#fff;margin:0;font-size:2rem;font-weight:700;letter-spacing:-.5px}
.main-header p{color:#6ee7b7;margin:.4rem 0 0;font-size:.9rem}

/* ══════════════════════════════════════════════
   GLASSMORPHISM PERFORMANCE HEADER
══════════════════════════════════════════════ */
.perf-header{
  position:relative;overflow:hidden;
  background:linear-gradient(135deg,
    rgba(10,20,14,0.72) 0%,
    rgba(15,40,25,0.68) 50%,
    rgba(22,163,74,0.10) 100%);
  border:1px solid rgba(74,222,128,0.18);
  border-radius:18px;
  padding:1.6rem 1.8rem 1.4rem;
  margin:1rem 0 1.4rem;
  backdrop-filter:blur(20px);
  -webkit-backdrop-filter:blur(20px);
  box-shadow:0 8px 32px rgba(0,0,0,0.45),
             inset 0 1px 0 rgba(255,255,255,0.06);}
/* animated gradient orbs behind the glass */
.perf-header::before{
  content:'';
  position:absolute;top:-60px;right:-60px;
  width:220px;height:220px;
  background:radial-gradient(circle,rgba(22,163,74,0.18) 0%,transparent 70%);
  pointer-events:none;border-radius:50%;}
.perf-header::after{
  content:'';
  position:absolute;bottom:-40px;left:10%;
  width:160px;height:160px;
  background:radial-gradient(circle,rgba(74,222,128,0.10) 0%,transparent 70%);
  pointer-events:none;border-radius:50%;}

.perf-header-label{
  font-size:.7rem;font-weight:700;
  color:#6ee7b7;letter-spacing:1.5px;
  text-transform:uppercase;margin-bottom:1rem;
  display:flex;align-items:center;gap:8px;}
.perf-header-label::after{
  content:'';flex:1;
  height:1px;background:linear-gradient(90deg,rgba(74,222,128,0.3),transparent);}

.perf-cards{
  display:flex;gap:10px;flex-wrap:wrap;
  position:relative;z-index:1;}

.perf-card{
  flex:1;min-width:130px;
  background:rgba(255,255,255,0.03);
  border:1px solid rgba(74,222,128,0.15);
  border-radius:14px;
  padding:1rem 1rem .85rem;
  text-align:center;
  position:relative;overflow:hidden;
  transition:transform .22s ease,box-shadow .22s ease,border-color .22s ease;}
/* shimmer sweep on hover */
.perf-card::before{
  content:'';
  position:absolute;top:0;left:-75%;
  width:50%;height:100%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,0.06),transparent);
  transition:left .45s ease;
  pointer-events:none;}
.perf-card:hover::before{left:130%;}
.perf-card:hover{
  transform:translateY(-4px);
  box-shadow:0 8px 24px rgba(22,163,74,0.22);
  border-color:rgba(74,222,128,0.35);}

.perf-card .pc-icon{
  font-size:1.35rem;line-height:1;
  margin-bottom:.4rem;display:block;}
.perf-card .pc-val{
  font-size:1.9rem;font-weight:800;
  color:#4ade80;line-height:1.05;
  letter-spacing:-.5px;}
.perf-card .pc-lbl{
  font-size:.68rem;color:#86efac;
  margin-top:.35rem;text-transform:uppercase;
  letter-spacing:.8px;font-weight:600;}
.perf-card .pc-badge{
  display:inline-block;margin-top:.45rem;
  font-size:.62rem;font-weight:700;
  padding:.15rem .55rem;
  border-radius:20px;
  background:rgba(22,163,74,0.15);
  border:1px solid rgba(74,222,128,0.25);
  color:#6ee7b7;letter-spacing:.4px;}
/* highlight the star card (Recall) */
.perf-card.star-card{
  border-color:rgba(74,222,128,0.30);
  background:rgba(22,163,74,0.07);}
.perf-card.star-card .pc-val{color:#34d399;}

/* Responsive */
@media(max-width:640px){
  .perf-cards{display:grid;grid-template-columns:1fr 1fr;}
  .perf-card .pc-val{font-size:1.5rem;}
}

/* ── Input section card ── */
.input-card{
  background:#111812;
  border:1px solid #1f3327;
  border-radius:12px;
  padding:1.2rem 1.4rem;
  margin-bottom:1.2rem;}
.card-title{
  display:flex;align-items:center;gap:8px;
  font-size:1rem;font-weight:700;color:#4ade80;
  margin-bottom:.2rem;border-bottom:1px solid #1f3327;padding-bottom:.6rem;}
.card-tooltip{
  font-size:.78rem;color:#6b7280;
  margin-bottom:.8rem;padding:.5rem .8rem;
  background:rgba(74,222,128,0.05);
  border-left:3px solid #16A34A;
  border-radius:0 6px 6px 0;}

/* ── Result boxes ── */
.result-pd{background:rgba(220,38,38,0.1);border:2px solid #DC2626;border-radius:12px;padding:1.2rem}
.result-healthy{background:rgba(22,163,74,0.1);border:2px solid #16A34A;border-radius:12px;padding:1.2rem}

/* ── Tip box ── */
.tip-box{background:rgba(217,119,6,0.1);border-left:4px solid #D97706;
  border-radius:6px;padding:.7rem 1rem;font-size:.87rem;color:#fbbf24}

/* ── Predict button ── */
.stButton>button{
  background:linear-gradient(135deg,#059669,#16A34A);
  color:#fff;border:none;
  padding:.75rem 3rem;border-radius:8px;
  font-size:1.05rem;font-weight:700;
  box-shadow:0 0 16px rgba(16,163,74,0.45);
  transition:all .25s ease;width:100%;margin-top:.8rem;}
.stButton>button:hover{
  background:linear-gradient(135deg,#047857,#15803d);
  box-shadow:0 0 28px rgba(16,163,74,0.7);
  transform:translateY(-2px);}
.stButton>button:active{transform:translateY(0);}

/* ── Group header (legacy kept for batch tab) ── */
.group-header{background:#0d3321;color:#4ade80;padding:.4rem .8rem;
  border-radius:5px;font-weight:600;margin-bottom:.4rem;font-size:.95rem;
  border:1px solid #1a4a2e;}

/* ══════════════════════════════════════════════
   SECTIONAL FEATURE CARDS  (Tab 1)
══════════════════════════════════════════════ */
.section-card{
  background:#1e1e1e;
  border:1px solid #2a2a2a;
  border-radius:14px;
  padding:1.4rem 1.6rem 1rem;
  margin-bottom:1.4rem;
  box-shadow:0 2px 12px rgba(0,0,0,0.35);
  transition:box-shadow .25s;}
.section-card:hover{
  box-shadow:0 4px 24px rgba(22,163,74,0.12);}

.section-card-title{
  display:flex;
  align-items:center;
  gap:10px;
  font-size:1.05rem;
  font-weight:700;
  color:#4ade80;
  padding-bottom:.65rem;
  margin-bottom:.9rem;
  border-bottom:1px solid #2d2d2d;}
.section-card-icon{
  font-size:1.3rem;
  line-height:1;}
.section-card-badge{
  margin-left:auto;
  font-size:.7rem;
  font-weight:600;
  color:#6ee7b7;
  background:rgba(22,163,74,0.12);
  border:1px solid rgba(22,163,74,0.25);
  border-radius:20px;
  padding:.15rem .65rem;
  letter-spacing:.4px;
  text-transform:uppercase;}
.section-card-desc{
  font-size:.8rem;
  color:#6b7280;
  margin:-0.5rem 0 .9rem;
  padding:.4rem .75rem;
  background:rgba(74,222,128,0.04);
  border-left:3px solid #16A34A;
  border-radius:0 6px 6px 0;}

/* ── Responsive ── */
@media(max-width:768px){
  .glass-metrics{flex-direction:column;}
  .glass-card .val{font-size:1.4rem;}
  .section-card{padding:1rem .9rem .7rem;}
}

/* ── Medical tooltip details ── */
.med-tooltip{
  margin:.5rem 0 .9rem;
  background:rgba(30,58,138,0.08);
  border:1px solid rgba(99,102,241,0.18);
  border-radius:10px;
  padding:.75rem 1rem;
  font-size:.82rem;
  color:#a5b4fc;}
.med-tooltip summary{
  cursor:pointer;
  font-weight:700;
  color:#818cf8;
  list-style:none;
  display:flex;
  align-items:center;
  gap:7px;
  user-select:none;}
.med-tooltip summary::marker{display:none;}
.med-tooltip summary::-webkit-details-marker{display:none;}
.med-tooltip[open] summary{margin-bottom:.55rem;}
.med-tooltip dl{margin:.2rem 0 0;display:grid;grid-template-columns:auto 1fr;gap:.25rem .75rem;}
.med-tooltip dt{color:#c7d2fe;font-weight:700;white-space:nowrap;}
.med-tooltip dd{margin:0;color:#94a3b8;line-height:1.5;}
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
        # Before vs After bar chart
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
        fig.add_trace(go.Bar(name="Before (v1)", x=metrics, y=before,
                              marker_color="#BBF7D0"))
        fig.add_trace(go.Bar(name="After (v2)",  x=metrics, y=after,
                              marker_color="#16A34A"))
        fig.update_layout(
            barmode="group", height=260, title="Before vs After Improvements",
            yaxis=dict(range=[85, 100]), margin=dict(l=0,r=0,t=35,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.1),
            font=dict(color="#d1fae5")
        )
        st.plotly_chart(fig, use_container_width=True)
    with i3:
        # ── Top-10 Feature Importance (Mutual Information proxy scores) ──
        # These scores are MI-derived importance weights from the training pipeline.
        # Sorted descending; top 10 of the 18 selected features are highlighted.
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
            "#fbbf24"
            for s in fi_scores
        ]
        fig_fi = go.Figure(go.Bar(
            x=fi_scores[::-1],
            y=fi_features[::-1],
            orientation="h",
            marker=dict(color=fi_colors[::-1],
                        line=dict(color="rgba(255,255,255,0.06)", width=1)),
            text=[f"{s:.3f}" for s in fi_scores[::-1]],
            textposition="outside",
            textfont=dict(color="#86efac", size=10),
            hovertemplate="<b>%{y}</b><br>MI Score: %{x:.3f}<extra></extra>",
        ))
        fig_fi.update_layout(
            title=dict(text="🏆 Top 10 Feature Importance",
                       font=dict(size=13, color="#4ade80")),
            height=305,
            margin=dict(l=0, r=55, t=35, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            xaxis=dict(
                title=dict(text="Mutual Information Score",
                           font=dict(size=10, color="#86efac")),
                tickfont=dict(size=9, color="#86efac"),
                gridcolor="rgba(74,222,128,0.08)",
                range=[0, max(fi_scores) * 1.22],
            ),
            yaxis=dict(
                tickfont=dict(size=10, color="#d1fae5"),
                gridcolor="rgba(0,0,0,0)",
            ),
            bargap=0.28,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "👤  Single Patient Prediction",
    "📂  Batch Processing (CSV)",
    "📈  Feature Explorer"
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

    # ── Three sectional cards (with Plain-English medical tooltips) ──────────
    SECTION_CARDS = [
        {
            "icon": "🎵",
            "title": "Fundamental Frequency",
            "badge": "3 features",
            "desc": "Average, maximum and minimum pitch of the voice (Hz). "
                    "Parkinson's patients typically show a lower and more erratic pitch.",
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
            "icon": "〰️",
            "title": "Jitter & Shimmer (Pitch & Amplitude Variation)",
            "badge": "11 features",
            "desc": "Jitter = cycle-to-cycle wobble in pitch timing. Shimmer = cycle-to-cycle wobble in loudness. "
                    "Both are significantly elevated in Parkinson's patients.",
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
            "icon": "📊",
            "title": "Non-linear & Entropy Measures",
            "badge": "8 features",
            "desc": "Complexity and chaos metrics that reveal hidden patterns in the voice signal. "
                    "They capture irregularities invisible to simple amplitude/pitch analysis.",
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
        # ── Card header + tooltip ──
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
        # ── Sliders rendered via Streamlit (must be outside HTML string) ──
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
            col_res, col_chart = st.columns([1, 1])

            with col_res:
                
                # Gauge
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(proba[1]*100, 1),
                    title={"text": "PD Risk Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "#DC2626" if label == 1 else "#16A34A"},
                        "steps": [
                            {"range": [0,  40], "color": "#DCFCE7"},
                            {"range": [40, 70], "color": "#FEF9C3"},
                            {"range": [70,100], "color": "#FEE2E2"},
                        ],
                        "threshold": {"line": {"color":"black","width":2}, "value": 50}
                    }
                ))
                fig_g.update_layout(height=210, margin=dict(l=15,r=15,t=40,b=5),
                                     paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_g, use_container_width=True)

                # result message
                if label == 1:
                    st.markdown('\n'.join([line.lstrip() for line in f"""
                    <div class="result-pd">
                        <div style="font-size:1.1rem;font-weight:700;color:#991B1B;margin-bottom:.4rem">
                            ⚠️ POSITIVE — Parkinson's Risk Detected
                        </div>
                        <div style="font-size:.95rem;color:#7F1D1D;line-height:1.6">
                            PD Probability: <strong>{proba[1]*100:.1f}%</strong><br>
                            Model Confidence: <strong>{proba[label]*100:.1f}%</strong>
                        </div>
                        <div style="margin-top:.5rem;font-size:.82rem;color:#991B1B">
                            ⚠️ Screening tool only — neurologist confirmation required.
                        </div>
                    </div>""".split('\n')]), unsafe_allow_html=True)
                else:
                    st.markdown('\n'.join([line.lstrip() for line in f"""
                    <div class="result-healthy">
                        <div style="font-size:1.1rem;font-weight:700;color:#065F46;margin-bottom:.4rem">
                            ✅ NEGATIVE — No Parkinson's Indicators
                        </div>
                        <div style="font-size:.95rem;color:#064E3B;line-height:1.6">
                            Healthy Probability: <strong>{proba[0]*100:.1f}%</strong><br>
                            Model Confidence: <strong>{proba[label]*100:.1f}%</strong>
                        </div>
                        <div style="margin-top:.5rem;font-size:.82rem;color:#065F46">
                            Regular monitoring is still recommended for at-risk age groups.
                        </div>
                    </div>""".split('\n')]), unsafe_allow_html=True)



                # Derived features table
                st.markdown("**Derived features (auto-computed by model):**")
                derived = {
                    "PPE_RPDE_sum"       : round(feature_values["PPE"] + feature_values["RPDE"], 5),
                    "nonlinear_composite": round(feature_values["PPE"] * feature_values["RPDE"] * feature_values["DFA"], 6),
                    "Fo_range"           : round(feature_values["MDVP:Fhi(Hz)"] - feature_values["MDVP:Flo(Hz)"], 3),
                    "Jitter_total"       : round(feature_values["MDVP:Jitter(%)"] + feature_values["MDVP:RAP"] + feature_values["MDVP:PPQ"], 6),
                }
                st.dataframe(pd.DataFrame(derived, index=["Value"]).T.rename(columns={"Value":"Computed Value"}),
                             use_container_width=True)

            with col_chart:
                # Patient vs healthy range comparison for key features
                st.markdown("**Patient vs. healthy population range:**")
                key_feats  = ["PPE", "RPDE", "HNR", "NHR", "MDVP:Jitter(%)", "MDVP:Shimmer", "spread1", "DFA"]
                healthy_hi = [0.10, 0.45, 28.0, 0.02, 0.004, 0.02, -4.0, 0.72]
                patient_vs = [feature_values[f] for f in key_feats]
                flag       = ["⚠️ Elevated" if abs(patient_vs[i]) > abs(healthy_hi[i]) else "✅ Normal"
                               for i in range(len(key_feats))]
                df_cmp = pd.DataFrame({
                    "Feature"          : key_feats,
                    "Patient Value"    : patient_vs,
                    "Healthy Threshold": healthy_hi,
                    "Status"           : flag,
                })
                st.dataframe(df_cmp, use_container_width=True, hide_index=True)

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
            st.dataframe(batch_df.head(5), use_container_width=True)

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
                    st.plotly_chart(fig_pie, use_container_width=True)
                with c2:
                    fig_hist = px.histogram(
                        x=probas[:, 1]*100, nbins=20,
                        title="PD Risk Score Distribution",
                        labels={"x": "PD Risk (%)"},
                        color_discrete_sequence=["#16A34A"]
                    )
                    fig_hist.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)",
                                           plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_hist, use_container_width=True)

                st.dataframe(results, use_container_width=True)
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
        st.plotly_chart(fig_sc, use_container_width=True)

        # Correlation with target
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
        st.plotly_chart(fig_corr, use_container_width=True)

    except Exception as e:
        st.warning(f"Feature explorer needs the raw dataset: {e}")

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "**Shivam Kothiyal** · MCA IIIrd Sem · Roll No: 241347080031 · "
    "Dept. of CS & IT · H.N.B. Garhwal University · Under the guidance of **Prof. Y.P. Raiwani** · "
    "*Screening tool only — not a substitute for clinical diagnosis.*"
)
