import os

def refactor_app():
    with open("webapp/app.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Find the sections
    tabs_start = content.find('# ── Tabs')
    dashboard_start = content.find('cv_scores = meta.get("cv_scores", {})')
    tab1_start = content.find('# ══════════════════════════════════════════════════════════════════════════\n# TAB 1 — SINGLE PATIENT')
    tab2_start = content.find('# ══════════════════════════════════════════════════════════════════════════\n# TAB 2 — BATCH PREDICTION')
    tab3_start = content.find('# ══════════════════════════════════════════════════════════════════════════\n# TAB 3 — FEATURE EXPLORER')
    tab4_start = content.find('# ══════════════════════════════════════════════════════════════════════════\n# TAB 4 — LIVE VOICE DIAGNOSTIC')
    footer_start = content.find('# ── Footer')

    header_css = content[:dashboard_start]
    dashboard_code = content[dashboard_start:tabs_start]
    tab1_code = content[tab1_start:tab2_start]
    tab2_code = content[tab2_start:tab3_start]
    tab3_code = content[tab3_start:tab4_start]
    tab4_code = content[tab4_start:footer_start]
    footer_code = content[footer_start:]

    # Layer 1 code
    layer1 = """
# ── LAYER 1: Select Workflow ──────────────────────────────────────────────
st.markdown("---")
workflow = st.radio(
    "Select Workflow",
    ["👤 Single Patient", "🎤 Live Voice Diagnostic", "📂 Batch Processing", "📈 Feature Explorer"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("---")
"""

    # We need to extract SECTION_CARDS definition from tab1_code
    section_cards_start = tab1_code.find('SECTION_CARDS = [')
    section_cards_end = tab1_code.find('    feature_values = {}')
    section_cards_code = tab1_code[section_cards_start:section_cards_end]

    # Predict code from tab1
    predict_start = tab1_code.find('    if st.button("🔍  Analyse & Predict"):')
    predict_code = tab1_code[predict_start:]

    # Indent dashboard_code by 8 spaces robustly
    indented_dashboard = ""
    for line in dashboard_code.splitlines():
        if line.strip():
            indented_dashboard += "        " + line + "\n"
        else:
            indented_dashboard += "\n"


    # Rewrite Single Patient logic using containers
    new_single_patient = f"""
# ══════════════════════════════════════════════════════════════════════════
# WORKFLOW: SINGLE PATIENT
# ══════════════════════════════════════════════════════════════════════════
if workflow == "👤 Single Patient":
    {section_cards_code}
    
    layer2 = st.container()
    layer3 = st.container()
    
    # Render Layer 3 first in logic to bind sliders
    with layer3:
        st.markdown("---")
        st.markdown("### LAYER 3: Data Science Deep-Dive")
        
        # Insert Model Performance Dashboard here
{indented_dashboard.replace('with st.expander("📊 Model Performance Dashboard", expanded=True):', 'with st.expander("📊 Model Performance Dashboard", expanded=False):')}

        with st.expander("⚙️ Advanced Signal Fine-Tuning (For Data Scientists & Researchers)", expanded=False):
            st.markdown("Adjust the 22 core acoustic features manually.")
            
            # The sliders into 3 columns
            col_fo, col_jit, col_nonlin = st.columns(3)
            
            feature_values = {{}}
            
            # Column 1: Fundamental Frequency
            with col_fo:
                st.markdown("#### 🎵 Fundamental Frequency")
                for feat in SECTION_CARDS[0]["features"]:
                    mn, mx, default, desc, step = FEATURE_META[feat]
                    fmt = "%.6f" if step < 0.001 else ("%.4f" if step < 0.01 else "%.3f")
                    feature_values[feat] = st.number_input(label=feat, min_value=float(mn), max_value=float(mx), value=float(default), step=float(step), format=fmt, help=desc, key=f"feat_{{feat}}")

            # Column 2: Jitter & Shimmer
            with col_jit:
                st.markdown("#### 〰️ Jitter & Shimmer")
                for feat in SECTION_CARDS[1]["features"]:
                    mn, mx, default, desc, step = FEATURE_META[feat]
                    fmt = "%.6f" if step < 0.001 else ("%.4f" if step < 0.01 else "%.3f")
                    feature_values[feat] = st.number_input(label=feat, min_value=float(mn), max_value=float(mx), value=float(default), step=float(step), format=fmt, help=desc, key=f"feat_{{feat}}")

            # Column 3: Non-linear & Entropy
            with col_nonlin:
                st.markdown("#### 📊 Non-linear & Entropy")
                for feat in SECTION_CARDS[2]["features"]:
                    mn, mx, default, desc, step = FEATURE_META[feat]
                    fmt = "%.6f" if step < 0.001 else ("%.4f" if step < 0.01 else "%.3f")
                    feature_values[feat] = st.number_input(label=feat, min_value=float(mn), max_value=float(mx), value=float(default), step=float(step), format=fmt, help=desc, key=f"feat_{{feat}}")

    with layer2:
        st.markdown("### LAYER 2: Core Action & Interpretation Engine")
        st.markdown("Click below to run the clinical model. You can fine-tune the patient's measurements in Layer 3 below.")
        
        # Display Plain English Explanations as permanent text cards
        st.info("ℹ️ **Plain English Biomarker Guide:**\\n\\n"
                "**Fundamental Frequency (Fo, Fhi, Flo)**: Your natural speaking pitch. PD patients' voices often drop lower over time.\\n\\n"
                "**Jitter & Shimmer**: Wobble in pitch timing and loudness. A healthy voice has very little of this. Larger shimmer = weaker vocal muscle control.\\n\\n"
                "**Non-linear Measures**: Complexity metrics like NHR, HNR, and PPE. Elevated Pitch Period Entropy (PPE) strongly predicts PD."
                )

{predict_code}
"""

    # Update Batch Processing, Feature Explorer, Live Voice
    tab2_code = tab2_code.replace('with tab2:', 'if workflow == "📂 Batch Processing":')
    tab3_code = tab3_code.replace('with tab3:', 'if workflow == "📈 Feature Explorer":')
    tab4_code = tab4_code.replace('with tab4:', 'if workflow == "🎤 Live Voice Diagnostic":')

    # Assemble new content
    new_content = header_css + layer1 + new_single_patient + tab4_code + tab2_code + tab3_code + footer_code

    with open("webapp/app_v2.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("app_v2.py created successfully.")

if __name__ == "__main__":
    refactor_app()
