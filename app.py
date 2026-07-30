"""MediPredict AI - entry point. Sets page config, loads the shared theme,
and renders sidebar branding. Streamlit's multi-page framework auto-discovers
pages/ for navigation."""

import streamlit as st
from src.ui import load_css, hero, feature_card

st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;padding:0.5rem 0 0.25rem 0;">
            <div style="font-size:1.6rem;">🩺</div>
            <div>
                <div style="font-weight:700;font-size:1.05rem;color:#0F6E56;">MediPredict AI</div>
                <div style="font-size:0.78rem;color:#5F5E5A;">Disease Risk Prediction</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("Navigate the workflow using the pages above, from dataset upload through to live risk prediction.")

hero(
    "🩺 MediPredict AI",
    "An end-to-end machine learning system for intelligent heart disease risk prediction — "
    "from raw data to a live, explainable prediction.",
)

st.markdown("#### Workflow")
cols = st.columns(4)
steps = [
    ("1", "Prepare data", "Upload, clean, and explore the dataset."),
    ("2", "Engineer features", "Encode, scale, and select the strongest predictors."),
    ("3", "Train & evaluate", "Compare 4 models on accuracy, F1, and ROC-AUC."),
    ("4", "Predict", "Enter patient data for an instant risk assessment."),
]
for col, (num, title, desc) in zip(cols, steps):
    with col:
        st.markdown(feature_card(num, title, desc), unsafe_allow_html=True)

st.markdown("")
st.info("👈 Start with **Home** in the sidebar, or jump directly to any stage of the workflow.")
