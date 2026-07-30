import streamlit as st
from src.ui import load_css, page_header

load_css()
page_header("ℹ", "About", "Project background, tech stack, and disclaimer.")

st.markdown(
    """
    **MediPredict AI** is an end-to-end machine learning web application built
    to demonstrate the complete ML lifecycle: data cleaning, exploratory
    analysis, feature engineering, model training and comparison, evaluation,
    and live prediction — wrapped in a professional Streamlit interface.

    **Tech stack:** Streamlit &middot; Pandas &middot; NumPy &middot; Scikit-learn &middot; Plotly &middot; Joblib

    **Dataset:** UCI Heart Disease dataset (Cleveland subset)

    **Author:** Qaim Ali — BS Computer Science, Abasyn University Peshawar, Pakistan
    &nbsp;|&nbsp; GitHub: [kammi11](https://github.com/kammi11)
    """
)

st.markdown("---")
st.warning(
    "⚠️ **Disclaimer:** This application was built for educational purposes. "
    "Predictions are not medical advice and must never be used for real "
    "clinical decisions. Always consult a licensed physician."
)
