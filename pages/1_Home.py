import streamlit as st
from src.ui import load_css, page_header, feature_card, disclaimer, stat_card

load_css()
page_header("🏠", "Home", "Your starting point for the full MediPredict AI workflow.")

disclaimer()

st.markdown("#### Workflow steps")
steps = [
    ("1", "Dataset", "Upload or preview the heart disease dataset."),
    ("2", "Data Cleaning", "Handle missing values, duplicates, and outliers."),
    ("3", "EDA", "Explore distributions, correlations, and patterns."),
    ("4", "Feature Engineering", "Encode, scale, and select features."),
    ("5", "Train Models", "Train Logistic Regression, Decision Tree, RF, SVM."),
    ("6", "Model Evaluation", "Compare metrics and select the best model."),
    ("7", "Predict Risk", "Enter patient data for a live prediction."),
    ("8", "Downloads", "Export the cleaned dataset, results, and model."),
]
for row_start in range(0, len(steps), 4):
    cols = st.columns(4)
    for col, (num, title, desc) in zip(cols, steps[row_start:row_start + 4]):
        with col:
            st.markdown(feature_card(num, title, desc), unsafe_allow_html=True)
    st.markdown("")

st.markdown("#### At a glance")
c1, c2, c3 = st.columns(3)
c1.markdown(stat_card("Models compared", "4"), unsafe_allow_html=True)
c2.markdown(stat_card("Evaluation metrics", "5+"), unsafe_allow_html=True)
c3.markdown(stat_card("Workflow stages", "10"), unsafe_allow_html=True)
