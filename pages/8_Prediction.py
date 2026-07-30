import streamlit as st

from src.predict import predict_patient
from src.config import FEATURE_LABELS
from src.model_utils import model_exists
from src.ui import load_css, page_header, disclaimer, risk_badge, stat_card

load_css()
page_header("🩺", "Predict Disease Risk", "Enter patient information to get an instant risk assessment.")
disclaimer("Educational tool only — not medical advice. Always consult a qualified physician.")

if not (model_exists("best_model") and model_exists("scaler")):
    st.warning("Please train and evaluate models first (Train Models → Model Evaluation pages).")
    st.stop()

st.markdown("Enter patient information below:")

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    age = c1.number_input(FEATURE_LABELS["age"], 18, 100, 50)
    trestbps = c2.number_input(FEATURE_LABELS["trestbps"], 80, 220, 130)
    chol = c3.number_input(FEATURE_LABELS["chol"], 100, 600, 240)

    c1, c2, c3 = st.columns(3)
    thalach = c1.number_input(FEATURE_LABELS["thalach"], 60, 220, 150)
    oldpeak = c2.number_input(FEATURE_LABELS["oldpeak"], 0.0, 10.0, 1.0, step=0.1)
    ca = c3.selectbox(FEATURE_LABELS["ca"], [0, 1, 2, 3])

    c1, c2, c3 = st.columns(3)
    sex = c1.selectbox(FEATURE_LABELS["sex"], [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = c2.selectbox(FEATURE_LABELS["cp"], [0, 1, 2, 3])
    fbs = c3.selectbox(FEATURE_LABELS["fbs"], [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    c1, c2, c3 = st.columns(3)
    restecg = c1.selectbox(FEATURE_LABELS["restecg"], [0, 1, 2])
    exang = c2.selectbox(FEATURE_LABELS["exang"], [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    slope = c3.selectbox(FEATURE_LABELS["slope"], [0, 1, 2])

    thal = st.selectbox(FEATURE_LABELS["thal"], [0, 1, 2, 3])

    submitted = st.form_submit_button("Predict risk", type="primary")

if submitted:
    inputs = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }

    with st.spinner("Running prediction..."):
        result = predict_patient(inputs)

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.markdown(stat_card("Prediction", "Disease" if result["prediction"] == 1 else "No disease"), unsafe_allow_html=True)
    col2.markdown(stat_card("Confidence", f"{result['probability']:.1%}"), unsafe_allow_html=True)
    with col3:
        st.markdown(
            f'<div class="mp-stat"><div class="mp-stat-label">Risk level</div>'
            f'<div style="margin-top:0.4rem;">{risk_badge(result["risk_level"])}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.subheader("Suggested next steps")
    st.caption("Educational only — not a substitute for professional medical advice.")
    for step in result["next_steps"]:
        st.markdown(f"- {step}")
