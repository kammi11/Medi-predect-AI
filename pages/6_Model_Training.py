import streamlit as st

from src.train import train_all_models
from src.model_utils import save_model
from src.ui import load_css, page_header

load_css()
page_header("🤖", "Train Models", "Trains 4 classifiers: Logistic Regression, Decision Tree, Random Forest, SVM.")

if "X_train" not in st.session_state:
    st.warning("Please run Feature Engineering first.")
    st.stop()

if st.button("Train all models", type="primary"):
    with st.spinner("Training models..."):
        models = train_all_models(st.session_state["X_train"], st.session_state["y_train"])
        for name, model in models.items():
            save_model(model, name.lower().replace(" ", "_"))
    st.session_state["trained_models"] = models
    st.success("All 4 models trained and saved to models/")

if "trained_models" in st.session_state:
    st.subheader("Trained models")
    for name in st.session_state["trained_models"]:
        st.write(f"✅ {name}")

    st.info("Head to **Model Evaluation** to compare performance and pick the best one.")
