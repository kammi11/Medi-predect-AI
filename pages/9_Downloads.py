import streamlit as st

from src.config import MODELS_DIR
from src.ui import load_css, page_header

load_css()
page_header("📥", "Downloads", "Export the cleaned dataset, evaluation results, and the trained model.")

if "clean_df" in st.session_state:
    st.subheader("Cleaned dataset")
    csv = st.session_state["clean_df"].to_csv(index=False).encode("utf-8")
    st.download_button("Download cleaned dataset (CSV)", csv, "heart_clean.csv", "text/csv")
else:
    st.info("Run Data Cleaning first to enable this download.")

if "eval_results" in st.session_state:
    st.subheader("Model comparison results")
    from src.evaluate import build_comparison_table
    comparison_csv = build_comparison_table(st.session_state["eval_results"]).to_csv(index=False).encode("utf-8")
    st.download_button("Download model comparison (CSV)", comparison_csv, "model_comparison.csv", "text/csv")
else:
    st.info("Run Model Evaluation first to enable this download.")

best_model_path = MODELS_DIR / "best_model.joblib"
if best_model_path.exists():
    st.subheader("Trained model")
    with open(best_model_path, "rb") as f:
        st.download_button("Download best model (.joblib)", f, "best_model.joblib")
else:
    st.info("Train and evaluate models first to enable this download.")
