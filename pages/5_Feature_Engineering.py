import streamlit as st

from src.feature_engineering import encode_features, split_data, scale_features, select_features
from src.model_utils import save_model
from src.data_loader import save_processed_data
from src.ui import load_css, page_header, stat_card

load_css()
page_header("⚙", "Feature Engineering", "Encode categorical features, scale numeric ones, and select the strongest predictors.")

df = st.session_state.get("clean_df")
if df is None:
    df = st.session_state.get("raw_df")
if df is None:
    st.warning("Please load and clean a dataset first.")
    st.stop()

if st.button("Run feature engineering", type="primary"):
    with st.spinner("Encoding, splitting, and scaling..."):
        encoded_df = encode_features(df)
        X_train, X_test, y_train, y_test = split_data(encoded_df)
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        top_features = select_features(X_train_scaled, y_train, k=10)
        save_model(scaler, "scaler")

    st.session_state.update({
        "X_train": X_train_scaled, "X_test": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "top_features": top_features,
    })
    st.success("Feature engineering complete. Scaler saved to models/scaler.joblib")

if "X_train" in st.session_state:
    st.subheader("Train / test split")
    col1, col2 = st.columns(2)
    col1.markdown(stat_card("Training rows", str(st.session_state["X_train"].shape[0])), unsafe_allow_html=True)
    col2.markdown(stat_card("Test rows", str(st.session_state["X_test"].shape[0])), unsafe_allow_html=True)

    st.subheader("Top features by ANOVA F-value")
    st.write(st.session_state["top_features"])

    st.subheader("Scaled training data preview")
    st.dataframe(st.session_state["X_train"].head(), use_container_width=True)
