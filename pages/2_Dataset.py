import streamlit as st
import pandas as pd

from src.data_loader import load_raw_data, load_uploaded_data, get_dataset_summary
from src.config import RAW_DATA_PATH
from src.ui import load_css, page_header, stat_card

load_css()
page_header("📂", "Dataset", "Upload the heart disease dataset, or load the bundled sample from data/raw/heart.csv.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_uploaded_data(uploaded_file)
    st.success("File uploaded successfully.")
elif RAW_DATA_PATH.exists():
    df = load_raw_data()
    st.info("Loaded bundled sample dataset from data/raw/heart.csv")
else:
    df = None
    st.error("No dataset found. Please upload a CSV to continue.")

if df is not None:
    st.session_state["raw_df"] = df

    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    summary = get_dataset_summary(df)
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(stat_card("Rows", str(summary["rows"])), unsafe_allow_html=True)
    col2.markdown(stat_card("Columns", str(summary["columns"])), unsafe_allow_html=True)
    col3.markdown(stat_card("Missing values", str(summary["missing_values"])), unsafe_allow_html=True)
    col4.markdown(stat_card("Duplicate rows", str(summary["duplicate_rows"])), unsafe_allow_html=True)
