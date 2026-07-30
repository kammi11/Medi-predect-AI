import streamlit as st

from src.preprocessing import clean_pipeline
from src.data_loader import save_processed_data
from src.ui import load_css, page_header

load_css()
page_header("🧹", "Data Cleaning", "Fix data types → handle missing values → remove duplicates → remove IQR outliers.")

if "raw_df" not in st.session_state:
    st.warning("Please load a dataset on the Dataset page first.")
    st.stop()

df = st.session_state["raw_df"]

if st.button("Run cleaning pipeline", type="primary"):
    with st.spinner("Cleaning dataset..."):
        clean_df, report = clean_pipeline(df)
        st.session_state["clean_df"] = clean_df
        st.session_state["cleaning_report"] = report

    st.success(f"Cleaning complete. {df.shape[0]} rows → {clean_df.shape[0]} rows.")

if "clean_df" in st.session_state:
    clean_df = st.session_state["clean_df"]
    report = st.session_state["cleaning_report"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Missing values filled")
        if report["missing_values"]:
            st.json(report["missing_values"])
        else:
            st.write("No missing values found.")

    with col2:
        st.subheader("Duplicates & outliers")
        st.write(f"Duplicate rows removed: **{report['duplicates_removed']}**")
        st.write(f"Outlier rows removed: **{report['outliers']['total_rows_removed']}**")

    st.subheader("Cleaned dataset preview")
    st.dataframe(clean_df.head(10), use_container_width=True)

    if st.button("Save cleaned dataset"):
        save_processed_data(clean_df)
        st.success("Saved to data/processed/heart_clean.csv")
