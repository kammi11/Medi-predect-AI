import streamlit as st

from src.eda import (
    plot_histogram, plot_boxplot, plot_correlation_heatmap,
    plot_countplot, plot_target_pie, plot_scatter, plot_distribution,
)
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.ui import load_css, page_header

load_css()
page_header("📊", "Exploratory Data Analysis", "Interactive charts to understand distributions, correlations, and patterns.")

df = st.session_state.get("clean_df")
if df is None:
    df = st.session_state.get("raw_df")
if df is None:
    st.warning("Please load and clean a dataset first.")
    st.stop()

st.subheader("Target class balance")
st.plotly_chart(plot_target_pie(df), use_container_width=True)
st.caption("A balanced target means accuracy will be a meaningful metric alongside precision/recall.")

st.subheader("Correlation heatmap")
st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)
st.caption("Look for features strongly correlated with `target` — these tend to be the most predictive.")

st.subheader("Feature distribution")
num_col = st.selectbox("Numeric feature", NUMERIC_FEATURES)
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(plot_histogram(df, num_col), use_container_width=True)
with c2:
    st.plotly_chart(plot_boxplot(df, num_col), use_container_width=True)

st.subheader("Categorical feature counts")
cat_col = st.selectbox("Categorical feature", CATEGORICAL_FEATURES)
st.plotly_chart(plot_countplot(df, cat_col), use_container_width=True)

st.subheader("Feature relationships")
c1, c2 = st.columns(2)
x_col = c1.selectbox("X axis", NUMERIC_FEATURES, index=0)
y_col = c2.selectbox("Y axis", NUMERIC_FEATURES, index=1)
st.plotly_chart(plot_scatter(df, x_col, y_col), use_container_width=True)

st.subheader("Distribution by outcome")
st.plotly_chart(plot_distribution(df, num_col), use_container_width=True)
