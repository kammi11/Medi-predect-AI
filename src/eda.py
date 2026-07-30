"""EDA chart-generation functions. Each returns a Plotly figure object -
pages call these and render with st.plotly_chart(). No st.* calls in here."""

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

from src.config import NUMERIC_FEATURES, TARGET_COLUMN, FEATURE_LABELS


def plot_histogram(df: pd.DataFrame, column: str):
    return px.histogram(
        df, x=column, color=TARGET_COLUMN, barmode="overlay", nbins=30,
        title=f"Distribution of {FEATURE_LABELS.get(column, column)}",
        labels={column: FEATURE_LABELS.get(column, column)},
    )


def plot_boxplot(df: pd.DataFrame, column: str):
    return px.box(
        df, y=column, x=TARGET_COLUMN, color=TARGET_COLUMN,
        title=f"{FEATURE_LABELS.get(column, column)} by disease outcome",
        labels={column: FEATURE_LABELS.get(column, column), TARGET_COLUMN: "Disease"},
    )


def plot_correlation_heatmap(df: pd.DataFrame):
    corr = df.corr(numeric_only=True)
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Feature correlation heatmap",
    )
    return fig


def plot_countplot(df: pd.DataFrame, column: str):
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "count"]
    return px.bar(
        counts, x=column, y="count",
        title=f"Count of {FEATURE_LABELS.get(column, column)} categories",
    )


def plot_target_pie(df: pd.DataFrame):
    counts = df[TARGET_COLUMN].value_counts().reset_index()
    counts.columns = [TARGET_COLUMN, "count"]
    counts[TARGET_COLUMN] = counts[TARGET_COLUMN].map({0: "No disease", 1: "Disease"})
    return px.pie(counts, names=TARGET_COLUMN, values="count", title="Target class balance")


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str):
    return px.scatter(
        df, x=x_col, y=y_col, color=TARGET_COLUMN,
        title=f"{FEATURE_LABELS.get(x_col, x_col)} vs {FEATURE_LABELS.get(y_col, y_col)}",
    )


def plot_pairplot(df: pd.DataFrame, columns: list[str] | None = None):
    if columns is None:
        columns = NUMERIC_FEATURES
    return px.scatter_matrix(
        df, dimensions=columns, color=TARGET_COLUMN,
        title="Pairwise feature relationships",
    )


def plot_distribution(df: pd.DataFrame, column: str):
    grouped = [df[df[TARGET_COLUMN] == cls][column].dropna() for cls in sorted(df[TARGET_COLUMN].unique())]
    labels = [f"Target = {cls}" for cls in sorted(df[TARGET_COLUMN].unique())]
    return ff.create_distplot(grouped, labels, show_hist=False)
