import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.evaluate import evaluate_all_models, build_comparison_table, select_best_model, get_feature_importance
from src.model_utils import save_model
from src.ui import load_css, page_header

load_css()
page_header("📈", "Model Evaluation", "Compare accuracy, precision, recall, F1, and ROC-AUC across all trained models.")

if "trained_models" not in st.session_state:
    st.warning("Please train models first.")
    st.stop()

if st.button("Evaluate all models", type="primary"):
    with st.spinner("Evaluating..."):
        results = evaluate_all_models(
            st.session_state["trained_models"], st.session_state["X_test"], st.session_state["y_test"]
        )
        best_name = select_best_model(results)
        save_model(st.session_state["trained_models"][best_name], "best_model")

    st.session_state["eval_results"] = results
    st.session_state["best_model_name"] = best_name
    st.success(f"Best model: {best_name} (saved as models/best_model.joblib)")

if "eval_results" in st.session_state:
    results = st.session_state["eval_results"]
    best_name = st.session_state["best_model_name"]

    st.subheader("Model comparison")
    comparison_df = build_comparison_table(results)
    st.dataframe(comparison_df, use_container_width=True)
    st.info(f"🏆 Best model (by F1 score): **{best_name}**")

    st.subheader("Confusion matrix & ROC curve")
    selected_model = st.selectbox("Select a model", list(results.keys()), index=list(results.keys()).index(best_name))
    metrics = results[selected_model]

    c1, c2 = st.columns(2)
    with c1:
        cm = metrics["confusion_matrix"]
        fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                            x=["No disease", "Disease"], y=["No disease", "Disease"],
                            title=f"Confusion matrix - {selected_model}")
        st.plotly_chart(fig_cm, use_container_width=True)

    with c2:
        fpr, tpr, _ = metrics["roc_curve"]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=selected_model))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
        fig_roc.update_layout(title=f"ROC curve - {selected_model} (AUC={metrics['roc_auc']:.3f})",
                               xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    st.subheader("Classification report")
    st.json(metrics["classification_report"])

    importance_df = get_feature_importance(
        st.session_state["trained_models"][selected_model], st.session_state["X_train"].columns.tolist()
    )
    if importance_df is not None:
        st.subheader("Feature importance")
        st.plotly_chart(px.bar(importance_df, x="importance", y="feature", orientation="h"), use_container_width=True)
