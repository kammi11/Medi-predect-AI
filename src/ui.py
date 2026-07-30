"""Shared UI components for a consistent, professional look across all pages.
Pure presentation helpers -- no ML logic lives here."""

import streamlit as st
from pathlib import Path

CSS_PATH = Path(__file__).resolve().parent.parent / "assets" / "style.css"


def load_css() -> None:
    """Inject the shared stylesheet. Call once at the top of every page."""
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


def page_header(icon: str, title: str, subtitle: str = "") -> None:
    """Render a consistent icon + title + subtitle header block."""
    st.markdown(
        f"""
        <div class="mp-page-header">
            <div class="mp-icon">{icon}</div>
            <h1>{title}</h1>
        </div>
        {f'<p class="mp-page-subtitle">{subtitle}</p>' if subtitle else ''}
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    """Render the large gradient hero banner used on the Home page."""
    st.markdown(
        f"""
        <div class="mp-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str) -> str:
    """Return HTML for one stat card. Use inside st.columns for a row of stats."""
    return f"""
    <div class="mp-stat">
        <div class="mp-stat-label">{label}</div>
        <div class="mp-stat-value">{value}</div>
    </div>
    """


def feature_card(step_num: str, title: str, description: str) -> str:
    """Return HTML for one workflow step card. Use inside st.columns."""
    return f"""
    <div class="mp-card">
        <span class="mp-step-num">Step {step_num}</span>
        <h4>{title}</h4>
        <p>{description}</p>
    </div>
    """


def risk_badge(risk_level: str) -> str:
    """Return HTML for a color-coded risk level badge."""
    css_class = {"Low": "mp-badge-low", "Moderate": "mp-badge-moderate", "High": "mp-badge-high"}[risk_level]
    return f'<span class="mp-badge {css_class}">{risk_level} risk</span>'


def disclaimer(text: str = "This tool is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.") -> None:
    st.markdown(f'<div class="mp-disclaimer">⚠️ {text}</div>', unsafe_allow_html=True)
