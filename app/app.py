"""
BrandHealth AI — Web Dashboard
================================
Main Streamlit entry point.
"""

import sys
from pathlib import Path

# Add project root to path and remove script directory to avoid naming collision with 'app' package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

cleaned_path = []
for p in sys.path:
    if not p:
        continue
    try:
        if Path(p).resolve() == SCRIPT_DIR:
            continue
    except Exception:
        pass
    cleaned_path.append(p)

sys.path = cleaned_path
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# Page config — must be first Streamlit command
st.set_page_config(
    page_title="BrandHealth AI — Giám sát Sức khỏe Thương hiệu",
    page_icon="",
    layout="centered",  # Centered layout looks much cleaner, minimal and elegant
    initial_sidebar_state="collapsed",
)

# Load custom CSS
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# CSS to completely hide the sidebar toggle button for a clean single-page look
st.markdown(
    """
    <style>
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        section[data-testid="stSidebar"] {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

from app.pages.batch_analysis import render_batch_analysis


def main():
    # Header Section
    st.markdown(
        """
        <div style="text-align: center; margin-top: 1rem; margin-bottom: 2rem;">
            <h1 style="font-size: 2.5rem; font-weight: 800; color: #ffffff; margin-bottom: 0.5rem;">BrandHealth AI</h1>
            <p style="font-size: 1.1rem; color: #ffffff; font-weight: 400;">
                Ứng dụng Giám sát Sức khỏe Thương hiệu từ Phản hồi Khách hàng
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Render the main analysis content
    render_batch_analysis()


if __name__ == "__main__":
    main()
