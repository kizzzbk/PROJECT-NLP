"""
BrandHealth AI — Web Dashboard
================================
Main Streamlit entry point.
PRD Section 6: Single-page Dashboard with batch CSV analysis.
No login system (PRD Section 4.2 Out-of-Scope).

Run: streamlit run app/app.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# Page config — must be first Streamlit command
st.set_page_config(
    page_title="BrandHealth AI — Giám sát Sức khỏe Thương hiệu",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from app.pages.batch_analysis import render_batch_analysis


def main():
    # Sidebar
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0;">
                <h1 style="font-size: 1.8rem; margin-bottom: 0.2rem;">🏥 BrandHealth AI</h1>
                <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 0;">
                    Giám sát Sức khỏe Thương hiệu
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.divider()
        
        # System info
        st.markdown(
            """
            <div style="padding: 0.5rem; background: rgba(102, 126, 234, 0.1); 
                        border-radius: 8px; font-size: 0.8rem;">
                <p style="margin: 0.2rem 0;">📐 <b>Mô hình hỗ trợ:</b></p>
                <ul style="margin: 0.2rem 0; padding-left: 1.2rem;">
                    <li>BiLSTM (Baseline)</li>
                    <li>BiLSTM + Attention</li>
                    <li>PhoBERT Fine-tuned</li>
                </ul>
                <p style="margin: 0.5rem 0 0.2rem;">
                    💡 <b>Dữ liệu:</b> NTC-SCV
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.divider()
        st.caption("© 2026 BrandHealth AI Pipeline v2.0")
    
    # Main content — single-page batch analysis (PRD Section 4.2)
    render_batch_analysis()


if __name__ == "__main__":
    main()
