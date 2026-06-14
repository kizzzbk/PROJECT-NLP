"""
Model Selector Component
=========================
Dropdown to select and load a sentiment model.
Uses st.cache_resource for efficient model loading.
"""

import streamlit as st
from pathlib import Path
from typing import Optional, Tuple


@st.cache_resource
def _load_predictor(model_name: str, checkpoint_path: str):
    """Cache-load a predictor to avoid reloading on every interaction."""
    from src.inference.predictor import SentimentPredictor
    return SentimentPredictor(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device="auto",
    )


def render_model_selector(key_prefix: str = "single") -> Tuple[str, Optional[object]]:
    """
    Render model selection dropdown and load the selected model.
    
    Args:
        key_prefix: Unique key prefix for Streamlit widgets.
        
    Returns:
        Tuple of (model_display_name, predictor_instance_or_None).
    """
    from src.inference.model_registry import ModelRegistry
    
    MODEL_DISPLAY = {
        "bilstm": "BiLSTM (Baseline)",
        "bilstm_attention": "BiLSTM + Attention",
    }

    
    # Check which models are available
    registry = ModelRegistry("models")
    available = registry.get_available_model_types()
    
    all_models = list(MODEL_DISPLAY.keys())
    
    # Default to first available, or first overall
    default_idx = 0
    if available:
        for i, m in enumerate(all_models):
            if m in available:
                default_idx = i
                break
    
    selected = st.selectbox(
        "🤖 Chọn Mô hình AI",
        options=all_models,
        format_func=lambda x: f"{MODEL_DISPLAY[x]} {'✅' if x in available else '⚠️ (chưa train)'}",
        index=default_idx,
        key=f"{key_prefix}_model_select",
    )
    
    display_name = MODEL_DISPLAY[selected]
    
    # Load predictor
    predictor = None
    if selected in available:
        checkpoint = registry.get_best_checkpoint(selected)
        if checkpoint:
            try:
                predictor = _load_predictor(selected, checkpoint)
            except Exception as e:
                st.error(f"❌ Lỗi tải mô hình: {e}")
    else:
        st.warning(
            f"⚠️ Mô hình **{display_name}** chưa được huấn luyện. "
            f"Chạy `python scripts/train.py --model {selected}` để train."
        )
    
    return display_name, predictor
