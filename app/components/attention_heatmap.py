"""
Attention Heatmap Component
=============================
Visualize attention weights as a color-coded heatmap.
Shows which words the model focused on for its prediction.
"""

from typing import List, Optional

import streamlit as st
import plotly.graph_objects as go
import numpy as np


def render_attention_heatmap(
    tokens: List[str],
    weights: List[float],
    title: str = "Attention Weights",
    max_tokens: int = 50,
):
    """
    Render an attention heatmap using Plotly.
    
    Each token is displayed with a colored background indicating
    its attention weight (darker = higher weight = more important).
    
    Args:
        tokens: List of word tokens.
        weights: Corresponding attention weights (0-1).
        title: Chart title.
        max_tokens: Maximum number of tokens to display.
    """
    if not tokens or not weights:
        st.info("Không có dữ liệu Attention để hiển thị.")
        return
    
    # Truncate if too long
    tokens = tokens[:max_tokens]
    weights = weights[:max_tokens]
    
    # Normalize weights to [0, 1]
    weights_arr = np.array(weights, dtype=float)
    if weights_arr.max() > 0:
        weights_norm = weights_arr / weights_arr.max()
    else:
        weights_norm = weights_arr
    
    # Create heatmap
    fig = go.Figure()
    
    # Single-row heatmap
    fig.add_trace(
        go.Heatmap(
            z=[weights_norm.tolist()],
            x=tokens,
            y=["Attention"],
            colorscale=[
                [0, "rgba(30, 41, 59, 0.3)"],    # Low attention: dark
                [0.5, "rgba(251, 191, 36, 0.6)"],  # Medium: amber
                [1, "rgba(239, 68, 68, 1.0)"],     # High attention: red
            ],
            showscale=True,
            colorbar=dict(
                title="Trọng số",
                titleside="right",
                thickness=15,
                len=0.8,
            ),
            text=[[f"{w:.3f}" for w in weights]],
            texttemplate="%{text}",
            textfont=dict(size=11, color="white"),
            hovertemplate="<b>%{x}</b><br>Attention: %{z:.4f}<extra></extra>",
        )
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=120,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=11),
            side="bottom",
        ),
        yaxis=dict(visible=False),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Also show text with inline highlighting
    st.markdown("**📝 Văn bản tô màu theo trọng số:**")
    
    html_parts = []
    for token, weight in zip(tokens, weights_norm):
        # Map weight to color intensity
        r = int(255 * weight)
        g = int(100 * (1 - weight))
        b = int(50 * (1 - weight))
        alpha = 0.2 + 0.6 * weight
        
        font_weight = "bold" if weight > 0.5 else "normal"
        font_size = f"{0.9 + 0.4 * weight:.1f}rem"
        
        html_parts.append(
            f'<span style="background: rgba({r},{g},{b},{alpha:.2f}); '
            f'padding: 2px 4px; margin: 1px; border-radius: 4px; '
            f'font-weight: {font_weight}; font-size: {font_size}; '
            f'display: inline-block;">{token}</span>'
        )
    
    st.markdown(
        f'<div style="line-height: 2.2; padding: 0.5rem;">{"".join(html_parts)}</div>',
        unsafe_allow_html=True,
    )
