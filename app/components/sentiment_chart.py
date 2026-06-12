"""
Sentiment Chart Component
===========================
Interactive Plotly PIE chart with click event for drill-down (FR-3.2).
PRD Section 6.2: Biểu đồ tròn tương tác hiển thị tỷ lệ % Tích cực / Tiêu cực.
"""

from typing import Optional

import streamlit as st
import plotly.graph_objects as go


def render_sentiment_chart(
    positive_count: int,
    negative_count: int,
) -> Optional[str]:
    """
    Render an interactive pie chart (Positive vs Negative).
    
    When user clicks on a slice, returns the selected sentiment label
    so the parent can filter the data table accordingly.
    
    Args:
        positive_count: Number of positive comments.
        negative_count: Number of negative comments.
        
    Returns:
        Selected sentiment label ("Tích cực" or "Tiêu cực") or None.
    """
    total = positive_count + negative_count
    
    fig = go.Figure()
    
    # Pie chart with 2 slices
    fig.add_trace(
        go.Pie(
            labels=["Tích cực", "Tiêu cực"],
            values=[positive_count, negative_count],
            marker=dict(
                colors=["#10b981", "#ef4444"],
                line=dict(color="#0e1117", width=3),
            ),
            textinfo="label+percent",
            textfont=dict(size=15, color="white"),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Số lượng: %{value:,}<br>"
                "Tỷ lệ: %{percent}<extra></extra>"
            ),
            hole=0.35,  # donut style
            pull=[0.02, 0.02],
            customdata=["Tích cực", "Tiêu cực"],
        )
    )
    
    fig.update_layout(
        height=420,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=13, color="#e2e8f0"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=50),
        annotations=[
            dict(
                text=f"<b>{total:,}</b><br>bình luận",
                x=0.5, y=0.5,
                font=dict(size=16, color="#e2e8f0"),
                showarrow=False,
            )
        ],
    )
    
    # Render with selection support
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        key="sentiment_pie_chart",
    )
    
    # Process click event
    selected_sentiment = None
    
    if event and "selection" in event:
        selection = event["selection"]
        points = selection.get("points", [])
        
        if points:
            point = points[0]
            # Get the label of the clicked slice
            label = point.get("label", None)
            if label in ["Tích cực", "Tiêu cực"]:
                selected_sentiment = label
    
    # Also provide button-based fallback
    st.markdown(
        "<p style='text-align: center; color: #64748b; font-size: 0.8rem;'>"
        "👆 Click vào miếng bánh hoặc dùng nút bên dưới</p>",
        unsafe_allow_html=True,
    )
    
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    
    with btn_col1:
        if st.button("😊 Tích cực", use_container_width=True, key="btn_pos"):
            selected_sentiment = "Tích cực"
    with btn_col2:
        if st.button("😠 Tiêu cực", use_container_width=True, key="btn_neg"):
            selected_sentiment = "Tiêu cực"
    with btn_col3:
        if st.button("📋 Tất cả", use_container_width=True, key="btn_all"):
            selected_sentiment = None
    
    return selected_sentiment
