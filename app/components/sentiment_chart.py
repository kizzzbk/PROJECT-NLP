"""
Sentiment Chart Component
===========================
Interactive Plotly bar chart with click event for drill-down (FR-3.3).
"""

from typing import Optional

import streamlit as st
import plotly.graph_objects as go


def render_sentiment_chart(
    positive_count: int,
    negative_count: int,
) -> Optional[str]:
    """
    Render an interactive 2-bar chart (Positive vs Negative).
    
    When user clicks on a bar, returns the selected sentiment label
    so the parent can filter the data table accordingly.
    
    Args:
        positive_count: Number of positive comments.
        negative_count: Number of negative comments.
        
    Returns:
        Selected sentiment label ("Tích cực" or "Tiêu cực") or None.
    """
    total = positive_count + negative_count
    
    fig = go.Figure()
    
    # Positive bar
    fig.add_trace(
        go.Bar(
            x=["Tích cực"],
            y=[positive_count],
            name="Tích cực",
            marker_color="#10b981",
            marker_line_color="#059669",
            marker_line_width=2,
            text=[f"{positive_count:,}<br>({positive_count/total:.1%})"],
            textposition="outside",
            textfont=dict(size=14, color="#10b981"),
            hovertemplate="<b>Tích cực</b><br>Số lượng: %{y:,}<br>Tỷ lệ: " + f"{positive_count/total:.1%}" + "<extra></extra>",
            customdata=["Tích cực"],
        )
    )
    
    # Negative bar
    fig.add_trace(
        go.Bar(
            x=["Tiêu cực"],
            y=[negative_count],
            name="Tiêu cực",
            marker_color="#ef4444",
            marker_line_color="#dc2626",
            marker_line_width=2,
            text=[f"{negative_count:,}<br>({negative_count/total:.1%})"],
            textposition="outside",
            textfont=dict(size=14, color="#ef4444"),
            hovertemplate="<b>Tiêu cực</b><br>Số lượng: %{y:,}<br>Tỷ lệ: " + f"{negative_count/total:.1%}" + "<extra></extra>",
            customdata=["Tiêu cực"],
        )
    )
    
    fig.update_layout(
        height=420,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=20, b=40),
        yaxis=dict(
            title="Số lượng bình luận",
            gridcolor="rgba(148, 163, 184, 0.1)",
            zeroline=False,
        ),
        xaxis=dict(
            tickfont=dict(size=14, color="#e2e8f0"),
        ),
        bargap=0.4,
    )
    
    # Render with selection support
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        key="sentiment_bar_chart",
    )
    
    # Process click event
    selected_sentiment = None
    
    if event and "selection" in event:
        selection = event["selection"]
        points = selection.get("points", [])
        
        if points:
            point = points[0]
            # Get the x-axis label of the clicked bar
            x_label = point.get("x", None)
            if x_label in ["Tích cực", "Tiêu cực"]:
                selected_sentiment = x_label
    
    # Also provide button-based fallback
    st.markdown(
        "<p style='text-align: center; color: #64748b; font-size: 0.8rem;'>"
        "👆 Click trực tiếp vào cột hoặc dùng nút bên dưới</p>",
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
