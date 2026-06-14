"""
Sentiment Chart Component
===========================
Simple Plotly PIE chart showing sentiment distribution (FR-3.2).
"""

import streamlit as st
import plotly.graph_objects as go


def render_sentiment_chart(
    positive_count: int,
    negative_count: int,
):
    """
    Render a static/simple pie chart (Positive vs Negative).
    
    Args:
        positive_count: Number of positive comments.
        negative_count: Number of negative comments.
    """
    total = positive_count + negative_count
    if total == 0:
        st.info("Chưa có dữ liệu để vẽ biểu đồ.")
        return
        
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
            textfont=dict(size=14, color="white"),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Số lượng: %{value:,}<br>"
                "Tỷ lệ: %{percent}<extra></extra>"
            ),
            hole=0.35,  # donut style
            pull=[0.02, 0.02],
        )
    )
    
    fig.update_layout(
        height=380,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=13, color="#311066"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=50),
        annotations=[
            dict(
                text=f"<b>{total:,}</b><br>bình luận",
                x=0.5, y=0.5,
                font=dict(size=14, color="#311066"),
                showarrow=False,
            )
        ],
    )
    
    st.plotly_chart(
        fig,
        use_container_width=True,
        key="sentiment_pie_chart",
    )

