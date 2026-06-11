"""
Comment Table Component
========================
Dynamic data table that shows filtered comments based on chart selection.
"""

import streamlit as st
import pandas as pd


def render_comment_table(df: pd.DataFrame, max_display: int = 100):
    """
    Render a filtered data table of comments.
    
    Args:
        df: DataFrame with columns ['Bình luận gốc', 'Sắc thái', 'Độ tự tin'].
        max_display: Maximum number of rows to display.
    """
    if df.empty:
        st.info("📭 Không có bình luận nào trong nhóm này.")
        return
    
    total = len(df)
    
    # Show count
    st.markdown(
        f"<p style='color: #94a3b8; font-size: 0.85rem;'>"
        f"Hiển thị {min(total, max_display)} / {total:,} bình luận"
        f"</p>",
        unsafe_allow_html=True,
    )
    
    # Sort by confidence (most confident first)
    display_df = df.sort_values("Độ tự tin", ascending=False).head(max_display)
    
    # Format for display
    display_df = display_df[["Bình luận gốc", "Sắc thái", "Độ tự tin"]].copy()
    display_df["Độ tự tin"] = display_df["Độ tự tin"].apply(lambda x: f"{x:.1%}")
    
    # Add sentiment emoji
    display_df["Sắc thái"] = display_df["Sắc thái"].apply(
        lambda x: f"😊 {x}" if "Tích" in str(x) else f"😠 {x}"
    )
    
    # Render with Streamlit dataframe
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        column_config={
            "Bình luận gốc": st.column_config.TextColumn(
                "📝 Bình luận",
                width="large",
            ),
            "Sắc thái": st.column_config.TextColumn(
                "🏷️ Sắc thái",
                width="small",
            ),
            "Độ tự tin": st.column_config.TextColumn(
                "📊 Độ tự tin",
                width="small",
            ),
        },
        hide_index=True,
    )
