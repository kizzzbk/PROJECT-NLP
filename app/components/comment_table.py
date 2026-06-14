"""
Comment Table Component
========================
Dynamic data table that shows filtered comments with Heatmap HTML highlighting.
PRD FR-3.3: Sắp xếp theo mức độ phân cực xác suất + Heatmap văn bản.

Sorting logic:
  - Khi filter "Tích cực": sắp từ prob gần 1.0 nhất xuống 0.5 (tích cực nhất lên đầu)
  - Khi filter "Tiêu cực": sắp từ prob gần 0.0 nhất lên 0.5 (tiêu cực nhất lên đầu)
  - Khi hiển thị tất cả: sắp theo abs(prob - 0.5) giảm dần (phân cực rõ nhất lên đầu)
"""

from typing import List, Optional

import streamlit as st
import pandas as pd
import numpy as np


def _build_heatmap_html(tokens: List[str], weights: List[float]) -> str:
    """
    Build inline HTML with color-coded token highlighting.
    
    Words with higher attention weight are highlighted more intensely.
    Uses warm tones (yellow→red) for emphasis.
    """
    if not tokens or not weights:
        return " ".join(tokens) if tokens else ""
    
    # Normalize weights to [0, 1]
    weights_arr = np.array(weights[:len(tokens)], dtype=float)
    w_max = weights_arr.max()
    if w_max > 0:
        weights_norm = weights_arr / w_max
    else:
        weights_norm = weights_arr
    
    html_parts = []
    for token, w in zip(tokens, weights_norm):
        if w > 0.6:
            # High importance: bold, red/orange background
            r, g, b = 239, 68, 68
            alpha = 0.3 + 0.5 * w
            style = (
                f"background:rgba({r},{g},{b},{alpha:.2f});"
                f"font-weight:bold;"
                f"padding:1px 3px;border-radius:3px;"
            )
        elif w > 0.3:
            # Medium importance: amber background
            r, g, b = 251, 191, 36
            alpha = 0.2 + 0.3 * w
            style = (
                f"background:rgba({r},{g},{b},{alpha:.2f});"
                f"padding:1px 2px;border-radius:3px;"
            )
        else:
            # Low importance: no highlight
            style = ""
        
        if style:
            html_parts.append(f'<span style="{style}">{token}</span>')
        else:
            html_parts.append(token)
    
    return " ".join(html_parts)


def render_comment_table(
    df: pd.DataFrame,
    selected_sentiment: Optional[str] = None,
    max_display: int = 100,
):
    """
    Render a filtered data table of comments with Heatmap highlighting.
    
    Sorting:
      - "Tích cực" selected: sort by prob_class_0 ascending (prob near 1.0 first)
      - "Tiêu cực" selected: sort by prob_class_0 ascending (prob near 0.0 first)
      - All: sort by |prob - 0.5| descending (most polarized first)
    
    Args:
        df: DataFrame with columns including 'Bình luận gốc', 'Sắc thái', 
            'prob_class_0', and optionally 'attention_weights', 'tokens'.
        selected_sentiment: Current filter ("Tích cực", "Tiêu cực", or None).
        max_display: Maximum number of rows to display.
    """
    if df.empty:
        st.info("📭 Không có bình luận nào trong nhóm này.")
        return
    
    total = len(df)
    
    # Sort based on selected sentiment
    display_df = df.copy()
    
    if "prob_class_0" in display_df.columns:
        if selected_sentiment == "Tích cực":
            # Most positive first: highest prob_class_0 (= prob of class 1 = prob positive)
            # prob_class_0 is probability of label 0 (negative)
            # So positive = 1 - prob_class_0, want highest first = lowest prob_class_0
            display_df = display_df.sort_values("prob_class_0", ascending=True)
        elif selected_sentiment == "Tiêu cực":
            # Most negative first: lowest prob (closest to 0.0) first
            # prob_class_0 is probability of label 0 (negative)
            # Highest prob_class_0 = most negative
            display_df = display_df.sort_values("prob_class_0", ascending=False)
        else:
            # All: most polarized first (furthest from 0.5)
            display_df["_polarity"] = (display_df["prob_class_0"] - 0.5).abs()
            display_df = display_df.sort_values("_polarity", ascending=False)
            display_df = display_df.drop(columns=["_polarity"])
    
    display_df = display_df.head(max_display)
    
    # Show count
    st.markdown(
        f"<p style='color: #94a3b8; font-size: 0.85rem;'>"
        f"Hiển thị {min(total, max_display)} / {total:,} bình luận"
        f"</p>",
        unsafe_allow_html=True,
    )
    
    # Build HTML table with Heatmap
    has_attention = ("attention_weights" in display_df.columns and 
                     "tokens" in display_df.columns)
    
    # Table header
    html = """
<div style="max-height: 500px; overflow-y: auto; border-radius: 8px; 
            border: 1px solid #ddd6fe;">
<table style="width: 100%; border-collapse: collapse; font-size: 0.9rem; background-color: #ffffff;">
<thead>
    <tr style="background: #4c1d95; position: sticky; top: 0; z-index: 1;">
        <th style="padding: 10px 12px; text-align: left; color: #ffffff; 
                   border-bottom: 2px solid #ddd6fe; width: 5%;">#</th>
        <th style="padding: 10px 12px; text-align: left; color: #ffffff; 
                   border-bottom: 2px solid #ddd6fe; width: 60%;">📝 Bình luận</th>
        <th style="padding: 10px 12px; text-align: center; color: #ffffff; 
                   border-bottom: 2px solid #ddd6fe; width: 15%;">🏷️ Sắc thái</th>
        <th style="padding: 10px 12px; text-align: center; color: #ffffff; 
                   border-bottom: 2px solid #ddd6fe; width: 20%;">📊 Xác suất</th>
    </tr>
</thead>
<tbody>
"""
    
    for idx, (_, row) in enumerate(display_df.iterrows(), 1):
        # Build heatmap text
        if has_attention and row.get("attention_weights") and row.get("tokens"):
            tokens = row["tokens"]
            weights = row["attention_weights"]
            if isinstance(tokens, str):
                import json
                try:
                    tokens = json.loads(tokens)
                    weights = json.loads(weights)
                except (json.JSONDecodeError, TypeError):
                    tokens = row["Bình luận gốc"].split()
                    weights = []
            heatmap_text = _build_heatmap_html(tokens, weights)
        else:
            heatmap_text = row["Bình luận gốc"]
        
        # Sentiment badge
        sentiment = row["Sắc thái"]
        if "Tích" in str(sentiment):
            badge_color = "#10b981"
            badge_bg = "rgba(16, 185, 129, 0.15)"
            emoji = "😊"
        else:
            badge_color = "#ef4444"
            badge_bg = "rgba(239, 68, 68, 0.15)"
            emoji = "😠"
        
        # Probability display
        prob_neg = row.get("prob_class_0", 0)
        prob_pos = 1 - prob_neg if prob_neg else row.get("Độ tự tin", 0)
        
        if "Tích" in str(sentiment):
            prob_display = f"{prob_pos:.1%}"
        else:
            prob_display = f"{prob_neg:.1%}"
        
        # Row background (alternating soft purple)
        row_bg = "#f5f3ff" if idx % 2 == 0 else "#ffffff"
        
        html += f"""
<tr style="background: {row_bg}; border-bottom: 1px solid #ddd6fe;">
    <td style="padding: 8px 12px; color: #7c3aed; font-size: 0.8rem; font-weight: 500;">{idx}</td>
    <td style="padding: 8px 12px; color: #311066; line-height: 1.6; font-weight: 500;">{heatmap_text}</td>
    <td style="padding: 8px 12px; text-align: center;">
        <span style="background: {badge_bg}; color: {badge_color}; 
                     padding: 3px 10px; border-radius: 12px; font-size: 0.8rem;
                     font-weight: 600;">
            {emoji} {sentiment}
        </span>
    </td>
    <td style="padding: 8px 12px; text-align: center; color: #4c1d95; 
               font-weight: 600;">{prob_display}</td>
</tr>
"""
    
    html += """
</tbody>
</table>
</div>
"""
    
    # Strip all leading spaces from each line in the HTML block to prevent markdown code block formatting
    cleaned_html = "\n".join([line.strip() for line in html.split("\n")])
    st.markdown(cleaned_html, unsafe_allow_html=True)

