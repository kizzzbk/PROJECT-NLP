"""
Executive Dashboard — Batch CSV Analysis
==========================================
PRD Section 6.2: Single-page dashboard with Pie Chart + Drill-down + Heatmap table.
Implements FR-3.2 (interactive pie chart) and FR-3.3 (heatmap table).
"""

import streamlit as st
import pandas as pd

from app.components.model_selector import render_model_selector
from app.components.sentiment_chart import render_sentiment_chart
from app.components.comment_table import render_comment_table


def render_batch_analysis():
    """Render the batch analysis page (main dashboard)."""
    # Model selector
    model_name, predictor = render_model_selector(key_prefix="batch")
    
    st.markdown("<br>", unsafe_allow_html=True)

    
    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Tải file CSV chứa bình luận",
        type=["csv"],
        help="File CSV cần có cột chứa văn bản bình luận (text, comment, review...)",
    )
    
    if uploaded_file is not None and predictor is not None:
        # Read CSV
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except Exception:
            try:
                df = pd.read_csv(uploaded_file, encoding="latin-1")
            except Exception as e:
                st.error(f"❌ Không đọc được file CSV: {e}")
                return
        
        # Auto-detect text column
        text_column = _detect_text_column(df)
        
        if text_column is None:
            st.warning("⚠️ Không tìm thấy cột văn bản. Vui lòng chọn:")
            text_column = st.selectbox("Chọn cột chứa bình luận:", df.columns)
        
        st.info(f"📋 File: **{uploaded_file.name}** | Số dòng: **{len(df):,}** | Cột văn bản: **{text_column}**")
        
        # Process button
        if st.button("🚀 Phân tích Hàng loạt", type="primary", use_container_width=True):
            _process_batch(df, text_column, predictor)
        
        # Show results if already processed
        if "batch_results" in st.session_state:
            _display_results()
    
    elif uploaded_file is not None and predictor is None:
        st.error("❌ Chưa có mô hình nào được huấn luyện. Vui lòng train mô hình trước.")
    
    else:
        # Show demo prompt
        st.markdown(
            """
            <div style="text-align: center; padding: 3rem; border: 2px dashed #334155; 
                        border-radius: 12px; margin: 2rem 0;">
                <span style="font-size: 3rem;">📂</span>
                <h3 style="margin-top: 1rem;">Kéo thả file CSV vào đây</h3>
                <p style="color: #94a3b8;">
                    Hỗ trợ file .csv chứa các bình luận của khách hàng.<br>
                    Hệ thống sẽ tự động phân loại Tích cực / Tiêu cực.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _detect_text_column(df: pd.DataFrame) -> str:
    """Auto-detect the text column in a DataFrame."""
    candidates = ["text", "comment", "review", "content", "bình luận", "sentence", "feedback"]
    
    for col in df.columns:
        if col.lower().strip() in candidates:
            return col
    
    # Fallback: find the column with the longest average string length
    str_cols = df.select_dtypes(include=["object"]).columns
    if len(str_cols) > 0:
        avg_lengths = {col: df[col].astype(str).str.len().mean() for col in str_cols}
        return max(avg_lengths, key=avg_lengths.get)
    
    return None


def _process_batch(df: pd.DataFrame, text_column: str, predictor):
    """Process all texts in the dataframe. Store prob, attention, tokens."""
    texts = df[text_column].astype(str).tolist()
    total = len(texts)
    
    progress_bar = st.progress(0, text="🔄 Đang phân tích...")
    results = []
    
    for i, text in enumerate(texts):
        result = predictor.predict_single(text)
        
        row = {
            "Bình luận gốc": text,
            "Bình luận đã xử lý": result["cleaned_text"],
            "Sắc thái": result["label"],
            "Độ tự tin": result["confidence"],
            "prob_class_0": result["probabilities"].get("Tiêu cực", 0.0),
        }
        
        # Store attention data for heatmap rendering
        if result.get("attention_weights"):
            row["attention_weights"] = result["attention_weights"]
            row["tokens"] = result["tokens"]
        else:
            row["attention_weights"] = None
            row["tokens"] = None
        
        results.append(row)
        
        if (i + 1) % max(1, total // 100) == 0:
            progress_bar.progress(
                (i + 1) / total,
                text=f"🔄 Đang phân tích... ({i + 1}/{total})",
            )
    
    progress_bar.progress(1.0, text="✅ Hoàn thành!")
    
    results_df = pd.DataFrame(results)
    st.session_state.batch_results = results_df
    st.success(f"✅ Đã phân tích thành công **{total:,}** dòng bình luận!")
    st.rerun()


def _display_results():
    """Display batch analysis results with pie chart and tabbed positive/negative tables."""
    results_df = st.session_state.batch_results
    
    # Summary metrics
    total = len(results_df)
    positive_count = (results_df["Sắc thái"] == "Tích cực").sum()
    negative_count = (results_df["Sắc thái"] == "Tiêu cực").sum()
    avg_confidence = results_df["Độ tự tin"].mean()
    
    # Top section: Metrics on the left, Pie chart on the right
    col_metrics, col_chart = st.columns([1, 1])
    
    with col_metrics:
        st.markdown("### 📈 Chỉ số Tổng quan")
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("📝 Tổng bình luận", f"{total:,}")
        st.metric("😊 Bình luận Tích cực", f"{positive_count:,}", f"{positive_count/total:.1%}")
        st.metric("😠 Bình luận Tiêu cực", f"{negative_count:,}", f"-{negative_count/total:.1%}", delta_color="inverse")
        st.metric("📊 Độ tự tin trung bình", f"{avg_confidence:.1%}")
    
    with col_chart:
        st.markdown("### 🥧 Biểu đồ Tỷ lệ Sắc thái")
        render_sentiment_chart(positive_count, negative_count)
    
    st.divider()
    
    # Bottom section: Tabs for Positive and Negative comments
    st.markdown("### 🔍 Chi tiết Bình luận của 2 nhóm")
    tab1, tab2 = st.tabs(["😊 Bình luận Tích cực (Độ tự tin cao lên đầu)", "😠 Bình luận Tiêu cực (Độ tự tin cao lên đầu)"])
    
    with tab1:
        positive_df = results_df[results_df["Sắc thái"] == "Tích cực"].copy()
        render_comment_table(positive_df, selected_sentiment="Tích cực")
    
    with tab2:
        negative_df = results_df[results_df["Sắc thái"] == "Tiêu cực"].copy()
        render_comment_table(negative_df, selected_sentiment="Tiêu cực")
    
    # Download button
    st.divider()
    export_df = results_df[["Bình luận gốc", "Sắc thái", "Độ tự tin", "prob_class_0"]].copy()
    csv_data = export_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "⬇️ Tải kết quả phân tích (CSV)",
        data=csv_data,
        file_name="brandhealth_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

