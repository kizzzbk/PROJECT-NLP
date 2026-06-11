"""
Tab 2: Executive Dashboard — Batch CSV Analysis
=================================================
PRD Section 6.2
Interactive bar chart with click-to-drill-down functionality (FR-3.3).
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from app.components.model_selector import render_model_selector
from app.components.sentiment_chart import render_sentiment_chart
from app.components.comment_table import render_comment_table


def render_batch_analysis():
    """Render the batch analysis page."""
    
    st.markdown(
        """
        <h1 style="text-align: center; margin-bottom: 0;">
            📊 Giám sát Phản hồi Hàng loạt
        </h1>
        <p style="text-align: center; color: #94a3b8; margin-top: 0.3rem; margin-bottom: 2rem;">
            Tải file CSV → Phân loại tự động → Click vào biểu đồ để xem chi tiết
        </p>
        """,
        unsafe_allow_html=True,
    )
    
    # Model selector
    model_name, predictor = render_model_selector(key_prefix="batch")
    
    st.divider()
    
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
    """Process all texts in the dataframe."""
    texts = df[text_column].astype(str).tolist()
    total = len(texts)
    
    progress_bar = st.progress(0, text="🔄 Đang phân tích...")
    results = []
    
    for i, text in enumerate(texts):
        result = predictor.predict_single(text)
        results.append({
            "Bình luận gốc": text,
            "Bình luận đã xử lý": result["cleaned_text"],
            "Sắc thái": result["label"],
            "Độ tự tin": result["confidence"],
        })
        
        if (i + 1) % max(1, total // 100) == 0:
            progress_bar.progress(
                (i + 1) / total,
                text=f"🔄 Đang phân tích... ({i + 1}/{total})",
            )
    
    progress_bar.progress(1.0, text="✅ Hoàn thành!")
    
    results_df = pd.DataFrame(results)
    st.session_state.batch_results = results_df
    st.rerun()


def _display_results():
    """Display batch analysis results with interactive chart."""
    results_df = st.session_state.batch_results
    
    # Summary metrics
    total = len(results_df)
    positive_count = (results_df["Sắc thái"] == "Tích cực").sum()
    negative_count = (results_df["Sắc thái"] == "Tiêu cực").sum()
    avg_confidence = results_df["Độ tự tin"].mean()
    
    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📝 Tổng bình luận", f"{total:,}")
    with col2:
        st.metric("😊 Tích cực", f"{positive_count:,}", f"{positive_count/total:.1%}")
    with col3:
        st.metric("😠 Tiêu cực", f"{negative_count:,}", f"{negative_count/total:.1%}")
    with col4:
        st.metric("📊 Độ tự tin TB", f"{avg_confidence:.1%}")
    
    st.divider()
    
    # Interactive chart
    col_chart, col_table = st.columns([1, 1])
    
    with col_chart:
        st.markdown("### 📊 Biểu đồ Tổng quan Sắc thái")
        selected_sentiment = render_sentiment_chart(positive_count, negative_count)
    
    with col_table:
        st.markdown("### 📋 Danh sách Bình luận Chi tiết")
        
        if selected_sentiment:
            st.info(f"🔍 Đang hiển thị: **{selected_sentiment}**")
            filtered = results_df[results_df["Sắc thái"] == selected_sentiment]
        else:
            st.info("💡 **Click vào cột trên biểu đồ** để lọc bình luận theo sắc thái")
            filtered = results_df
        
        render_comment_table(filtered)
    
    # Download button
    st.divider()
    csv_data = results_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "⬇️ Tải kết quả (CSV)",
        data=csv_data,
        file_name="brandhealth_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
