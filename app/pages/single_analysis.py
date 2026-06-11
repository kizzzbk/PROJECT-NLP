"""
Tab 1: Real-time Playground — Single Text Analysis
====================================================
PRD Section 6.1
"""

import streamlit as st

from app.components.model_selector import render_model_selector
from app.components.attention_heatmap import render_attention_heatmap


def render_single_analysis():
    """Render the single text analysis page."""
    
    st.markdown(
        """
        <h1 style="text-align: center; margin-bottom: 0;">
            🔬 Phân tích Đơn lẻ
        </h1>
        <p style="text-align: center; color: #94a3b8; margin-top: 0.3rem; margin-bottom: 2rem;">
            Nhập câu văn bản để phân tích sắc thái cảm xúc theo thời gian thực
        </p>
        """,
        unsafe_allow_html=True,
    )
    
    # Model selector
    model_name, predictor = render_model_selector()
    
    st.divider()
    
    # Text input
    col_input, col_btn = st.columns([5, 1])
    
    with col_input:
        user_text = st.text_area(
            "✍️ Nhập bình luận của khách hàng",
            placeholder="Ví dụ: sản phẩm quá tệ, ship chậm lắm, đóng gói sơ sài...",
            height=100,
            key="single_text_input",
        )
    
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button(
            "🚀 Phân tích",
            use_container_width=True,
            type="primary",
        )
    
    # Sample texts for quick testing
    with st.expander("💡 Câu mẫu để thử nghiệm", expanded=False):
        sample_cols = st.columns(2)
        
        positive_samples = [
            "Sản phẩm rất tốt, mình rất hài lòng",
            "Giao hàng nhanh, đóng gói cẩn thận",
            "Hàng xịn xò, giá tốt, sẽ ủng hộ tiếp",
        ]
        negative_samples = [
            "Sp quá tệ, ship chậm, k bao giờ mua nữa",
            "Hàng fake, ko giống hình, thất vọng",
            "Dịch vụ tệ, nv hỗ trợ chậm, đắt mà dở",
        ]
        
        with sample_cols[0]:
            st.markdown("**✅ Tích cực:**")
            for s in positive_samples:
                if st.button(s, key=f"pos_{s[:10]}"):
                    st.session_state.single_text_input = s
                    st.rerun()
        
        with sample_cols[1]:
            st.markdown("**❌ Tiêu cực:**")
            for s in negative_samples:
                if st.button(s, key=f"neg_{s[:10]}"):
                    st.session_state.single_text_input = s
                    st.rerun()
    
    # Analysis results
    if analyze_btn and user_text and predictor is not None:
        with st.spinner("🔄 Đang phân tích..."):
            result = predictor.predict_single(user_text)
        
        st.divider()
        
        # Results layout
        col_result, col_attn = st.columns([1, 1])
        
        with col_result:
            st.markdown("### 🏷️ Kết quả Dự đoán")
            
            # Label with color
            label = result["label"]
            confidence = result["confidence"]
            
            if result["label_id"] == 1:
                label_color = "#10b981"
                label_emoji = "😊"
                label_bg = "rgba(16, 185, 129, 0.1)"
            elif result["label_id"] == 0:
                label_color = "#ef4444"
                label_emoji = "😠"
                label_bg = "rgba(239, 68, 68, 0.1)"
            else:
                label_color = "#6b7280"
                label_emoji = "❓"
                label_bg = "rgba(107, 114, 128, 0.1)"
            
            st.markdown(
                f"""
                <div style="background: {label_bg}; border: 2px solid {label_color}; 
                            border-radius: 12px; padding: 1.5rem; text-align: center;">
                    <span style="font-size: 3rem;">{label_emoji}</span>
                    <h2 style="color: {label_color}; margin: 0.5rem 0 0.2rem;">{label}</h2>
                    <p style="font-size: 1.5rem; font-weight: bold; margin: 0;">
                        {confidence:.1%}
                    </p>
                    <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">Độ tự tin</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # Probability distribution
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**📊 Phân phối xác suất:**")
            
            for lbl, prob in result["probabilities"].items():
                color = "#10b981" if "Tích" in lbl else "#ef4444"
                st.markdown(
                    f"""
                    <div style="margin: 0.3rem 0;">
                        <span>{lbl}</span>
                        <div style="background: #1e293b; border-radius: 8px; height: 24px; overflow: hidden;">
                            <div style="background: {color}; width: {prob*100}%; height: 100%; 
                                        border-radius: 8px; display: flex; align-items: center; 
                                        padding-left: 8px; font-size: 0.8rem; font-weight: bold;">
                                {prob:.2%}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            # Preprocessing details
            with st.expander("🧹 Chi tiết tiền xử lý"):
                st.text(f"Gốc:  {result['original_text']}")
                st.text(f"Sạch: {result['cleaned_text']}")
        
        with col_attn:
            st.markdown("### 🔥 Attention Heatmap")
            
            if result.get("attention_weights") and result.get("tokens"):
                render_attention_heatmap(
                    tokens=result["tokens"],
                    weights=result["attention_weights"],
                    title=f"Trọng số Attention — {model_name}",
                )
            else:
                st.info(
                    "ℹ️ Attention Heatmap chỉ khả dụng với mô hình "
                    "**BiLSTM + Attention** hoặc **PhoBERT Fine-tuned**. "
                    "Hãy chọn mô hình phù hợp ở trên."
                )
    
    elif analyze_btn and not user_text:
        st.warning("⚠️ Vui lòng nhập văn bản để phân tích.")
    
    elif analyze_btn and predictor is None:
        st.error("❌ Chưa có mô hình nào được huấn luyện. Vui lòng train mô hình trước.")
