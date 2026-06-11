# 📄 TÀI LIỆU YÊU CẦU SẢN PHẨM (PRD)
**Dự án:** Hệ thống Giám sát Sức khỏe Thương hiệu từ Phản hồi Khách hàng ứng dụng Học sâu (BrandHealth AI Pipeline)  
**Tác giả:** Kỹ sư AI (Phan Anh)  
**Phiên bản:** 1.2  
**Ngày cập nhật:** Ngày 9 tháng 6 năm 2026  

---

## 1. Tổng quan dự án (Project Overview)

### 1.1. Bối cảnh & Vấn đề (Context & Problem Statement)
Trong kỷ nguyên số, doanh nghiệp tiếp nhận một lượng khổng lồ phản hồi phi cấu trúc của khách hàng từ nhiều nền tảng mạng xã hội và thương mại điện tử. Dữ liệu này chứa rất nhiều ngôn ngữ mạng phức tạp (teencode, từ lóng, mỉa mai). 

Việc rà soát thủ công gây quá tải nhân sự, trong khi ban quản trị cần một cái nhìn trực quan, tức thì về xu hướng dịch chuyển cảm xúc của khách hàng để đánh giá sức khỏe thương hiệu. Do đó, một hệ thống tự động hóa từ khâu tiền xử lý đến phân tích và trực quan hóa tương tác là vô cùng cấp thiết.

### 1.2. Giải pháp (Solution)
Xây dựng hệ thống tổng thể (End-to-End Pipeline) ứng dụng các mô hình Học sâu (Deep Learning) từ cơ bản đến nâng cao (BiLSTM, Attention, PhoBERT) trên tập dữ liệu tiếng Việt NTC-SCV. Hệ thống không chỉ phân loại sắc thái cảm xúc mà còn cung cấp một Web Dashboard tương tác cao, cho phép đào sâu (drill-down) vào chi tiết từng bình luận dựa trên các bộ lọc trực quan.

---

## 2. Mục tiêu chiến lược (Objectives & Key Results - OKRs)

*   **Mục tiêu học thuật:** Làm chủ và so sánh tường minh hiệu năng của 3 thế hệ kiến trúc NLP: Mạng tuần tự (BiLSTM), Mạng chú ý (BiLSTM + Attention), và Mô hình tiền huấn luyện hai chiều (PhoBERT).
*   **Mục tiêu sản phẩm (MVP):**
    *   **KR1 (Độ chính xác):** Mô hình tối tân (PhoBERT Fine-tuned) đạt $F1\text{-score} \ge 85\%$ trên tập kiểm thử.
    *   **KR2 (Trực quan hóa tương tác):** Thiết kế thành công Dashboard cho phép click chuột để xem chi tiết dữ liệu phân loại thay vì chỉ hiển thị biểu đồ tĩnh.

---

## 3. Đối tượng người dùng (User Personas)

### Persona 1: Người Quản lý Thương hiệu / Chủ Doanh nghiệp (End-User)
*   **Hành vi:** Không có kiến thức kỹ thuật về AI/Code. Thường xuyên bận rộn, chỉ tập trung vào các chỉ số cốt lõi và xu hướng dữ liệu để đưa ra quyết định vận hành hoặc truyền thông.
*   **Nhu cầu cốt lõi:** Cần biết nhanh tỷ lệ khách hàng đang phàn nàn (Tiêu cực) là bao nhiêu, và muốn lập tức đọc được tỉ lệ bình luận tích cực, tiêu cực và biết cụ thể những khách hàng đó đang chửi bới, không hài lòng về điều gì để xử lý.
---

## 4. Phạm vi sản phẩm (Product Scope)

### 4.1. MVP In-Scope (Các tính năng PHẢI CÓ trong phiên bản này)

*   **Module Lõi AI (Model Zoo):**
    *   Xây dựng luồng tiền xử lý (Làm sạch văn bản, chuẩn hóa tiếng Việt, xử lý từ lóng/teencode bằng từ điển mapping, tách từ ghép).
    *   Huấn luyện và đóng gói 3 mô hình: **BiLSTM**, **BiLSTM + Attention Mechanism**, và **PhoBERT Fine-tuned** để phân loại Sắc thái cảm xúc (Sentiment: Tích cực / Tiêu cực).

*   **Giao diện Tác vụ đơn lẻ (Single-text Inference):**
    *   Cung cấp giao diện cho phép nhập một câu văn bản bất kỳ để kiểm tra kết quả dự đoán Sắc thái cảm xúc theo thời gian thực.
    *   Tích hợp biểu đồ **Attention Heatmap** (đối với mô hình có cơ chế Attention) để trực quan hóa các từ khóa mà mô hình đang tập trung vào.

*   **Giao diện Tác vụ hàng loạt (Batch Inference & Interactive Dashboard):**
    *   Cho phép người dùng tải lên file CSV chứa danh sách các bình luận thô.
    *   Hệ thống tự động xử lý hàng loạt và xuất ra biểu đồ tổng quan: **Biểu đồ 2 cột** thể hiện tổng số lượng bình luận Tích cực và Tiêu cực.
    *   **Tính năng tương tác cốt lõi:** Khi người dùng click vào một cột bất kỳ trên biểu đồ (ví dụ: cột Tiêu cực), hệ thống sẽ lập tức hiển thị bảng danh sách chi tiết các bình luận thuộc nhóm sắc thái đó ở ngay bên dưới.

### 4.2. Out-of-Scope (Các tính năng TUYỆT ĐỐI KHÔNG LÀM trong phiên bản này)

*   **Hệ thống cào dữ liệu tự động (Crawl engine):** Không viết bot tự động cào real-time từ Facebook/Shopee để tránh bài toán chặn IP. Dữ liệu được giả lập qua file CSV tĩnh (NTC-SCV và dữ liệu tự chuẩn bị).
*   **Gán nhãn khía cạnh sâu (Aspect-Based Sentiment) và Highlight văn bản liên quan:** 
    *   *Không* bắt mô hình đọc và gán nhãn sắc thái cho từng khía cạnh nhỏ trong câu (Ví dụ: Đồ ăn, Dịch vụ, Giá cả).
    *   *Không* dựng biểu đồ cột ngang chứa số lượng ý kiến tiêu cực của từng khía cạnh một.
    *   *Không* làm tính năng tương tác click vào từng khía cạnh thì hiển thị ra tất cả các bình luận về khía cạnh đó và highlight (bôi đậm/tô màu) phần văn bản liên quan.
*   **Hệ thống phân quyền người dùng:** Có phân quyền Admin/Staff để quản lý.
*   **Nền tảng triển khai phức tạp:** Không làm ứng dụng di động (Mobile App). Chỉ tập trung vào giao diện Web Dashboard chạy trên Local hoặc Deploy Cloud dạng WebApp gọn nhẹ (sử dụng Streamlit hoặc Gradio).

---

## 5. User Stories & User Flow

### 5.1. User Stories (Câu chuyện người dùng)
*   **US-01 (Phân tích nhanh):** Là một người dùng, tôi muốn nhập một câu teencode bất kỳ vào hệ thống để kiểm tra xem mô hình AI có tiền xử lý đúng và nhận diện được sắc thái mỉa mai của câu đó hay không.
*   **US-02 (Giải thích mô hình):** Là một người dùng, tôi muốn nhìn thấy biểu đồ nhiệt Attention Heatmap sau khi test một câu để hiểu rõ cơ chế phân bổ trọng số từ khóa của mô hình mạng sâu.
*   **US-03 (Phân tích hàng loạt):** Là một Chủ doanh nghiệp, tôi muốn tải lên một file CSV chứa 1,000 đánh giá từ Shopee để hệ thống tự quét và phân loại hàng loạt thay vì phải đọc từng câu.
*   **US-04 (Đào sâu dữ liệu):** Là một Chủ doanh nghiệp, khi nhìn thấy cột "Tiêu cực" trên biểu đồ vọt lên cao, tôi muốn click thẳng vào cột đó để xem ngay lập tức danh sách văn bản của những bình luận chê bai đó nhằm tìm cách khắc phục kịp thời.

### 5.2. User Flow (Luồng trải nghiệm người dùng)
[Truy cập Web Dashboard]
│
├──► Giao diện 1: Thử nghiệm Đơn lẻ (Single Test)
│       └──► Nhập câu văn ──► Bấm Predict ──► Xem Sắc thái + Biểu đồ Attention Heatmap
│
└──► Giao diện 2: Phân tích Hàng loạt (Batch Process)
└──► Kéo thả file CSV ──► Hệ thống xử lý ──► Hiển thị Biểu đồ 2 Cột (Pos/Neg)
│
└──► Click vào Cột Tiêu cực (hoặc Tích cực)
│
└──► Hiển thị Bảng bình luận chi tiết tương ứng bên dưới

---

## 6. Thiết kế giao diện (UI Design Concept)

Giao diện được thiết kế theo cấu trúc một trang duy nhất (Single-page Dashboard) chia làm hai Tab chức năng chính để tối ưu hóa trải nghiệm trên Streamlit.

### 6.1. Tab 1: Phân tích Đơn lẻ (Real-time Playground)
*   **Vùng điều khiển (Top):** Một hộp chọn (Dropdown) để lựa chọn Mô hình suy luận (`BiLSTM`, `BiLSTM + Attention`, `PhoBERT Fine-tuned`). Bên dưới là một ô nhập liệu (`Text Input`) với placeholder: *"Nhập bình luận của khách hàng tại đây..."*. Bấm nút `Phân tích`.
*   **Vùng kết quả (Bottom):** 
    *   Bên trái: Hiển thị nhãn dự đoán bằng mác màu trực quan (`Tích cực` - Màu xanh / `Tiêu cực` - Màu đỏ) kèm độ tự tin % (Confidence Score).
    *   Bên phải: Nếu chọn mô hình Attention, hiển thị một ma trận biểu đồ nhiệt (Heatmap) tô màu các từ có trọng số cao trong câu (ví dụ: từ `tệ`, `chờ lâu` sẽ được tô đậm).

### 6.2. Tab 2: Giám sát Phản hồi Hàng loạt (Executive Dashboard)
*   **Bộ cấu hình đầu vào:** Thành phần kéo thả tệp (`File Uploader`) chấp nhận định dạng `.csv`.
*   **Khối trực quan hóa dữ liệu (Mục tiêu tương tác FR-3.3):**
    *   Hiển thị một biểu đồ thanh gồm 2 cột (Cột Xanh: Tích cực | Cột Đỏ: Tiêu cực). Biểu đồ được cấu hình ở dạng tương tác sinh động (`Interactive Plotly Chart`).
*   **Bảng dữ liệu chi tiết (Dynamic Data Table):** 
    *   Mặc định khi mới xử lý xong: Bảng trống hoặc hiển thị toàn bộ dữ liệu.
    *   Khi người dùng **Click vào Cột Đỏ (Tiêu cực)**: Bảng lập tức tải và hiển thị danh sách tất cả các hàng có nhãn là `Tiêu cực`, gồm 2 cột: `Bình luận gốc` và `Độ tự tin`.

---

## 7. Yêu cầu tính năng chi tiết (Functional Requirements)

### Epic 1: Mô-đun Tiền xử lý dữ liệu (ETL Pipeline)
*   **FR-1.1:** Hệ thống phải tự động loại bỏ ký tự đặc biệt vô nghĩa, xử lý teencode về từ gốc trước khi vector hóa văn bản.
*   **FR-1.2:** Hệ thống phải thực hiện tách từ ghép tiếng Việt (Word Segmentation) để đảm bảo ngữ nghĩa cho mô hình học sâu.

### Epic 2: Mô-đun Lõi AI (Deep Learning Model)
*   **FR-2.1:** Mô hình phải nhận đầu vào là chuỗi văn bản đã qua tiền xử lý và trả về phân phối xác suất của 2 lớp: Tích cực (1) và Tiêu cực (0).
*   **FR-2.2:** Mô hình BiLSTM + Attention phải xuất được ma trận trọng số Attention (Attention Weights) của câu phục vụ cho việc vẽ Heatmap.

### Epic 3: Giao diện Web Dashboard (Interactive UI Layer)
*   **FR-3.1:** Hệ thống phải cung cấp một Text Box để người dùng test nhanh ngẫu nhiên 1 câu, trả ra Label + Độ tự tin (Confidence score) + Biểu đồ nhiệt Attention Heatmap sau dưới 200ms.
*   **FR-3.2:** Hệ thống phải hỗ trợ kéo-thả file `.csv`. Sau khi xử lý xong, phải render ra biểu đồ 2 cột (Positive vs Negative).
*   **FR-3.3 (Drill-down action):** Hệ thống phải bắt được sự kiện Click (On-click event) của người dùng vào các cột của biểu đồ để thực hiện filter dữ liệu tương ứng trong bảng chi tiết dưới biểu đồ.

---

## 8. Yêu cầu phi chức năng (Non-Functional Requirements)

*   **Tính khả dụng (Usability):** Giao diện Web được phát triển bằng **Streamlit** (kết hợp thư viện `streamlit-plotly-events` để bắt sự kiện click trên biểu đồ) nhằm tối giản hóa code Frontend nhưng vẫn đảm bảo tính phản hồi tốt.
*   **Môi trường hệ thống (Environment):** Mã nguồn chạy ổn định trên hệ điều hành Ubuntu, đóng gói các thư viện phụ thuộc qua file `requirements.txt`.

---

## 9. Kế hoạch phát hành & Giai đoạn phát triển (Release Plan & Phases)

Dự án được chia làm 3 giai đoạn cuốn chiếu nghiêm ngặt để kiểm soát rủi ro kỹ thuật:

┌────────────────────────────────────────────────────────────────────────┐
│                        LỘ TRÌNH TRIỂN KHAI DỰ ÁN                       │
├────────────────────────────┬────────────────────────────┬──────────────┤
│ Giai đoạn & Thời gian      │ Công việc trọng tâm        │ Sản phẩm     │
├────────────────────────────┼────────────────────────────┼──────────────┤
│ Phase 1: Data & Baseline   │ Xử lý sạch tập NTC-SCV     │ Bản mẫu      │
│ (Tuần 1)                   │ Huấn luyện BiLSTM gốc      │ Terminal     │
├────────────────────────────┼────────────────────────────┼──────────────┤
│ Phase 2: Core AI Upgrading │ Thêm Attention, Fine-tune  │ File Save    │
│ (Tuần 2)                   │ PhoBERT, xuất ma trận trọng│ Model        │
├────────────────────────────┼────────────────────────────┼──────────────┤
│ Phase 3: Dashboard Integration│ Code giao diện Streamlit│ Hoàn chỉnh   │
│ (Tuần 3)                   │ kết nối mô hình & click xem│ Hệ thống     │
└────────────────────────────┴────────────────────────────┴──────────────┘
---

## 10. Chỉ số đo lường mức độ thành công (Success Metrics)

┌────────────────────────────────────────────────────────────────────────┐
│                        TIÊU CHUẨN ĐÁNH GIÁ MVP                         │
├─────────────────────┬──────────────────────┬───────────────────────────┤
│      Kiến trúc      │  Accuracy / F1-Score │ Trực quan hóa Giao diện   │
├─────────────────────┼──────────────────────┼───────────────────────────┤
│ BiLSTM (Baseline)   │      ~ 70 - 75%      │ Biểu đồ 2 cột + Click xem │
│ BiLSTM + Attention  │      ~ 75 - 80%      │ Bổ sung Attention Heatmap │
│ PhoBERT Fine-tuned  │      ~ 85 - 90%      │ Biểu đồ 2 cột + Click xem │
└─────────────────────┴──────────────────────┴───────────────────────────┘