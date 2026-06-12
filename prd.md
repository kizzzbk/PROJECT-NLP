# 📄 TÀI LIỆU YÊU CẦU SẢN PHẨM (PRD)
**Dự án:** ỨNG DỤNG CÁC MÔ HÌNH MẠNG NƠ-RON HỌC SÂU TRONG CẢNH BÁO RỦI RO ĐÁNH GIÁ XẤU 
TRÊN NỀN TẢNG GIAO ĐỒ ĂN 
**Tác giả:** Kỹ sư AI (Phan Anh)  
**Phiên bản:** 2.0  
**Ngày cập nhật:** Ngày 12 tháng 6 năm 2026  

---

## 1. Tổng quan dự án (Project Overview)

### 1.1. Bối cảnh & Nỗi đau thực tế
Các cửa hàng F&B vừa và nhỏ (SMEs/Local Brands) hoạt động trên các ứng dụng giao đồ ăn (GrabFood, ShopeeFood) sống dựa hoàn toàn vào điểm sao hiển thị ($Rating$). Thuật toán của App sẽ tự động phạt (gỡ quyền tham gia khuyến mãi, tụt vị trí tìm kiếm) nếu điểm của quán bị kéo thấp bởi các đánh giá tiêu cực $1^{\star}$. Tuy nhiên, chủ quán nhỏ thường bị quá tải do không thể đọc thủ công hàng ngàn bình luận mạng phi cấu trúc chứa đầy teencode, từ lóng và sự mỉa mai, dẫn đến việc bỏ lỡ "giờ vàng" (trong vòng 24 giờ) để nhắn tin đền bù và thuyết phục khách hàng chỉnh sửa lại số sao trên App.

### 1.2. Giải pháp 
Xây dựng một ứng dụng Dashboard thông minh dựa trên kỹ nghệ học sâu (Deep Learning). Hệ thống tự động đồng hóa văn bản mạng phức tạp, phân loại sắc thái nhị phân trực quan và cho phép chủ quán lập tức "đào sâu" vào các bình luận tiêu cực nghiêm trọng nhất. Đặc biệt, hệ thống ứng dụng chính trọng số nội bộ của mô hình AI để làm nổi bật (highlight) các từ khóa gây lỗi vận hành, giúp chủ quán hiểu lý do khách chửi bới chỉ trong 3 giây để đưa ra kịch bản đền bù khẩn cấp.

---

## 2. Mục tiêu chiến lược (Objectives & Key Results)

* **Mục tiêu học thuật:** Làm chủ, thực nghiệm và so sánh tường minh hiệu năng phân loại của 3 thế hệ kiến trúc NLP từ kinh điển đến tối tân: Mạng tuần tự hai chiều (**BiLSTM**), Mạng tích hợp trọng số tập trung (**BiLSTM + Attention Mechanism**), và Mô hình ngôn ngữ lớn tiền huấn luyện ngữ cảnh (**PhoBERT Fine-tuned**).
* **Mục tiêu sản phẩm (MVP):**
    * **KR1:** Thiết kế thành công biểu đồ tròn tương tác hiển thị tỷ lệ % sắc thái cảm xúc.
    * **KR2:** Hiện thực hóa tính năng "Drill-down" tự động lọc và sắp xếp bình luận theo mức độ nghiêm trọng (tiệm cận hai đầu xác suất $0.0$ và $1.0$).
    * **KR3:** Xây dựng bộ giải mã trọng số (Attention Weights/Prediction Probability) để bôi đậm văn bản dạng trực quan hóa Heatmap theo mức độ quan trọng của từ ngữ ngay trên bảng chi tiết.

---

## 3. Đối tượng người dùng (User Personas)

### 3.1. Chủ cửa hàng F&B nhỏ / Quản lý vận hành (End-User)
* **Hành vi:** Không có kiến thức về kỹ thuật AI/Code. Thời gian eo hẹp do phải kiêm nhiệm nhiều đầu việc tại cửa hàng. Chỉ truy cập hệ thống vào cuối ngày hoặc cuối tuần bằng cách xuất file báo cáo từ App giao đồ ăn ra và ném vào Dashboard.
* **Nhu cầu cốt lõi:** Muốn biết ngay tỷ lệ phần trăm thiệt hại (bao nhiêu % khách chê). Khi click vào nhóm chê, họ cần đọc ngay những câu có mức độ tức giận cao nhất (tiêu cực nhất) và muốn nhìn thấy ngay từ khóa cốt lõi bị chê (ví dụ: **dở**, **chờ lâu**, **thối**) được làm nổi bật để lập tức copy mẫu câu xin lỗi và gửi đền bù cho khách, cứu vãn điểm sao hiển thị.
---

## 4. Phạm vi sản phẩm (Product Scope)

### 4.1. MVP In-Scope (Các tính năng PHẢI CÓ)
* **Module Lõi AI (Model Zoo):**
    * Luồng tiền xử lý: Xóa ký tự đặc biệt, chuẩn hóa Unicode dấu tiếng Việt, mapping từ điển teencode/từ lóng về từ gốc, tách từ ghép (`underthesea`).
    * Đóng gói 3 mô hình phân loại nhị phân (Tích cực / Tiêu cực): BiLSTM, BiLSTM+Attention, PhoBERT.
* **Giao diện Tác vụ hàng loạt (Batch Inference & Interactive Dashboard):**
    * Thành phần kéo thả tệp CSV chứa danh sách bình luận thô.
    * Xuất biểu đồ tròn (`Pie Chart`) thể hiện tỷ lệ % số lượng bình luận Tích cực (Xanh) và Tiêu cực (Đỏ).
    * **Tính năng tương tác cốt lõi:** Khi click chuột vào miếng bánh "Tiêu cực" hoặc "Tích cực" trên biểu đồ tròn, bảng dữ liệu bên dưới lập tức filter chỉ hiển thị nhóm đó.
    * **Thuật toán sắp xếp ưu tiên:** Dữ liệu trong bảng phải được sắp xếp theo mức độ phân cực rõ ràng nhất (Hiển thị các câu có xác suất tiến gần về 2 đầu $0.0$ - cực kỳ tiêu cực và $1.0$ - cực kỳ tích cực lên trên cùng).
    * **Trực quan hóa trọng số (Heatmap in Table):** Tại bảng chi tiết, văn bản bình luận sẽ được bôi đậm tự động dưới dạng Heatmap. Các từ có trọng số quyết định nhãn (từ tốt, xuất sắc ở câu tích cực; từ dở, tệ, lâu ở câu tiêu cực) phải in đậm rõ nét hơn các từ còn lại dựa trên Attention Weight hoặc hệ số tương quan của PhoBERT.

### 4.2. Out-of-Scope (Các tính năng TUYỆT ĐỐI KHÔNG LÀM)

* **Hệ thống phân quyền người dùng:** Không Tích hợp màn hình Login phân quyền tài khoản Admin (Quản lý cấp cao, xem toàn bộ, xóa dữ liệu) và Staff (Nhân viên, chỉ upload file và tương tác phản hồi).
* *Không* viết bot cào dữ liệu tự động (Crawl Engine) thời gian thực từ Facebook/Shopee để tránh chặn IP. Dữ liệu nạp bằng file CSV tĩnh.
* *Không* bắt mô hình làm tác vụ gán nhãn khía cạnh sâu (ABSA) (Không phân tách câu thành Đồ ăn, Dịch vụ, Giá cả).
* *Không* dựng biểu đồ cột ngang chứa số lượng ý kiến tiêu cực theo từng khía cạnh.
* *Không* làm tính năng tương tác click vào từng khía cạnh nhỏ và bôi đậm văn bản theo khía cạnh đó.
* *Không* phát triển ứng dụng di động (Mobile App), hệ thống chỉ chạy duy nhất giao diện Web Dashboard bằng **Streamlit**.

---

## 5. User Stories & User Flow

### 5.1. User Stories
* **US-01:** Là một Chủ quán, tôi muốn đăng nhập vào hệ thống để bảo mật dữ liệu kinh doanh của cửa hàng mình.
* **US-02:** Là một Chủ quán, tôi muốn kéo thả file CSV chứa 1,000 bình luận tuần qua của quán lên hệ thống để AI tự động xử lý hàng loạt.
* **US-03:** Là một Chủ quán, tôi muốn nhìn thấy một biểu đồ tròn trực quan để biết ngay tỷ lệ phần trăm khách hàng đang tức giận (Tiêu cực) trong tuần qua chiếm bao nhiêu %.
* **US-04:** Là một Chủ quán, khi tôi click vào miếng bánh Tiêu cực trên biểu đồ tròn, tôi muốn bảng bên dưới chỉ hiện các câu chê và phải đẩy các câu chê nặng nhất (xác suất gần lớp 0 nhất) lên đầu để tôi giải quyết trước.
* **US-05:** Là một Chủ quán, tôi muốn các từ khóa chí mạng khiến khách tức giận (như *bẩn*, *tanh*, *giao lâu*) phải được bôi đậm rực lên dạng Heatmap để tôi biết ngay lỗi vận hành nằm ở đâu mà không cần đọc hết cả câu dài.

### 5.2. User Flow (Luồng trải nghiệm người dùng)

[Màn hình Đăng nhập] ──► Kiểm tra quyền (Admin / Staff) ──► [Vào Dashboard chính]
│
[Bảng dữ liệu chi tiết dưới dạng Heatmap] ◄── [Click miếng bánh] ◄── [Hiển thị Biểu đồ Tròn % Pos/Neg] ◄── [Kéo thả file CSV]
(Sắp xếp ưu tiên theo 2 đầu xác suất)

---

## 6. Thiết kế giao diện (UI Design Concept)

Giao diện WebApp được thiết kế tối giản, tập trung vào luồng xử lý hàng loạt trên một trang duy nhất (Single-page Dashboard).

### 6.1. Thanh điều hướng bên (Sidebar)
* Hiển thị thông tin người dùng đang đăng nhập (Mác quyền: `Admin` hoặc `Staff`).
* Hộp chọn (`Dropdown`) cấu hình thuật toán mô hình lõi sử dụng để quét dữ liệu: `BiLSTM`, `BiLSTM + Attention`, hoặc `PhoBERT Fine-tuned`.
* Nút Đăng xuất (`Logout`).

### 6.2. Vùng Trung tâm: Giám sát Phản hồi Hàng loạt (Executive Dashboard)
* **Khối nạp dữ liệu (Top):** Thành phần `st.file_uploader` dạng kéo thả file `.csv`. Sau khi xử lý xong sẽ xuất ra thông báo: *"Đã phân tích thành công X dòng bình luận"*.
* **Khối trực quan hóa (Middle):** Render một biểu đồ tròn tương tác hiển thị tỷ lệ phần trăm (Ví dụ: 75% Màu xanh Tích cực | 25% Màu đỏ Tiêu cực).
* **Khối hiển thị chi tiết (Bottom - Dynamic Data Table):** * Mặc định: Bảng trống.
    * Khi người dùng **Click vào vùng màu Đỏ (Tiêu cực)** trên biểu đồ tròn: Bảng lập tức xuất hiện danh sách các câu bị đoán là lớp 0, câu có xác suất gần $0.0$ nhất nằm trên cùng. Cột "Bình luận" sử dụng định dạng HTML để hiển thị văn bản dạng Heatmap (Cụm từ quan trọng được bao bọc bởi thẻ màu đậm nền vàng/đỏ tăng dần theo trọng số).

---

## 7. Yêu cầu tính năng chi tiết (Functional Requirements)

### Epic 1: Mô-đun Tiền xử lý dữ liệu (ETL Pipeline)
* **FR-1.1:** Hệ thống phải tự động làm sạch văn bản, chuyển đổi teencode về từ gốc thông qua từ điển mapping được định nghĩa trước.
* **FR-1.2:** Hệ thống phải thực hiện tách từ ghép tiếng Việt để bảo toàn cấu trúc ngữ nghĩa cho mô hình học sâu.

### Epic 2: Mô-đun Lõi AI (Deep Learning Model)
* **FR-2.1:** Các mô hình phải trả về một mảng số thực đại diện cho phân phối xác suất (Prediction Probability) của 2 lớp $[0, 1]$.
* **FR-2.2:** Mô hình phải xuất được ma trận trọng số (Attention Weights hoặc hệ số Score của Layer cuối cùng) để chuyển giao sang mô-đun UI làm tham số vẽ độ đậm nhạt cho Heatmap của từng từ.

### Epic 3: Giao diện Web Dashboard (Interactive UI Layer)
* **FR-3.1:** Hệ thống phải bảo mật bằng màn hình Login, kiểm tra tài khoản từ Mock DB phân biệt rõ hai nhóm quyền Admin và Staff.
* **FR-3.2:** Hệ thống phải hỗ trợ render biểu đồ tròn tương tác, bắt được sự kiện On-click vào từng phân vùng cảm xúc để kích hoạt bộ lọc cho bảng dữ liệu.
* **FR-3.3:** Bảng dữ liệu phải thực hiện hai tác vụ động: Một là sắp xếp toán học theo khoảng cách tiến về hai đầu cực trị xác suất ($0.0$ và $1.0$), hai là tự động biên dịch trọng số AI thành mã HTML `<b>` hoặc `<span>` màu để hiển thị chữ bôi đậm dạng Heatmap theo thời gian thực.

---

## 8. Yêu cầu phi chức năng (Non-Functional Requirements)

* **Tính khả dụng (Usability):** Giao diện Web được phát triển bằng **Streamlit** (kết hợp thư viện mở rộng như `streamlit-plotly-events` hoặc thành phần HTML tùy biến để bắt mượt mà sự kiện click trên biểu đồ tròn).
* **Môi trường hệ thống (Environment):** Mã nguồn chạy ổn định trên Ubuntu, quản lý thư viện tập trung bằng file `requirements.txt`.

---

## 9. Kế hoạch phát hành & Giai đoạn phát triển (Release Plan & Phases)

| Giai đoạn & Thời gian | Công việc trọng tâm | Sản phẩm đầu ra |
| :--- | :--- | :--- |
| **Phase 1: Data & Security** (Tuần 1) | Sạch hóa dữ liệu NTC-SCV; Code Module Login và Phân quyền tài khoản. | Hệ thống có màn hình xác thực Authentication thô. |
| **Phase 2: Core Model Zoo** (Tuần 2) | Huấn luyện và tối ưu 3 mô hình BiLSTM, Attention, PhoBERT; Trích xuất ma trận trọng số khả vi. | Các file lưu checkpoint trọng số mô hình dạng `.pt` hoặc `.bin`. |
| **Phase 3: Interactive UI** (Tuần 3) | Code Dashboard Streamlit, vẽ cấu trúc biểu đồ tròn tương tác, render bảng văn bản Heatmap HTML. | Hệ thống sản phẩm AI hoàn chỉnh (End-to-End System). |

---

## 10. Chỉ số đo lường mức độ thành công (Success Metrics)

* **Chỉ số mô hình (Model Metrics):** Độ chính xác toàn cục ($Accuracy$) và chỉ số dung hòa sai lầm ($F1\text{-score}$) của mô hình PhoBERT Fine-tuned bắt buộc phải đạt $\ge 85\%$ trên tập dữ liệu kiểm thử độc lập.
* **Chỉ số trải nghiệm (UI Metrics):** Thời gian từ lúc người dùng click vào miếng bánh biểu đồ tròn đến khi bảng chi tiết bên dưới lọc xong dữ liệu và render thành công chữ bôi đậm Heatmap không được vượt quá **500ms**.