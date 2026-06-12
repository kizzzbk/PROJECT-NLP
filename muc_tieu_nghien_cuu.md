# PHẦN 2: MỤC TIÊU NGHIÊN CỨU

## 2.1. Đặt vấn đề và Định hướng Nghiên cứu
Trong xu thế chuyển đổi số, dữ liệu phản hồi phi cấu trúc từ khách hàng trên mạng xã hội và các trang thương mại điện tử ngày càng bùng nổ. Nguồn dữ liệu này chứa đựng những thông tin giá trị giúp doanh nghiệp đánh giá sức khỏe thương hiệu một cách khách quan. Tuy nhiên, rào cản lớn nhất trong việc khai thác dữ liệu này là tính phức tạp của ngôn ngữ mạng tiếng Việt, nơi mà các hiện tượng teencode, từ lóng, viết tắt hay các sắc thái mỉa mai xuất hiện phổ biến. Từ thực trạng đó, nghiên cứu này hướng đến việc thiết lập một quy trình phân tích cảm xúc khép kín từ khâu tiền xử lý dữ liệu thô đến việc áp dụng các kiến trúc học sâu nhằm nâng cao độ chính xác và tính thực tiễn của hệ thống giám sát.

## 2.2. Tiến trình từ Tiền xử lý đến các Kiến trúc Học sâu
Nghiên cứu được thiết kế theo một lộ trình kỹ thuật chặt chẽ, đi từ khâu xử lý dữ liệu nền tảng đến việc thử nghiệm và tối ưu hóa các mô hình phân loại sắc thái cảm xúc.

Đầu tiên, quy trình tiền xử lý được xây dựng để giải quyết triệt để tính nhiễu của văn bản tiếng Việt. Dữ liệu thô sau khi thu thập sẽ đi qua luồng lọc bỏ các ký tự đặc biệt không mang thông tin ngữ nghĩa, chuẩn hóa định dạng chữ viết, và sử dụng bộ từ điển ánh xạ để đưa teencode cùng từ lóng về từ gốc tương đương. Tiếp theo, bước tách từ ghép tiếng Việt được áp dụng nhằm đảm bảo các cụm từ mang ý nghĩa thống nhất không bị chia cắt sai lệch khi chuyển đổi thành các véc-tơ đặc trưng đầu vào cho mô hình học sâu.

Sau khi dữ liệu được làm sạch và chuẩn hóa, nghiên cứu triển khai hệ thống huấn luyện và đánh giá trên ba thế hệ mô hình học sâu có cấu trúc và độ phức tạp tăng dần:
- Kiến trúc mạng tuần tự hai chiều BiLSTM được sử dụng làm mô hình cơ sở để khai thác thông tin ngữ cảnh từ cả chiều xuôi và chiều ngược của chuỗi văn bản.
- Mô hình BiLSTM kết hợp cơ chế chú ý Attention được phát triển nhằm tăng cường khả năng tập trung vào các từ khóa mang tính quyết định cảm xúc trong câu, đồng thời cho phép trích xuất trọng số phục vụ trực quan hóa bản đồ nhiệt.
- Mô hình ngôn ngữ tiền huấn luyện quy mô lớn PhoBERT được tinh chỉnh sâu trên tập dữ liệu tiếng Việt nhằm tận dụng tối đa tri thức ngôn ngữ đã được học trước đó, hướng tới mục tiêu tối ưu hóa độ chính xác và đạt chỉ số đo lường hiệu năng cao nhất.

## 2.3. Mục tiêu Nghiên cứu Cụ thể
Với định hướng kết hợp giữa lý thuyết học sâu và ứng dụng thực tiễn, nghiên cứu xác định các mục tiêu cốt lõi sau:

Thứ nhất, kiểm chứng và đánh giá định lượng vai trò của khâu tiền xử lý đối với hiệu năng huấn luyện mô hình. Nghiên cứu cần làm rõ mức độ đóng góp của việc chuẩn hóa teencode và tách từ ghép tiếng Việt trong việc cải thiện tốc độ hội tụ cũng như độ chính xác của các mô hình học sâu.

Thứ hai, thực hiện so sánh đối chứng toàn diện hiệu năng phân loại cảm xúc giữa mạng tuần tự BiLSTM, mô hình kết hợp Attention và mô hình tiền huấn luyện PhoBERT. Qua đó, nghiên cứu sẽ phân tích chi tiết sự đánh đổi giữa thời gian huấn luyện, tài nguyên tính toán cần thiết và chất lượng dự báo của từng thế hệ kiến trúc.

Thứ ba, xây dựng hệ thống ứng dụng thực tiễn có khả năng tương tác trực quan cao. Mục tiêu là phát triển giao diện giám sát cho phép người dùng click trực tiếp vào các phần biểu đồ thống kê để xem ngay lập tức danh sách các bình luận chi tiết tương ứng bên dưới, đáp ứng nhu cầu đào sâu dữ liệu nhanh chóng của doanh nghiệp.
