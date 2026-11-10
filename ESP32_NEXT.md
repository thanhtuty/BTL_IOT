# Phần chưa có trong ZIP: chạy AI trực tiếp trên ESP32-S3

Tài liệu và code ở đây chỉ để xác minh pipeline trên máy tính bằng PyTorch.

ESP32-S3 CAM 4MB Flash, 2MB PSRAM **không đủ để chạy trực tiếp PalmBridge gốc**.
Cần làm thêm sau khi baseline trên máy tính đã được đánh giá:

1. Xác định camera, thư viện driver và chất lượng ảnh ROI tại thiết bị.
2. Xây quy trình chụp cố định bằng khung định vị bàn tay hoặc thuật toán lấy ROI tiết kiệm RAM.
3. Chọn mô hình rất nhỏ/thuật toán đặc trưng truyền thống, đo kích thước model và peak RAM.
4. Nếu dùng ESP-DL, thiết kế với các toán tử được hỗ trợ, lượng tử hóa và chuyển `.espdl`.
5. Xây giao diện C/C++ camera -> crop -> normalize -> inference -> cosine -> relay/servo.
6. Đo FAR/FRR/tốc độ ngay trên ảnh do ESP32 chụp; thêm cơ chế chống giả mạo để dùng như khóa thật.

**Không phải** chỉ đổi `.pt` thành `.espdl` là xong. Sẽ cần mô hình tương thích riêng.
