# DATASET_V2 — ẢNH ROI LÒNG BÀN TAY

Đây là nơi đặt **ảnh đã cắt vùng lòng bàn tay** để huấn luyện V2. Mỗi thư mục person_01..04 tương ứng đúng một thành viên, không đổi ID giữa các lần train.

- session_01/: 150–200 ảnh mỗi người, chụp các buổi để train (tối thiểu 5)
- session_02/: 25–50 ảnh mỗi người, dùng validation và hiệu chỉnh ngưỡng (tối thiểu 2)
- session_03/: 25–50 ảnh mới mỗi người, test cuối cùng
- unknown/session_02, unknown/session_03: ảnh người ngoài nhóm, không trộn với thành viên. Nên có ít nhất 100 ảnh unknown ở mỗi tập, từ nhiều bàn tay.

Nếu chụp ảnh **nguyên bàn tay**, thay vì ROI, bỏ ảnh vào thư mục tương ứng dưới `photos_raw_v2/`, sau đó dùng `python -m palm_v2.cli prepare-roi` (cần MediaPipe trên máy tính). Không lưu một ảnh vào cả train và test.

**Không tải ảnh sinh trắc lên GitHub công khai.** File .gitkeep ở các thư mục là placeholder. Ảnh được bổ sung vào thư mục local trên máy khi clone branch này. Xem `palm_v2/README.md` để biết lệnh huấn luyện.
