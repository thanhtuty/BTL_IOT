# Nguồn mã và kiến trúc

Nguồn duy nhất cho kiến trúc và tiền xử lý trong dự án này: file người dùng gửi `palmbridge.ipynb`, **cell 2** — `PALMBRIDGE V1 - FEW-SHOT X-PALM FINE-TUNING`.

| Trong dự án | Nguồn / dòng (file xuất riêng theo cell) | Nội dung |
|---|---|---|
| `preprocessing.py` | `NormSingleROI`, dòng 544–587 | Lớp chuẩn hóa y nguyên từ notebook |
| `roi.py` | Viết mới | Dùng MediaPipe Hand Landmarker để căn/cắt ROI từ ảnh toàn bàn tay; đây là bước tiền xử lý cho ảnh camera, không phải thành phần của checkpoint PalmBridge |
| `models/backbone.py` | dòng 693–1248 | `LearnableGaborLayer`, `CompetitivePool`, `CustomCompNetBackbone`, `OfficialGaborConv2d`, `OfficialCompetitiveBlock`, `OfficialCompNetBackbone` giữ đúng cấu trúc/tên lớp |
| `models/palmbridge.py` | `PalmBridge`, dòng 1256–1301 | Codebook và blending y nguyên; thêm wrapper suy luận theo `PalmBridgeFineTuneModel.get_embedding()` dòng 1402–1454 |
| `models/checkpoint.py` | Từ cách nạp checkpoint dòng 1461–1733, cách lưu dòng 2750–2820 | Viết lại loader gọn hơn, chỉ dành cho inference và `weights_only=True` |
| `enroll.py`, `verify.py`, `database.py`, `evaluate.py`, `main.py` | Viết mới cho dự án IoT của nhóm | Đăng ký, crop ROI tùy chọn và so sánh embedding (không có nguyên xi trong notebook) |

**Không copy:** training loop, optimizer, classifier ArcMargin và code tạo đồ thị vì không cần cho inference. Nếu về sau fine-tune, quay lại notebook gốc. `Official...`/`Custom...` chỉ chọn một biến thể tương ứng checkpoint.
