# PalmLock V2 — 4 người, Gabor nhẹ + Depthwise CNN + Cosine

Đây là **nhánh thử nghiệm V2**, độc lập hoàn toàn với mô hình PalmBridge 128×128 nguyên bản ở thư mục gốc. Giữ nguyên các file V1 để đối chiếu. Mục tiêu: **huấn luyện trên PC, suy luận 64×64 trực tiếp trên ESP32 thường có PSRAM**, sau khi đã có dữ liệu và benchmark.

## 1. Ảnh cần chụp

Khuyến nghị **200–300 ảnh/người**, 4 người ~800–1.200 ảnh. Không chụp liên tiếp 200 khung gần như trùng nhau. Thu ảnh 3 buổi khác nhau, có thay đổi ánh sáng/góc nhẹ, cùng một tay (nên dùng cùng tay trong enroll/test). Giữ lòng bàn tay ở trung tâm, chụp rõ đường vân, tránh rung/nhòe và các vật che khuất.

**Quan trọng:** V2 sử dụng **ROI lòng bàn tay hình vuông** (không phải ảnh toàn thân hoặc ảnh bàn tay có nền rất rộng). Dùng vùng lòng bàn tay dưới các ngón và trên cổ tay; nếu chụp ảnh cả tay hãy cắt ROI trước khi đưa vào dataset. Bộ nạp dữ liệu chỉ *crop giữa* rồi resize 64×64; không tự phát hiện bàn tay. Nên thu từ chính camera/cấu hình sẽ chạy thực tế. Ảnh đánh giá phải chụp buổi khác ảnh huấn luyện.

| Thư mục mỗi người | Số ảnh gợi ý | Dùng để |
|---|---:|---|
| `session_01/` | 150–200 | Train CNN và tạo template người |
| `session_02/` | 25–50 | Validation và hiệu chỉnh ngưỡng |
| `session_03/` | 25–50 | Test cuối cùng, không dùng chọn ngưỡng |

Tập `unknown/` phải chứa **người ngoài bốn thành viên**, chụp trong `session_02` (hiệu chỉnh) và `session_03` (kiểm tra độc lập). Gợi ý ít nhất 10 người ngoài nhóm, nhiều điều kiện chụp, tổng tối thiểu 100 ảnh unknown ở mỗi phần nếu có thể; càng ít ảnh thì FAR càng bất định. Không sử dụng ảnh cùng người/same capture trong cả validation và test. Unknown không dùng để train như class thứ năm: được dùng để thiết lập ngưỡng từ chối.

## 2. Cấu trúc dataset (GitHub chỉ chứa file .gitkeep)

```text
dataset_v2/
  person_01/
    session_01/  # train
    session_02/  # validate
    session_03/  # test
  person_02/
    session_01/
    session_02/
    session_03/
  person_03/
    session_01/
    session_02/
    session_03/
  person_04/
    session_01/
    session_02/
    session_03/
  unknown/
    session_01/  # không sử dụng; dành cho mở rộng
    session_02/  # người lạ hiệu chỉnh
    session_03/  # người lạ test
```

Định dạng được hỗ trợ: .jpg, .jpeg, .png, .bmp, .webp. Có thể để ảnh dưới nhiều thư mục con bên trong từng session. Không nên chụp ảnh của cùng một buổi rồi chia ngẫu nhiên vào 3 session.

**Ảnh và template là dữ liệu sinh trắc học nhạy cảm**: đã được cấu hình gitignore để tránh commit nhầm; nên lưu bản sao riêng và mã hóa nếu cần. `*.pt`, mẫu centroid, header xuất từ mô hình đều không được commit công khai.

## 3. Cài đặt (Windows PowerShell)

Từ thư mục gốc `BTL_IOT`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r palm_v2/requirements.txt
python -m palm_v2.cli inspect-data
```

Chưa có ảnh thì `inspect-data` hiển thị tất cả bằng 0; lệnh train sẽ báo thiếu dữ liệu thay vì tạo trọng số giả.

## 4. Chạy đủ pipeline (sau khi thu ảnh)

```powershell
python -m palm_v2.cli inspect-data
python -m palm_v2.cli train --epochs 40 --batch-size 16
python -m palm_v2.cli enroll
python -m palm_v2.cli calibrate
python -m palm_v2.cli evaluate
python -m palm_v2.cli verify --image "dataset_v2/person_01/session_03/test_001.jpg"
python -m palm_v2.cli export
```

Các artefact tạo ở `palm_v2/artifacts/` (best_model.pt, templates.npz, thresholds.json), không đẩy lên GitHub. Khi đổi checkpoint hoặc dataset huấn luyện, phải **enroll và calibrate lại**. Ngưỡng không phải số cố định 0.85. Báo cáo FAR/FRR chỉ đáng tin khi có nhiều bàn tay khác nhau và test độc lập.

## 5. Mô hình mới và quan hệ với V1

```text
ROI 64x64 grayscale
 -> positive-mask normalization
 -> 4 fixed Gabor 7x7
 -> Conv 3x3, 4->16, stride 2
 -> [DW 3x3 + PW 1x1 + BN + ReLU] x5
 -> GlobalAveragePooling
 -> embedding 128D, L2 normalized
 -> cosine với 4 centroid
 -> threshold -> person_01..04 hoặc unknown
```

**Khác với V1**: giảm số filter Gabor từ 32×15×15 xuống 4×7×7, thay Conv thường bằng depthwise separable, giảm embedding 512 xuống 128 và **không tái sử dụng PalmBridge 512 prototype cũ**. Đây là mô hình mới phải train từ đầu; thuật toán mới *chưa có độ chính xác được chứng minh*.

## 6. Triển khai lên ESP32

Lệnh `export` sinh `palm_v2/firmware/generated_palm_v2.h`, gồm float32 CNN, centroid và ngưỡng. File C++ `PalmLiteV2.cpp/.h` là bộ suy luận tham chiếu không cần TensorFlow Lite. Xem `palm_v2/firmware/README.md` để tích hợp vào camera. Code hiện **chưa được build hoặc benchmark trên ESP32 thật**. Cần kiểm tra độ lệch giữa kết quả Python và C++, chất lượng ROI camera, Flash, PSRAM, latency, FAR/FRR thực tế, và chống ảnh giả trước khi sử dụng mở cửa thật.

Các file gốc ở root vẫn là baseline PC V1; đừng nhầm lệnh `python main.py` (V1) với `python -m palm_v2.cli` (V2).
