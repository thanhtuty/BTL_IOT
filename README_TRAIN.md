# PalmLock-AI — Giao diện nạp ảnh & Fine-tune trên VS Code

Bản bổ sung tương thích với dự án: `https://github.com/thanhtuty/BTL_IOT`.
**Giữ nguyên toàn bộ các file hiện có.** Chỉ cần chép hai file `train_app.py` và `finetune.py` vào thư mục gốc repository; `launch_train.bat` là tuỳ chọn giúp mở ứng dụng nhanh hơn.

## 1. Chuẩn bị

```powershell
git clone https://github.com/thanhtuty/BTL_IOT.git
cd BTL_IOT
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements_train.txt
```

Nếu đã có môi trường `.venv` trong VS Code thì **không cần tạo lại**. Nếu PowerShell chặn Activate, dùng `\.venv\Scripts\python.exe` trong các lệnh.

Đảm bảo tồn tại checkpoint tại:

```text
weights/best_palmbridge_V1_embedding.pt
```

Repository GitHub đã loại file `.pt` bằng `.gitignore`. Đây là yêu cầu bắt buộc: không có checkpoint gốc thì **không thể fine-tune PalmBridge**, chỉ có thể train mô hình mới từ đầu (không thuộc quy trình này).

## 2. Mở giao diện

```powershell
python -m streamlit run train_app.py --server.address 127.0.0.1
```

Trình duyệt mở `http://localhost:8501`. Ứng dụng **chạy trên máy của bạn**; ảnh không được gửi lên cloud.

## 3. Chọn ảnh ESP32-CAM

- Giao diện có 4 ô `Người 1` đến `Người 4`, mỗi ô nhận **nhiều JPEG/PNG hoặc ZIP**.
- **Khuyến nghị mỗi người có ít nhất 3 file ZIP**, được chụp ở 3 lượt/buổi khác nhau. Mỗi ZIP đại diện cho một `session`, không trộn ảnh của một phiên sang train và test.
- Nếu chỉ có một thư mục với nhiều JPG và muốn kiểm tra xem code có chạy hay không, đánh dấu `Cho phép chạy thử khi <3 lượt chụp/người`. Chế độ này chia ảnh ngẫu nhiên và **không dùng độ chính xác test làm bằng chứng thực nghiệm**.
- Với ảnh ESP32 chứa **toàn bàn tay**: chọn `Toàn bàn tay (tự cắt ROI)`, ứng dụng dùng MediaPipe và tự tải `weights/hand_landmarker.task` lần đầu (chỉ tải mô hình từ Google, **không gửi ảnh**).
- Với ảnh đã crop đúng **lòng bàn tay**: chọn `ROI đã cắt sẵn`.
- Tất cả ảnh sẽ được chuyển thành ROI grayscale; `preprocessing.build_transform()` của mô hình gốc resize và chuẩn hóa thành 128x128 khi train.

Nhấn `Lưu ảnh và bắt đầu Fine-tune` để: **lưu ảnh → cắt ROI → chia tập → nạp trọng số → huấn luyện các lớp cuối → tạo checkpoint**.

## 4. Kết quả

```text
dataset/esp32_uploads/                  # Ảnh vừa tải lên, theo người/session
dataset/esp32_prepared/train/           # ROI được đưa vào train
dataset/esp32_prepared/val/             # ROI validation
dataset/esp32_prepared/test/            # ROI test
weights/palmbridge_esp32_finetuned.pt  # Checkpoint mới
weights/finetune_report.json           # Kết quả đánh giá
weights/finetune_history.csv           # Log từng epoch
```

Test accuracy được tính theo **4 danh tính đã biết** bằng nearest-centroid embedding. Đây **không** phải khả năng từ chối người lạ. Không dùng kết quả để mở khóa cửa thật cho đến khi hiệu chỉnh ngưỡng cosine, đánh giá FAR/FRR, và có phương án mở khóa cơ học.

Sau khi hoàn thành, mở Terminal để thử:

```powershell
python main.py --weights weights/palmbridge_esp32_finetuned.pt inspect
python main.py --weights weights/palmbridge_esp32_finetuned.pt embedding --image dataset/esp32_prepared/test/person_01/TEN_FILE.png
```

**Thay `TEN_FILE.png` bằng tên ảnh có thật trong thư mục test.** Sau đó dùng `enroll/verify/evaluate` hiện có trong `main.py` với cùng `--weights`.

## 5. Chạy bằng CLI không dùng giao diện

Nếu ảnh của bạn đã sắp vào `dataset/esp32_uploads/person_XX/session_XX/*.jpg`:

```powershell
python finetune.py --raw dataset/esp32_uploads --input-format full-hand --epochs 12 --batch-size 4
```

Nếu đã cắt ROI:

```powershell
python finetune.py --raw dataset/esp32_uploads --input-format roi --epochs 12 --batch-size 4
```

Để chỉ thử code với dưới 3 phiên chụp mỗi người:

```powershell
python finetune.py --raw dataset/esp32_uploads --input-format roi --epochs 1 --batch-size 2 --allow-random-split
```

## 6. Giới hạn cần biết

- Vì ảnh là sinh trắc học, chỉ sử dụng ảnh khi người được chụp đồng ý; giới hạn quyền truy cập, giữ riêng dữ liệu, không commit ảnh hoặc `.pt` lên GitHub.
- Train trên CPU có thể chậm; VS Code không tự cung cấp GPU.
- Ảnh quá mờ hoặc MediaPipe crop sai sẽ không được sửa bằng fine-tune; kiểm tra ROI trước và loại ảnh lỗi.
- Checkpoint PalmBridge được nạp qua `models/checkpoint.py`, bắt buộc đúng định dạng. File `.pt` từ model khác không thể dùng thay.
- Chưa đánh giá trên dữ liệu thật của bạn; test kỹ thuật chỉ khẳng định code chạy với dữ liệu giả lập, **không khẳng định độ chính xác**.