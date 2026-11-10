# PalmLock-AI — Starter Kit nhận diện lòng bàn tay (PC)

**Nguồn:** Trích mã kiến trúc/tiền xử lý trong `palmbridge.ipynb` của người dùng (cell 2). Phần CLI, đăng ký, so sánh, database được viết thêm phục vụ nhóm 4 người.

**Trạng thái:** Repository chứa mã nguồn chạy **Python/PyTorch trên máy tính**, không phải firmware ESP32-S3. Checkpoint, model dò tay, ảnh lòng bàn tay và database sinh trắc được giữ ngoài GitHub. Độ chính xác trên ảnh thực tế chưa được đánh giá đầy đủ.

## 1. Điều kiện tiên quyết

- Python 3.10–3.12 (hoặc bản Python được PyTorch hỗ trợ), pip, máy Windows/Linux.
- Có checkpoint `best_palmbridge_V1_embedding.pt` được tạo ra từ PalmBridge **cell 2**, hoặc checkpoint `best_palmbridge_V1_xpalm.pt` tương thích.
- Có ảnh ROI đã cắt sẵn; hoặc cài MediaPipe và tải model `hand_landmarker.task` để dùng chế độ tự dò/cắt từ ảnh cả bàn tay.

## 2. Cài đặt trên Windows (PowerShell)

```powershell
cd PalmLock-AI
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu không kích hoạt được venv, dùng `.\.venv\Scripts\python.exe main.py ...` và `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

Nếu máy cần GPU, chọn bản torch và torchvision tương ứng từ https://pytorch.org/get-started/locally/; có thể thử CPU trước.

## 3. Bổ sung trọng số quan trọng

Repository không chứa checkpoint `.pt`. Từ notebook Kaggle `palmbridge.ipynb` cell 2 (sau khi chạy xong fine-tuning), lấy file output:

```text
/kaggle/working/palmbridge_V1_xpalm_finetune/outputs/best_palmbridge_V1_embedding.pt
```

Đặt file tại `weights/best_palmbridge_V1_embedding.pt` hoặc truyền đường dẫn thật bằng `--weights`. Các file `.pt` bị loại khỏi GitHub.

> Chú ý: tiêu đề cell có ghi V1 nhưng `PRETRAINED_PATH` trong code nạp `palmbridge_V2.pt`. Phải lấy checkpoint khớp đúng với lần huấn luyện và không trộn với checkpoint CompNet/CCNet.

## 4. Chạy thử bước đầu

```powershell
python main.py inspect
python main.py embedding --image dataset\person_01\roi_01.png
```

`inspect` sẽ in loại backbone `official` hay `custom`, kích thước embedding 512, kích thước ảnh 128×128.

### 4.1 Ảnh chụp cả bàn tay

Tải model MediaPipe Hand Landmarker từ [Google AI Edge](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) và lưu thành `weights/hand_landmarker.task`:

```powershell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" `
  -OutFile "weights\hand_landmarker.task"
```

Xem trước ROI và khung crop màu đỏ:

```powershell
python main.py crop-roi --image photos\hand.jpg
```

Lệnh tạo ảnh ROI và ảnh gốc có khung đỏ trong `roi_previews/`, tách khỏi thư mục ảnh đăng ký. Kiểm tra khung đỏ có ôm phần lòng bàn tay mong muốn không. Có thể chỉnh khung bằng `--roi-width-scale` và `--roi-height-scale`.

Ảnh nên thấy rõ một bàn tay đang mở, lòng bàn tay hướng về camera, gồm cả cổ tay và các khớp gốc ngón tay; tránh để bàn tay sát mép ảnh.

Sau khi xem trước, dùng ảnh toàn bàn tay trực tiếp bằng `--input-format full-hand`:

```powershell
python main.py embedding --image photos\hand.jpg --input-format full-hand
python main.py verify --image photos\hand.jpg --input-format full-hand
```

Khi đăng ký hoặc đánh giá cả thư mục ảnh, thêm cùng tùy chọn `--input-format full-hand`. Dùng cùng một cách chụp và crop khi enroll lẫn verify. Crop tự động giúp thống nhất ảnh đầu vào; độ chính xác vẫn cần được đánh giá với ảnh của nhóm vì ROI có thể khác cách crop của dữ liệu huấn luyện.

## 5. Chuẩn bị dataset của nhóm

Ví dụ (file ảnh không có sẵn trong ZIP):

```text
dataset/
  person_01/
    roi_01.png
    roi_02.png
    ...
  person_02/
    ...
  person_03/
    ...
  person_04/
    ...
tests_data/
  person_01/
    test_01.png
  person_02/
    test_01.png
  outsider_01/
    test_01.png
```

Ảnh trong `dataset/` dùng **đăng ký**. Ảnh trong `tests_data/` là ảnh **mới chụp tách riêng**, dùng **đánh giá**; không được dùng lại ảnh đăng ký trong test. Hai thư mục có thể chứa ROI đã cắt sẵn hoặc ảnh cả bàn tay; với ảnh cả bàn tay, thêm `--input-format full-hand` vào lệnh tương ứng. Thu thập thêm ảnh người ngoài nhóm để đo từ chối người lạ. Nên thống nhất cùng một tay, cách chụp và khung ROI giữa enroll và verify.

## 6. Đăng ký 4 thành viên (enroll, KHÔNG phải train)

```powershell
python main.py enroll --user person_01 --images dataset\person_01
python main.py enroll --user person_02 --images dataset\person_02
python main.py enroll --user person_03 --images dataset\person_03
python main.py enroll --user person_04 --images dataset\person_04
```

Nếu thư mục chứa ảnh chụp cả bàn tay, thêm `--input-format full-hand` vào từng lệnh enroll.

Template vector của mỗi người lưu vào `database/users.json`, ràng buộc với SHA-256 checkpoint hiện tại để tránh so sánh khác model. Nếu enroll lại một user, template cũ sẽ được thay thế. Thư mục database chứa **dữ liệu sinh trắc học nhạy cảm**, không nên chia sẻ công khai hoặc commit lên GitHub.

## 7. Nhận diện ảnh mới

```powershell
# Mặc định dùng ngưỡng demo 0.85; cần hiệu chỉnh trước khi dùng thực tế
python main.py verify --image tests_data\person_01\test_01.png

# Có thể ghi đè bằng ngưỡng đã hiệu chỉnh từ dữ liệu độc lập:
python main.py verify --image tests_data\person_01\test_01.png --threshold 0.90
```

Với ảnh cả bàn tay, thêm `--input-format full-hand` vào lệnh verify.

**0.85 chỉ là ngưỡng mặc định cho demo, chưa được xác thực.** Không dùng để mở khóa thật trước khi đo FAR/FRR trên dữ liệu độc lập.

## 8. Đánh giá tập test giữ riêng

```powershell
# Chỉ ví dụ minh họa tham số. Ngưỡng cần được hiệu chỉnh từ dữ liệu độc lập.
python main.py evaluate --tests tests_data --threshold 0.85
```

Nếu `tests_data/` chứa ảnh cả bàn tay, thêm `--input-format full-hand`.

Chương trình báo `FAR`, `FRR` và tỷ lệ nhận sai ID của thành viên. Với 4 người và ít ảnh, các tỷ lệ dễ bị biến động mạnh. Để lựa chọn ngưỡng khách quan, hãy dùng tập **hiệu chỉnh** tách riêng khỏi `tests_data` nếu đủ dữ liệu.

## 9. Chạy unit tests (không cần checkpoint thật)

```powershell
python -m unittest discover -s tests -v
```

Kiểm thử ở đây chỉ xác nhận shape/forward/preprocessing/checkpoint nhân tạo; KHÔNG có nghĩa mô hình nhận diện đã tốt. Test checkpoint nhân tạo được tạo bằng trọng số ngẫu nhiên ở thư mục tạm, KHÔNG dùng để xác thực người.

## 10. Luồng file và ý nghĩa

| File | Chức năng |
|---|---|
| `config.py` | Kích thước ảnh 128, vector 512, đường dẫn mặc định |
| `preprocessing.py` | `Resize`, `ToTensor`, `NormSingleROI` |
| `roi.py` | Dò landmark và căn/cắt ROI lòng bàn tay từ ảnh toàn bàn tay |
| `models/backbone.py` | Hai kiến trúc CompNet từ notebook |
| `models/palmbridge.py` | Codebook vector `P` và hàm tạo embedding |
| `models/checkpoint.py` | Nạp checkpoint đúng backbone, có kiểm tra |
| `inference.py` | Model được nạp một lần và trích vector từ ảnh |
| `enroll.py` | Tạo template trung bình và lưu người đăng ký |
| `verify.py` | So sánh cosine, đưa ra ứng viên và kết quả theo ngưỡng |
| `evaluate.py` | Đánh giá tập test độc lập |
| `main.py` | Giao diện dòng lệnh |

Nguồn và dòng code: xem `SOURCE_MAP.md`. Quy trình ESP32-S3 giai đoạn sau: xem `ESP32_NEXT.md`.

## 11. Bảo mật và giới hạn

- Chỉ nạp checkpoint từ nguồn tin cậy; `torch.load(weights_only=True)` hạn chế rủi ro pickle.
- `users.json` chứa dữ liệu sinh trắc học; không upload/public hoặc chia sẻ tùy ý.
- Ảnh giả (ảnh in/ảnh trên màn hình) có thể đánh lừa hệ thống nếu không có kiểm tra sống (*liveness*).
- Tên `person_01` tránh dùng thông tin cá nhân trong DB.
- **ESP32-S3 4MB Flash, 2MB PSRAM** không đủ chạy PalmBridge/CompNet nguyên bản. ZIP chỉ là điểm xuất phát đánh giá baseline, không phải firmware sẵn chạy trên board.

