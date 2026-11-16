# B5 — thử Gabor tuần tự và ROI 64×64 (Custom Compact CompNet)

Chạy **trên PC Python/PyTorch** trong nhánh `experiment/compact-compnet-esp32`.
Bản B5 **chưa phải firmware ESP32** và **không chạy trên Wokwi**. Nó là phép
thử có kiểm chứng để xem có giảm được feature-map RAM mà giữ nhận diện không.

## Mục tiêu & các bản cần giữ riêng

| Phiên bản | Độ phân giải | Checkpoint | Dùng để |
|---|---|---|---|
| Original | 128×128 | weights/palmbridge_esp32_finetuned.pt | B1–B4, không sửa |
| Compact B2 | 128×128 | weights/palmbridge_compact_pool2.pt | Mốc 17/32, không sửa |
| Compact B5 | 64×64 | **weights/palmbridge_compact_roi64.pt** | Train mới từ B2 |
| Streaming Gabor | 128×128, cũng hỗ trợ 64 | **dùng chung trọng số B2/B5** | Kiểm thử tính tương đương khi suy luận |

Cả 128 và 64 dùng đúng code Gabor, 4 CNN blocks, PalmBridge 512-vector
và embedding 512 chiều. 64 chỉ giảm độ phân giải tensor đầu vào; hình học
Pooling `AdaptiveAvgPool2d((2,2))` và số tham số vẫn như B2.

### Bước 0 — Update GitHub (không mất .pt/ảnh)

Mở PowerShell trong **PalmLock-AI-Git** (không đứng ở thư mục `code`):

```powershell
git branch --show-current
git status --short
git fetch origin "refs/heads/experiment/compact-compnet-esp32"
git merge --ff-only FETCH_HEAD
```

Nếu Git từ chối `merge` vì có code chưa commit, đừng dùng `git reset --hard`.
Hãy giữ lại thay đổi và kiểm tra trước; ảnh `dataset/`, `test_photos/`
và file `weights/*.pt` được ignore.

### Bước 1 — Kiểm thử cấu trúc B5

```powershell
python -m unittest discover -s tests -p "test_b5.py" -v
```

Test được thực hiện bằng tensor/checkpoint tạo ngẫu nhiên, kiểm tra:
- Gabor `ReLU -> max` tuần tự so với xử lý đủ 32 kênh;
- embedding 512-D giữa backbone chuẩn và streaming cùng tham số;
- checkpoint Custom Compact 64 và 128 được load đúng `image_size`;
- số byte của bộ đệm Gabor theo lý thuyết.

### Bước 2 — B5A: không cần train, đánh giá Gabor tuần tự

```powershell
python experiments/b5.py memory
python experiments/b5.py stream-test --samples 3 --device cpu
```

Tạo:
- `experiments/results/b5_gabor_memory_estimate.json`
- `experiments/results/b5_stream_report.json`

Mục tiêu của `stream-test`: so sánh đầu ra thật trên cùng ảnh ROI Validation
và cùng checkpoint Compact B2, kỳ vọng cosine của hai embedding gần 1,
sai lệch Gabor tối đa dưới 1e-4.

| Chỉ tính Gabor FP32, 1 ảnh | Bản 32 map | Streaming 2 buffer |
|---|---:|---:|
| ROI 128×128 | 2,097,152 byte (2 MiB) | 131,072 byte (128 KiB) |
| ROI 64×64 | 524,288 byte (512 KiB) | 32,768 byte (32 KiB) |

Các con số trên **không** bao gồm convolution scratch, CNN activation,
camera buffer, Gabor kernels, MediaPipe hay runtime. Streaming có thể
**chậm hơn** do chạy nhiều phép conv riêng, vì vậy báo cáo cũng đo
thời gian chạy PC (không áp dụng trực tiếp cho ESP32).

### Bước 3 — B5B: Fine-tune ROI 64×64 (vẫn dùng ảnh cũ)

```powershell
python experiments/b5.py train64 --epochs 12 --batch-size 4 --device cpu --seed 42
```

- Khởi tạo từ `weights/palmbridge_compact_pool2.pt`, không train từ zero.
- Tự lấy `dataset/esp32_prepared/train` và `val`; đây đều là ROI đã cắt.
- Chỉ resize thành **64×64 trước chuẩn hóa**, không thay nhãn.
- Giữ nguyên BN statistics và chỉ fine-tune block4, embed, PalmBridge.
- Chọn epoch bằng accuracy trên Validation, hòa thì dùng cosine margin.
- KHÔNG dùng `test_photos` để train hoặc chọn epoch.
- Checkpoint đầu ra: `weights/palmbridge_compact_roi64.pt` (khác tên B2).
- Báo cáo: `experiments/results/b5_train64_history.json`.

Kiểm tra model 64:

```powershell
python main.py --weights weights/palmbridge_compact_roi64.pt inspect
```

Kỳ vọng: `backbone_type: custom_compact`, `image_size: [64,64]`.
`PalmEmbedder` tự điều chỉnh resize theo metadata `image_size`; checkpoint
128 cũ vẫn resize 128.

### Bước 4 — B5C: So sánh cùng 32 ảnh Test (sau train64)

```powershell
python experiments/b5.py compare64 --tests test_photos --input-format full-hand --device cpu
```

Nếu ảnh Test đã **là ROI**, thay `--input-format roi`.

Bộ kiểm thử:
1. Mỗi ảnh cả bàn tay được MediaPipe crop thành ROI đúng **một lần**.
2. Từ cùng ROI, tạo input 128 và 64 bằng hai phép resize riêng.
3. Chạy đúng checkpoint 128 và 64, mỗi model có centroid tính từ
   chính train ảnh tương ứng cùng cách tiền xử lý.
4. Xuất nhầm lẫn từng người `person_01..04`, score cosine và Accuracy.

Kết quả:
- `experiments/results/b5_compare_summary.json`;
- `experiments/results/b5_compare_per_image.csv`.

**Điểm bắt buộc:** model B2 trên ảnh Test trước đó đạt 17/32 (53,125%).
B5 đang kiểm tra khả năng giữ hoặc cải thiện kết quả nhưng **chưa có số
mới cho model 64**. Không được suy luận rằng ROI 64 sẽ tăng accuracy.
Vì 32 ảnh đã được xem nhiều lần, khi chọn mô hình cuối cùng cần thu
**một bộ ảnh Test mới hoàn toàn độc lập**, và thêm người ngoài 4 danh tính.

### Bước 5 — Kết luận khả năng lên ESP32

- Tối ưu streaming Gabor chỉ áp dụng cho **đầu ra 32 filter** tại inference,
  không giảm bộ nhớ những CNN blocks khác và không tự chuyển sang C++.
- ROI 64 giảm bộ nhớ các tensor đầu vào và convolution features, **không
  giảm số tham số** của Compact.
- MediaPipe trên PC vẫn được sử dụng để cắt ROI trong phép thử; để chạy
  độc lập trên ESP32-CAM cần pipeline ROI tương thích.
- Muốn firmware thật phải đo bộ nhớ đỉnh và thời gian trên chip,
  thay những toán tử không tương thích, lượng tử hóa activation và weight
  bằng toolchain tương thích ESP-DL/ESP-PPQ.
- Không điều khiển khóa thật với accuracy hiện tại vì Person 04 bị nhầm
  nghiêm trọng và chưa hiệu chỉnh ngưỡng/FAR/FRR.

### Quy trình gửi kết quả

Hãy chạy theo thứ tự:

```powershell
python -m unittest discover -s tests -p "test_b5.py" -v
python experiments/b5.py stream-test --samples 3 --device cpu
python experiments/b5.py train64 --epochs 12 --batch-size 4 --device cpu --seed 42
python experiments/b5.py compare64 --tests test_photos --input-format full-hand --device cpu
```

Gửi lại `b5_stream_report.json` và `b5_compare_summary.json`.
