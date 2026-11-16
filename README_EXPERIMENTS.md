# Thử nghiệm tối ưu Custom CompNet trên ESP32-CAM (B1 → B4)

Branch: `experiment/compact-compnet-esp32`. **KHÔNG ghi đè** model/ảnh/checkpoint/database trước đây.
Checkpoint cũ: `weights/palmbridge_esp32_finetuned.pt`; checkpoint mới sau train:
`weights/palmbridge_compact_pool2.pt`. Tất cả giữ trên máy bạn.

## 0) Chuyển nhánh và kiểm tra môi trường (VS Code PowerShell)

```powershell
git fetch origin
git switch -c experiment/compact-compnet-esp32 --track origin/experiment/compact-compnet-esp32
python -m pip install -r requirements.txt
python main.py --weights weights/palmbridge_esp32_finetuned.pt inspect
```

Checkpoint gốc bắt buộc có `backbone_type = custom`; nếu không, dừng để kiểm tra.
Không dùng nhánh `feat/live-esp32cam-palmbridge` để train bản 2×2.

## B1 — kiểm tra ROI và nhãn (KHÔNG train)

```powershell
python experiments/roi_audit.py --prepared dataset/esp32_prepared --tests test_photos --test-format full-hand
```

Xem kết quả trong:

- `experiments/results/roi_contact_sheet.jpg`: đối chiếu ROI train / validation / ảnh mới của 4 người. Cần quan sát kỹ Person 01 và Person 04, chú ý vùng cắt, hướng tay, bàn tay trái/phải, ánh sáng.
- `experiments/results/roi_audit.json`: ảnh lỗi, độ tối/độ sáng bất thường, trùng lặp chính xác giữa split, dấu hiệu ảnh gần trùng qua dHash.
- **dHash chỉ là gợi ý**, không phải bằng chứng trùng ảnh hay gán nhãn đúng/sai.

Nếu thấy ROI bị cắt lệch, **chưa train tiếp**. Chụp/sắp xếp lại ảnh theo buổi chụp độc lập; đừng sửa riêng ảnh Test để lấy kết quả đẹp.
Nếu ảnh test đã là ROI, thay `--test-format roi`.

## B2 — thay AdaptiveAvgPool 4×4 → 2×2, fine-tune bằng ảnh TRAIN

Trong `models/compact.py`: giữ nguyên Gabor 32 filter, CNN block1..block4, PalmBridge 512×512 và embedding 512; chỉ thay GAP và Linear.

- Linear cũ: 4096→512 = 2.097.152 tham số.
- Linear mới: 1024→512 = 524.288 tham số (giảm đúng **75% số tham số FC**).
- Khởi tạo FC mới bằng tổng các trọng số 2×2 không gian từ FC gốc; các layer còn lại nạp từ checkpoint cũ.
- Chỉ train block4, embed và PalmBridge; thêm ArcFace head phụ **chỉ để train**.
- Checkpoint được chọn bằng accuracy **cosine nearest centroid trên validation**; khi tất cả epoch cùng accuracy (ví dụ 100%) thì chọn epoch có **cosine margin trung bình lớn nhất** giữa người đúng và đối thủ gần nhất. Đây vẫn chỉ là tiêu chí Validation nhỏ, không bảo đảm tổng quát hóa.
- Kết quả validation vẫn lạc quan nếu ảnh bị chia ngẫu nhiên cùng buổi: cần 3 buổi chụp độc lập/người để kiểm chứng.
- Ảnh `test_photos` KHÔNG được dùng để train hoặc chọn epoch.

```powershell
python experiments/benchmark.py train --epochs 12 --batch-size 4 --device cpu --seed 42
```

Lưu checkpoint **mới**: `weights/palmbridge_compact_pool2.pt`.

Kiểm tra nó nạp được qua main.py:

```powershell
python main.py --weights weights/palmbridge_compact_pool2.pt inspect
```

Kết quả phải có `backbone_type: custom_compact`. Sau đó:

```powershell
python experiments/benchmark.py compare --tests test_photos --input-format full-hand --device cpu
```

Hai phiên bản được đánh giá bằng **CÙNG ảnh**, **CÙNG ROI** và **CÙNG cách lấy centroid từ `dataset/esp32_prepared/train`**.
Kết quả: `experiments/results/compare_summary.json` + `compare_per_image.csv`.
Kết quả này có thể khác 13/32 ban đầu vì lần trước dùng database enroll đã lưu, còn lần so sánh mới tạo centroid từ tập train nhằm công bằng giữa hai checkpoint. Không thể so sánh hai accuracy đo bởi hai giao thức khác nhau mà kết luận tiến bộ.

## B3 — nghiên cứu độ nhạy trọng số INT8

```powershell
python experiments/benchmark.py quantize --device cpu
```

Tạo `experiments/results/int8_proxy_report.json` và `compact_int8_weights_only.npz`.
Phương pháp: lượng tử hóa đối xứng per-tensor từng parameter INT8 rồi **giải lượng tử lại về FP32** để chạy thử trong PyTorch; tính tương đồng embedding, accuracy nearest-centroid trên validation và kích thước trọng số.

**Rất quan trọng:** Đây là **weight-only quantization sensitivity test**, chưa phải mô hình W8A8 / firmware ESP32:
- Các activation và convolution vẫn thực thi FP32.
- Còn phải test hậu lượng tử hóa W8A8/W8A16 với ESP-PPQ, hỗ trợ toán tử và sự thay đổi sai số.
- Không được đổi đuôi `.npz` thành `.espdl` hoặc gọi đây là AI trên chip.

ESP-DL chính thức dùng `.espdl`. Với ESP32 classic, ESP-PPQ target=`c`, build ESP-IDF target=`esp32` (theo tài liệu Espressif). Nhưng custom LearnableGabor sinh filter động và PalmBridge argmin/gather phải được xử lý/tách graph; **chưa có exporter hoàn chỉnh**.

Tham khảo:
- https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_quantize_model.html
- https://github.com/espressif/esp-dl/blob/master/operator_support_state.md

## B4 — báo cáo dung lượng cho ESP32-CAM 4 MB Flash, 2 MB PSRAM

```powershell
python experiments/benchmark.py size
```

Tạo `experiments/results/esp32_feasibility.json`. Báo cáo kích thước checkpoint, số tham số, FP32, cận dưới trọng số INT8 và bộ đệm Gabor FP32.
**Đây là dự toán**, không phải đo RAM đỉnh thực tế trên chip. Tensor Gabor gốc có kích thước 32×128×128×4 = 2.097.152 bytes, bằng 2 MiB trước khi tính ảnh gốc/hoạt động khác. Chưa có ESP-DL memory planner và firmware tương thích nên **không thể tuyên bố model đã vừa board**.

## Trình tự chuẩn và tiêu chí quyết định

1. **B1**: mở contact sheet, sửa ảnh bị cắt lệch hoặc nhãn sai, gom thành ít nhất ba session độc lập/người, tuyệt đối giữ riêng một buổi cuối làm Test.
2. **B2**: train và so sánh trên cùng ảnh, theo dõi cả 4 người. Không đánh giá dựa riêng vào accuracy tổng hợp.
3. **B3**: nếu accuracy không tệ đi nhiều, đo mức thay đổi cosine khi lượng tử hóa; chỉ khi khả thi mới nghiên cứu W8A8 và runtime.
4. **B4**: đo kích thước và hỗ trợ toán tử; nếu vượt Flash/PSRAM thì phải thay đổi kiến trúc thêm hoặc đổi board.
5. Trước khi mở khóa thật: phải bổ sung người lạ, ngưỡng nhận diện dựa FAR/FRR, kiểm tra giả mạo và cơ chế mở khóa an toàn.

Tất cả kết quả và ảnh sinh trắc của nhóm nằm trong `experiments/results/`, đã được `.gitignore` loại khỏi Git.
Không sử dụng ảnh Test để fine-tune, định ngưỡng hay chọn kiến trúc.

**Trạng thái:** Code đã được xây dựng; checkpoint và ảnh dữ liệu thật không có trên GitHub nên không thể chạy đánh giá accuracy thật cho 4 người từ xa. Các lệnh trên chạy trực tiếp trên máy bạn.


## Lưu ý sau khi hoàn tất 12 epoch mà Validation luôn 100%

Bản `benchmark.py` đầu tiên chỉ lưu khi `val_acc > best`; nếu tất cả
epoch đều 100%, checkpoint lưu ngay ở epoch **1**. Đã sửa cách chọn epoch:
ưu tiên accuracy, khi hòa thì lấy **mean cosine margin** lớn nhất trên Validation.
Điều này KHÔNG có nghĩa accuracy trên ảnh mới sẽ tốt hơn, đặc biệt khi chỉ
có 15 ảnh Validation.

Đối với người đã train với bản code cũ:

```powershell
Copy-Item weights/palmbridge_compact_pool2.pt weights/palmbridge_compact_pool2_epoch1_backup.pt
git fetch origin "refs/heads/experiment/compact-compnet-esp32"
git merge --ff-only FETCH_HEAD
python -m unittest discover -s tests -p "test_custom_compact.py" -v
python experiments/benchmark.py train --epochs 12 --batch-size 4 --device cpu --seed 42
```

Kiểm tra epoch được chọn:

```powershell
python -c "import torch; p=torch.load('weights/palmbridge_compact_pool2.pt',map_location='cpu',weights_only=True); print('Epoch:',p['epoch'],'Val accuracy:',p['val_centroid_accuracy'],'Cosine margin:',p.get('val_cosine_margin'))"
```

## B5 — Gabor streaming và ROI 64×64

Đã bổ sung thử nghiệm riêng, **không ghi đè** checkpoint B2.
Xem toàn bộ trình tự và lệnh chính xác tại [README_B5.md](README_B5.md).

```powershell
python -m unittest discover -s tests -p "test_b5.py" -v
python experiments/b5.py memory
python experiments/b5.py stream-test --samples 3 --device cpu
python experiments/b5.py train64 --epochs 12 --batch-size 4 --device cpu --seed 42
python experiments/b5.py compare64 --tests test_photos --input-format full-hand --device cpu
```

B5 hiện mới đánh giá trên **PyTorch PC**, chưa có .espdl hoặc firmware ESP32-CAM.
Checkpoint 64 cần đăng ký template sinh trắc học mới nếu sau này dùng trong
ứng dụng verify thật; database đã enroll với checkpoint 128 không tương thích
với checkpoint 64 (hash và vector embedding thay đổi).
