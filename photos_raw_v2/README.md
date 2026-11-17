# PHOTOS_RAW_V2 — ẢNH GỐC TỪ ĐIỆN THOẠI/CAMERA

Đặt ảnh chụp **nguyên bàn tay** theo cùng cấu trúc person_01..04/unknown và session_01..03. Sau đó chạy offline trên máy tính:

```powershell
pip install mediapipe
python -m palm_v2.cli prepare-roi
```

Ảnh gốc được giữ nguyên; ảnh ROI sẽ được ghi sang `dataset_v2/`. Đừng đưa ảnh thật của người dùng lên GitHub công khai. Đối với demo trực tiếp trên ESP32, crop ROI của ảnh live phải giống ROI dùng huấn luyện — thiết bị không chạy MediaPipe.
