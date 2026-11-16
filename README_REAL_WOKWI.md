# Mô phỏng đầu-cuối PalmLock: ảnh mới + AI thật + thiết bị Wokwi

**Đây là chế độ riêng; KHÔNG dùng bảng điểm ghi sẵn của `wokwi/`.**

```text
Webcam LAPTOP (ảnh thật) hoặc ESP32-CAM THẬT /capture
    ↓ JPEG mới mỗi lần bấm
PC Python: MediaPipe HandLandmarker → cắt ROI thật
    ↓ NormSingleROI → 64x64 (hoặc 128 nếu checkpoint yêu cầu)
PC Python: CustomCompactBackbone + PalmBridge, weights checkpoint ĐÃ TRAIN
    ↓ embedding 512 chiều; cosine với centroid Train 4 người
    ↓ MQTT chỉ gửi P,person_index,cosine_x10000,sequence
Wokwi: ESP32 DevKitC nhận MQTT → OLED, LEDs, buzzer, servo, công tắc cửa
```

**PHẢI PHÂN BIỆT:** AI chạy THẬT trên PC, không chạy trong Wokwi và không
chạy trên chip ESP32-CAM. Wokwi hiện không có camera OV2640 để lấy hình
như board thật. Nếu bài của thầy bắt buộc thuật toán chạy trên board,
bộ thử này chỉ xác thực tích hợp camera/AI/thiết bị, **chưa hoàn thành
yêu cầu xử lý trên chip**. Muốn chứng minh trên chip cần viết firmware
chuyển đổi/tối ưu, đo RAM và thời gian trên board thật.

## Bước 1 — cập nhật code và kiểm tra trọng số

Mở PowerShell trong `PalmLock-AI-Git`, đang ở nhánh
`experiment/compact-compnet-esp32`:

```powershell
git fetch origin "refs/heads/experiment/compact-compnet-esp32"
git merge --ff-only FETCH_HEAD
python -m pip install -r requirements.txt
python -m pip install -r live/requirements_live.txt
python main.py --weights weights/palmbridge_compact_roi64.pt inspect
```

Model mặc định là Compact ROI64 từ B5. Nếu chưa có bản ROI64,
thay `--weights weights/palmbridge_compact_pool2.pt` khi chạy server.
Chương trình tự đọc metadata image_size của checkpoint.

Cần có:
- `weights/palmbridge_compact_roi64.pt` (hoặc pool2 128)
- `weights/hand_landmarker.task`
- `dataset/esp32_prepared/train/person_01..person_04/` đã có ROI Train
- Webcam laptop (hoặc ESP32-CAM thật trong cùng LAN)

**Lưu ý:** đừng dùng `database/*.json` đã enroll bằng checkpoint khác.
Ứng dụng mới này **tính lại 4 centroid từ Train** với chính checkpoint
đã chọn, đúng cách so sánh B5 (tuyệt đối không dùng Test để enroll).

## Bước 2 — tạo topic MQTT riêng cho phiên thử

Mở PowerShell, chạy:

```powershell
python -c "import secrets; print('palmlock/demo/' + secrets.token_hex(12))"
```

Ví dụ lệnh in ra:

```text
palmlock/demo/6fa9d6c800bc7a52101a52d8
```

**Đây là ví dụ, hãy dùng kết quả ngẫu nhiên của bạn.**
Nội dung phải giống nhau trong mã Wokwi và lệnh chạy Python.

## Bước 3 — tạo mạch Wokwi LIVE (không dùng mạch replay cũ)

1. Mở https://wokwi.com/projects/new/esp32
2. Copy `wokwi_live/diagram.json` → tab `diagram.json`.
3. Copy `wokwi_live/sketch.ino` → tab `sketch.ino`.
4. Tạo tab `libraries.txt`, copy `wokwi_live/libraries.txt`.
5. Trong `sketch.ino` tìm dòng:
   ```cpp
   static const char *TOPIC = "palmlock/demo/REPLACE_WITH_SAME_RANDOM_ID";
   ```
   Đổi thành topic ngẫu nhiên vừa tạo.
6. **Start Simulation**. Trong Serial Monitor, chờ:
   ```text
   [MQTT] CONNECTED, subscribed: palmlock/demo/<topic-cua-ban>
   ```

Nếu chưa CONNECTED, kiểm tra Wi-Fi Wokwi-GUEST, Internet/gateway,
topic và broker công cộng.

### Thiết bị trong mạch

| Phần cứng | Chân trên ESP32 DevKitC mô phỏng |
|---|---|
| Servo PWM | GPIO18 |
| OLED SDA/SCL | GPIO21 / GPIO22 |
| Buzzer | GPIO19 |
| Cảm biến cửa (công tắc) | GPIO23, LOW = đóng |
| LED xanh/đỏ | GPIO26 / GPIO27 |

Chân trên ESP32-CAM THẬT KHÔNG giống DevKitC; nhiều chân đã được
camera chiếm dụng. **Không sao chép sơ đồ này sang module camera.**

## Bước 4 — chạy AI + camera thật trên PC

Tại PowerShell đang bật `.venv`:

```powershell
python live/live_wokwi.py --topic "palmlock/demo/THAY_BANG_ID_CUA_BAN" --weights weights/palmbridge_compact_roi64.pt
```

Mở **http://127.0.0.1:8766** trong Chrome/Edge trên laptop:

1. Nhấn **Bật webcam** và chấp nhận quyền camera.
2. Giơ lòng bàn tay đủ sáng, thấy rõ gốc các ngón và cổ tay.
3. Nhấn **Chụp ảnh và chạy AI**.
4. Web sẽ hiện ảnh thật vừa chụp, ROI mới được crop, kết quả
   `person_01..04`, cả bốn điểm cosine.
5. MQTT gửi **một frame mới cho chính ảnh đó**.
6. Wokwi hiển thị người dự đoán và điểm lên OLED;
   servo/LED/buzzer hoạt động để bạn **test chức năng thiết bị**.

Nếu Wokwi nhận được gói, Serial Monitor in
`[REAL PC AI] image seq=... predicted person_.. cosine=...`.

**Mặc định `BENCH_PULSE_EVERY_RESULT = true` trong Wokwi**:
- Servo **quay thử 90° trong 2,5 giây cho MỌI kết quả AI hợp lệ**.
- Điều này chỉ thử cơ cấu chốt khi một frame nhận diện hoàn tất,
  **KHÔNG có nghĩa người đó đã xác thực**; kể cả nhận sai cũng quay.
- Wokwi vẫn hiện PASS/FAIL của ngưỡng 0.85. Dự án hiện chưa có
  ngưỡng thật đủ an toàn (Person01/04 nhầm nhau).
- Để chỉ xoay khi cosine đạt ngưỡng demo 0.85, đặt
  `BENCH_PULSE_EVERY_RESULT = false`. Khi đó mô hình hiện tại thường
  không đủ 0.85 để quay servo, nhưng UI vẫn hiện kết quả thật.
- **KHÔNG nạp firmware này vào khóa thật**.

Công tắc cửa ở `OPEN` sẽ chặn chuyển động ngay cả trong thử nghiệm.

### Dùng camera OV2640 thật thay webcam

Nếu có board thật đã nạp `live/esp32cam/esp32cam_live.ino`, in IP
`http://192.168.1.123/capture` trong Serial Monitor, chạy:

```powershell
python live/live_wokwi.py --topic "palmlock/demo/THAY_BANG_ID_CUA_BAN" --camera-url http://192.168.1.123/capture
```

Trên trang web, nhấn **Chụp từ ESP32-CAM**. Đây là JPEG mới lấy từ
sensor OV2640 thật, sau đó cùng pipeline AI trên PC. **Không thay đổi
AI hoặc nội dung MQTT**. Nếu chưa có camera board, webcam laptop là
sensor ảnh thật phù hợp để thử pipeline trước.

## Điểm cần biết về MQTT demo

- Wokwi Free có thể kết nối mạng ra Internet qua Wokwi Public Gateway.
- Ví dụ dùng `test.mosquitto.org:1883`: **broker công cộng KHÔNG TLS,
  KHÔNG xác thực**, có thể bị người khác nghe/gửi gói giả. Không dùng
  cho cửa thật, không sử dụng topic dễ đoán, không chia sẻ topic.
- **Không có ảnh, ROI, vector embedding hay mật khẩu Wi-Fi nào gửi qua
  broker**. Chỉ gửi mã người dự đoán + cosine + số thứ tự ảnh. Những
  giá trị này vẫn là thông tin có nguồn gốc sinh trắc, do đó chỉ nên
  dùng dữ liệu thử nghiệm có sự đồng ý của các thành viên.
- Ảnh và ROI lưu cục bộ dưới `live_captures/`, đã gitignore.
- Nếu không muốn dùng broker công cộng, dùng Private IoT Gateway của
  Wokwi (tính năng trả phí) và broker MQTT cục bộ. Khi đó cần đổi
  `--broker` của Python và `BROKER` của sketch đồng bộ.
- Tất cả ảnh và model thật được xử lý trên PC; Wokwi không thấy JPEG.

Nguồn tham khảo:
- https://docs.wokwi.com/guides/esp32-wifi
- https://test.mosquitto.org/

## Kiểm thử đơn vị không cần model/dữ liệu sinh trắc

```powershell
python -m unittest discover -s tests -p "test_live_wokwi_protocol.py" -v
```

## Khi chạy thử xong

Chúng ta đã thử được luồng **ảnh mới → mô hình thật → OLED/servo/LED**
trên Wokwi, không phải phần **AI chạy trên ESP32-CAM**. Để hoàn thành
xử lý trên chip cần riêng một bước chuyển pipeline ROI, model và
runtime sang ESP32/ESP32-S3 và đo trên board.
