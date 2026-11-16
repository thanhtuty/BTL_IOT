# PalmLock — nạp firmware lên ESP32-CAM THẬT (kiểm tra trên board)

> **Điều kiện kỹ thuật bắt buộc:** Đây là firmware thực sự chụp ảnh
> OV2640 và chạy một thuật toán đặc trưng gradient trên ESP32; nó **KHÔNG
> chạy mô hình CompNet/PalmBridge (.pt) mà nhóm đã train**. Các checkpoint,
> thư mục `models/`, `experiments/`, `wokwi/` không bị thay đổi.
> Vì nhận diện sinh trắc bằng histogram gradient nhỏ chưa được kiểm định,
> **KHÔNG dùng firmware làm khóa an ninh thật**.
>
> Không có quyền truy cập cổng USB của bạn, nên chưa nạp hay chạy thử
> được trên phần cứng. File đã có trên GitHub để bạn mở bằng Arduino IDE.

## 1. Board áp dụng

- **AI-Thinker ESP32-CAM / OV2640**, chip classic ESP32-D0WDQ6-V3
- Flash **4 MB**, PSRAM **2 MB** (theo thông số đã kiểm tra trên board)
- Mã dùng pin map chuẩn của **AI Thinker ESP32-CAM**, không dùng cho
  ESP32-S3-EYE hoặc ESP32-S3-CAM có bố trí chân khác.
- **Không dùng microSD** vì GPIO13 và GPIO14 đã được sử dụng.

Nếu board của bạn không ghi AI Thinker ESP32-CAM hoặc chân camera
không giống bản chuẩn, **không nạp trước khi xác định pinout**.

## 2. Linh kiện và cách nối dây THỰC TẾ

| Linh kiện | Đấu nối |
|---|---|
| ESP32-CAM | Chân 5V và GND với nguồn 5V ổn định |
| Camera OV2640 | Cáp dẹt camera gắn trên board, không nối thêm GPIO |
| Servo SG90/MG90S (thử trên bàn) | Dây SIGNAL (cam/vàng) → **GPIO13 ESP32-CAM** |
| Servo V+ | **5V từ nguồn riêng** đủ dòng (không lấy từ GPIO) |
| Servo GND | GND nguồn servo → **GND chung ESP32-CAM** |
| Nút nhấn chụp | Một đầu → **GPIO14**, đầu còn lại → GND |
| USB-UART để nạp | TX adapter → U0R/RX0; RX adapter → U0T/TX0 |
| Nguồn & GND lập trình | Adapter GND → GND board; 5V nguồn phù hợp → 5V |

**Bắt buộc GND chung giữa camera, servo và nguồn.**
Không nối tín hiệu UART 5V vào RX0 3.3V của ESP32-CAM.
Nếu có GPIO0 → GND để vào flash mode, **tháo ra sau khi upload**,
bấm RST rồi mới chạy bình thường.
Nguồn servo nên riêng và đủ dòng vì servo lúc khởi động có thể làm
ESP32 brownout/reset; không dùng chốt khóa tải nặng ở bước thử.

**OLED và buzzer:** chưa đấu vào board ESP32-CAM trong firmware này;
nhiều GPIO đã bị OV2640 chiếm dụng, GPIO12/15 là chân boot strapping.
Không nối theo sơ đồ **DevKitC của Wokwi** lên ESP32-CAM.
Nếu nhóm có ESP32 DevKit thứ hai, giai đoạn tiếp theo có thể cho
ESP32-CAM gửi kết quả qua UART sang board điều khiển OLED/buzzer.

## 3. Cấu hình Arduino IDE

1. Cài **esp32 by Espressif Systems** trong Boards Manager.
2. Chọn **Tools → Board → AI Thinker ESP32-CAM**.
3. Chọn đúng **Tools → Port** (ví dụ COM4, phải xác định lại).
4. Flash Size **4 MB**; PSRAM **Enabled** nếu menu board có tùy chọn;
   Upload Speed trước tiên đặt **115200**.
5. Cài thêm thư viện **ESP32Servo** trong Library Manager.
6. Mở đúng **`esp32cam_onchip_bench.ino`** trong thư mục này.
   Thư viện `esp_camera.h`, `WebServer.h`, `Preferences.h`,
   `WiFi.h` có sẵn trong Arduino ESP32 core.
7. Kiểm tra và đổi **`AP_PASSWORD`** trong code nếu cần.
8. Bấm **Verify/Compile** rồi **Upload**.
9. Nếu dùng USB-TTL rời: giữ GPIO0 nối GND trong lúc upload, sau khi
   hoàn thành tháo GPIO0 khỏi GND và bấm Reset.

> Lỗi upload cũ `Invalid head of packet (0x65)` là vấn đề nạp/serial,
> không chứng minh code AI sai. Chọn 115200, cấp nguồn ổn định,
> giảm dây dài, và bảo đảm board thật vào flash mode đúng cách.
> Tránh kết luận thành công chỉ vì chương trình **Compile** xong.

## 4. Board chạy không cần Python, laptop chỉ dùng làm giao diện

Sau khi boot, Serial Monitor ở 115200 thường in:

```text
[BOARD] PSRAM: yes
[CAMERA] OV2640 grayscale QQVGA initialized
[AP] SSID: PalmLock-CAM-BENCH
[AP] Open http://192.168.4.1/
[READY] On-device camera + descriptor + 4 template storage.
```

Nếu Serial Monitor báo `[CAMERA] init FAILED: 0x...`,
hãy kiểm tra dây camera, camera pinout, nguồn và PSRAM. Khi đã chạy
đúng, có thể rút cáp lập trình, giữ nguồn ổn định và dùng điện thoại.

1. Kết nối điện thoại/laptop vào Wi-Fi do board tạo:
   - Tên Wi-Fi: **PalmLock-CAM-BENCH**
   - Mật khẩu mặc định của firmware: **PalmLock1234** (demo only)
   - Điện thoại có thể báo "Wi-Fi không có Internet": vẫn **giữ kết nối**
2. Mở **http://192.168.4.1/** trong trình duyệt.
3. Nhấn **Chụp ảnh mới từ OV2640**.
4. Căn lòng bàn tay vào **hình chữ nhật xanh** trên ảnh.
5. Với mỗi người 01–04, nhấn nút **Đăng ký người ...** ít nhất **5 lần**,
   khuyến nghị 8–12 ảnh độc lập. Mỗi lần là ảnh camera mới.
   Nên thay nhẹ khoảng cách và góc bàn tay nhưng giữ cả bốn người
   cùng cách đặt tay, không dùng hai tay trái/phải lẫn nhau.
6. Nhấn **Chụp ảnh mới + Nhận diện TRÊN BOARD**.
   ESP32 tự chụp grayscale 160×120, crop vùng giữa 96×96,
   trích 180 đặc trưng và so sánh cosine với bốn template trong flash.
   Trình duyệt nhận `prediction`, `scores`, `cosine`,
   `margin`, `latency_ms`, `matched`.
7. Nhấn **Thử servo 2 giây** để thử motor quay 90° → 0° (không phụ thuộc
   dự đoán AI). Nút nhấn GPIO14-GND thực hiện scan và in Serial.

**Các ảnh template lưu trong flash?** Firmware KHÔNG lưu ảnh gốc;
nó lưu **bốn vector đặc trưng** và số ảnh đăng ký trong NVS.
Đây vẫn là dữ liệu liên quan sinh trắc, cần quản lý riêng và có thể xóa
bằng nút **Xóa dữ liệu đăng ký**.

## 5. Bảo vệ mạch & hạn chế kỹ thuật

- **AUTO_UNLOCK_ON_MATCH = false** theo mặc định: nhận diện không tự
  mở khóa. Cách này tránh tự mở vì mô hình chưa được xác thực trên
  người lạ hoặc các điều kiện ánh sáng khác nhau.
- `COSINE_GATE=0.94`, `MARGIN_GATE=0.07` chỉ là **ngưỡng thử nghiệm
  ban đầu**, không phải FAR/FRR đã được đánh giá. Trạng thái `matched`
  không có nghĩa đó là người được xác thực an toàn.
- Trang HTTP chạy không có TLS/xác thực riêng ngoài mật khẩu Wi-Fi AP.
  Bất kỳ ai vào AP có thể đăng ký lại, test servo hoặc xóa template.
  **Chỉ dùng trên bàn thử**, không đấu tới cửa thật.
- ROI là vùng giữa cố định (không phải MediaPipe). Không có phát hiện
  bàn tay, chống ảnh giả, chống phát lại, phát hiện người lạ tin cậy.
  Độ chính xác có thể thấp, không dự đoán trước được.
- Board xử lý ở 160×120, vùng ROI 96×96 để phù hợp ESP32 cổ điển
  4MB Flash/2MB PSRAM. Không thể suy ra tốc độ/độ chính xác tương đương
  checkpoint ROI64.
- **Mô hình CompNet/PalmBridge gốc**: cần bước chuyển đổi riêng
  (.pt → định dạng được hỗ trợ → W8A8 quantization → xử lý custom Gabor,
  pooling và PalmBridge → đo peak RAM/flash → test embedding sai số).
  Khi bước này chưa xong, không được ghi trong báo cáo rằng board
  đang chạy mô hình đã huấn luyện.

## 6. Kết quả thử nghiệm phải ghi trong báo cáo

Ghi ảnh chụp màn hình và log:
- Boot info: camera, PSRAM, địa chỉ 192.168.4.1.
- Ảnh preview thật từ OV2640 + vùng ROI.
- Số ảnh enrollment Person 01..04.
- Ba lần test cho mỗi người và một người lạ.
- Kết quả 4 cosine, `margin`, thời gian xử lý.
- Video servo quay khi dùng nút **test servo** trên bàn.

Các kết quả chỉ chứng minh camera + tiền xử lý + so sánh ảnh **trên board**;
chưa chứng minh CompNet/PalmBridge inference và chưa đủ để cấp quyền
mở khóa cửa thật.
