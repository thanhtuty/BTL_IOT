# PalmLock — Mẫu chạy thuật toán TRÊN ESP32 giả lập Wokwi

**Đây là mạch Wokwi độc lập; không cần chạy Python, webcam, Wi-Fi hay MQTT.**

Mục đích: **kiểm tra đường xử lý ảnh → thuật toán nhận dạng → logic mở khóa → servo/OLED/LED/buzzer** bằng chương trình C++ **chạy trên chip ESP32 mô phỏng**.

> **GIỚI HẠN QUAN TRỌNG**
>
> - Wokwi sử dụng **ESP32 DevKitC mô phỏng**, không có camera OV2640 ảo.
> - Dữ liệu ảnh trong bản Wokwi này là **khung grayscale 160×120 tổng hợp được tạo ngay trong ESP32**. Đây không phải ảnh chụp từ webcam, không phải ảnh lòng bàn tay thật.
> - Phần trích đặc trưng 180 chiều và so sánh cosine được xây dựng theo **cùng công thức gradient/cell brightness** trong firmware thực `firmware/esp32cam_onchip_bench/esp32cam_onchip_bench.ino`.
> - **KHÔNG chạy checkpoint CompNet + PalmBridge `.pt` đã train.** Bộ trích đặc trưng này chỉ để thử phần firmware và phần điện tử, **không đủ căn cứ sử dụng làm khóa bảo mật**.
> - Chế độ quay servo khi khớp mẫu tổng hợp chỉ được bật trong **bản Wokwi**; bản thử phần cứng thật giữ `AUTO_UNLOCK_ON_MATCH=false`.

## 1. Các file cần đưa lên Wokwi

| File GitHub | Tab Wokwi |
|---|---|
| [`wokwi_onchip/sketch.ino`](sketch.ino) | `sketch.ino` |
| [`wokwi_onchip/diagram.json`](diagram.json) | `diagram.json` |
| [`wokwi_onchip/libraries.txt`](libraries.txt) | `libraries.txt` |

**Không copy từ thư mục `wokwi/` (replay) hoặc `wokwi_live/` (MQTT + AI trên PC).**
Trong phiên bản này **không có** `demo_scores.h`, `WiFi.h`, `PubSubClient`, `live_wokwi.py`.

### Chạy

1. Mở **https://wokwi.com/projects/new/esp32**.
2. Mở tab `sketch.ino`, `Ctrl+A`, dán toàn bộ nội dung `wokwi_onchip/sketch.ino`.
3. Mở tab `diagram.json`, `Ctrl+A`, dán toàn bộ nội dung `wokwi_onchip/diagram.json`.
4. Mở **Library Manager** và thêm `ESP32Servo`, `Adafruit SSD1306`, `Adafruit GFX Library`; hoặc tạo tab `libraries.txt` với nội dung file tương ứng.
5. Nhấn **Start Simulation**. Không cần đợi Wi-Fi/MQTT.
6. Mở **Serial Monitor**, theo dõi dòng:

```text
=== PalmLock ON-CHIP WOKWI DEMO ===
NO MQTT, NO Python, NO live camera, NO trained CompNet/PalmBridge.
Image source is SYNTHETIC grayscale patterns in ESP32 RAM.
```

## 2. Điều khiển trực tiếp trên Wokwi

| Phím/nút | Tác dụng |
|---|---|
| **E — ENROLL** | ESP32 tạo 8 khung grayscale thử nghiệm cho mỗi trong 4 danh tính và **tự chạy thuật toán trích đặc trưng**, lưu centroid tổng vào NVS |
| **N — NEXT** | Chọn lần lượt **person_01 → person_02 → person_03 → person_04 → UNKNOWN** |
| **S — SCAN** | ESP32 tự tạo ảnh thử nghiệm MỚI của người đã chọn rồi chạy toàn bộ thuật toán cosine **ngay trên chip giả lập** |
| **T — SERVO** | Quay servo 90° trong 2 giây để kiểm tra cơ cấu chốt riêng, không phụ thuộc AI |
| **DOOR** | Gạt trái = CLOSED, phải = OPEN; khi OPEN thì chặn lệnh quay servo |

**Thử theo thứ tự:**

1. Để công tắc **DOOR = CLOSED**.
2. Nhấn **E** một lần; Serial Monitor in `[ENROLL] 4 x 8 synthetic frames`. Có thể chờ mô phỏng xử lý xong.
3. Chọn mặc định **person_01** rồi nhấn **S**; ESP32 sẽ in:
   - `[SIM CAMERA] source=person_01, frame=160x120...`
   - `[ON-ESP32 GRADIENT] pred=person_01 cos=... margin=... matched=YES ...`
   - `[BENCH LOCK] PULSE...` nếu pass ngưỡng mô phỏng và cửa đang CLOSED.
4. Nhấn **N**, rồi **S** để thử Person 02; lặp lại với 03 và 04.
5. Nhấn **N** đến **UNKNOWN**, rồi **S**. Điểm và margin sẽ được tính từ ảnh tổng hợp thứ năm; kỳ vọng nhánh *NO MATCH*. Đây **không phải** phép đo độ an toàn người lạ thật.
6. Gạt công tắc **DOOR = OPEN**, nhấn **S** hoặc **T** để kiểm tra motor không mở.
7. Bấm **T** để thử servo độc lập khi cửa CLOSED.

Sau khi servo quay 90°, chương trình tự đưa về 0° sau **2 giây**.

## 3. Phần nào đúng với firmware ESP32-CAM vật lý?

| Thành phần | Wokwi (`wokwi_onchip/`) | Board thật (`firmware/esp32cam_onchip_bench/`) |
|---|---|---|
| Bộ xử lý | ESP32 giả lập | ESP32-D0WDQ6-V3 vật lý |
| Nguồn ảnh | **Khung ảnh tổng hợp trong RAM** | **Ảnh grayscale OV2640 chụp thật** |
| Cỡ ảnh & ROI | 160×120, cắt giữa 96×96 | 160×120, cắt giữa 96×96 |
| Đặc trưng | Gradient 180 chiều trên ESP32 | Gradient 180 chiều trên ESP32 |
| So sánh | Cosine với mẫu đăng ký | Cosine với mẫu đăng ký |
| Mẫu 4 người | Tạo từ 4 họa tiết tổng hợp | Đăng ký từ ảnh chụp người thật |
| Camera OV2640 | Không có | Có |
| Servo | GPIO18 (DevKitC) | GPIO13 (ESP32-CAM, không microSD) |
| OLED/LED/buzzer | Có | Chưa đấu trên bản board thật |
| Tự quay servo theo kết quả | Có, **chỉ để demo giả lập** | Tắt theo mặc định |
| Model CompNet + PalmBridge | **Chưa có** | **Chưa có** |

**Không dùng pinout Wokwi DevKitC để đấu vào ESP32-CAM thật**: nhiều GPIO trên module camera đã bị OV2640 chiếm dụng.

## 4. Vì sao cần bước này?

Bản Wokwi kiểm tra được:
- firmware C++ ESP32 chạy được luồng xử lý đặc trưng;
- lưu và đọc 4 bộ mẫu nhận diện;
- xếp hạng 4 điểm cosine và quyết định ngưỡng;
- màn hình, tín hiệu cảnh báo, servo và cảm biến trạng thái cửa.

Nó **không kiểm tra** khả năng chụp ảnh OV2640 thật, độ chính xác sinh trắc trên ảnh thật hay tốc độ chạy của CompNet/PalmBridge. Những điều đó vẫn phải đánh giá trên module thật sau khi chuẩn bị firmware tương thích.

## 5. Tài liệu tiếp theo

- Board thật: [hướng dẫn ESP32-CAM](../firmware/esp32cam_onchip_bench/README.md)
- Chương trình chụp ảnh OV2640 thật + AI trên PC (tùy chọn thử tích hợp): [README_REAL_WOKWI.md](../README_REAL_WOKWI.md)
