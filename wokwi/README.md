# PalmLock-AI — Wokwi demo lắp mạch khóa cửa

**Mục tiêu:** trình diễn ESP32 điều khiển khóa servo, màn hình OLED,
còi buzzer, LED, cảm biến trạng thái cửa và ba nút bấm. Hệ thống phát
lại **điểm cosine đã tính trên PC**, không xử lý ảnh hoặc AI trên Wokwi.

Wokwi chỉ có **ESP32 DevKitC** ở sơ đồ này. Đây KHÔNG phải chân cắm
ESP32-CAM. Không được sao chép GPIO18/19/21/22/23 sang board camera
thật vì nhiều chân trên ESP32-CAM đã dùng cho OV2640.

## 1. Chạy nhanh trên trình duyệt

1. Mở https://wokwi.com/projects/new/esp32 .
2. Copy toàn bộ `wokwi/sketch.ino` vào tab `sketch.ino`.
3. Copy toàn bộ `wokwi/diagram.json` vào tab `diagram.json`.
4. Tạo file mới **demo_scores.h** trong dự án Wokwi và paste nội dung
   `wokwi/demo_scores.h`.
5. Mở **Library Manager** để thêm 3 thư viện trong `libraries.txt`
   (hoặc tạo tab `libraries.txt`):
   - ESP32Servo
   - Adafruit SSD1306
   - Adafruit GFX Library
6. Nhấn **Start Simulation** và mở Serial Monitor.
7. Mặc định công tắc cửa ở **bên trái = CLOSED**.

### Điều khiển

| Điều khiển | Thao tác | Kết quả |
|---|---|---|
| **NEXT** | Bấm nút xanh / phím N | Chọn ảnh trong danh sách replay |
| **SCAN** | Bấm nút xanh lá / phím S | Phát 4 cosine scores của một ảnh qua UART loopback |
| **TEST 95%** | Bấm nút vàng / phím T | Tạo tín hiệu **giả lập 0.95** để thử servo |
| **DOOR** | Công tắc trượt, trái/phải | Trái = CLOSED, phải = OPEN |
| **OLED** | Không cần thao tác | Hiển thị người dự đoán, điểm và trạng thái |
| **Buzzer** | Tự động | Âm cao khi mở; âm trầm khi từ chối |
| **LED đỏ/xanh** | Tự động | Đỏ = khóa; xanh = mở |
| **Servo** | Tự động | Góc 0° đóng, góc 90° mở; tự đóng sau 2,5 giây |

**Lưu ý:** tất cả điểm cosine thật trong bảng mẫu thấp hơn ngưỡng demo
0.85 nên phím SCAN sẽ DENIED. **Phím TEST chỉ để kiểm tra phần cứng,
KHÔNG phải AI nhận diện**. Nếu cửa ở vị trí OPEN, kể cả TEST cũng DENIED.

## 2. Danh sách linh kiện và chân nối TRÊN WOKWI (ESP32 DevKitC)

| Linh kiện | Chân thiết bị | Chân ESP32 DevKitC |
|---|---|---|
| OLED SSD1306 | SDA | GPIO21 |
| OLED SSD1306 | SCL | GPIO22 |
| OLED SSD1306 | VCC / GND | 5V / GND (chỉ Wokwi) |
| Servo khóa | PWM | GPIO18 |
| Servo khóa | V+ / GND | 5V / GND (chỉ Wokwi) |
| Buzzer | Pin 2 (+) | GPIO19 |
| Buzzer | Pin 1 (-) | GND |
| LED xanh | GPIO26 → R 220Ω → A | Cathode → GND |
| LED đỏ | GPIO27 → R 220Ω → A | Cathode → GND |
| NEXT | Một chân → GPIO32 | Chân đối diện → GND |
| SCAN | Một chân → GPIO33 | Chân đối diện → GND |
| TEST | Một chân → GPIO25 | Chân đối diện → GND |
| Công tắc DOOR SPDT | Chân 2 (common) | GPIO23, INPUT_PULLUP |
| Công tắc DOOR SPDT | Chân 1 | GND |
| Công tắc DOOR SPDT | Chân 3 | Bỏ trống |
| UART replay loopback | TX2 GPIO17 | RX2 GPIO16 |

Công tắc trượt mặc định trái sẽ nối chân 1 với chân 2,
GPIO23 = LOW = CLOSED. Trượt phải: chân 2 và 3 kết nối,
chân 3 bỏ trống, GPIO23 được kéo HIGH = OPEN.

### Test 4 tình huống

1. Giữ DOOR = CLOSED; bấm T. Servo quay tới 90°, LED xanh sáng,
   buzzer kêu, OLED hiển thị OPEN. Sau 2,5 giây, servo trở về 0°.
2. Gạt DOOR = OPEN; bấm T. Buzzer báo từ chối, servo không mở.
3. Gạt DOOR = CLOSED; bấm S. Hiển thị dự đoán từ điểm cosine thực
   nhưng kết quả bị từ chối bởi ngưỡng demo 0.85.
4. Bấm N đổi ảnh rồi bấm S; nhìn 4 điểm ở Serial Monitor. Có thể
   so sánh với đánh giá Compact64 trên PC.

## 3. Phát lại kết quả Compact 64×64 mới từ B5

Nếu đã chạy `python experiments/b5.py compare64`, trên máy Windows:

```powershell
python wokwi/export_scores.py --csv experiments/results/b5_compare_per_image.csv --source roi64
```

Lệnh tạo lại `wokwi/demo_scores.h` từ **32 vector cosine mới**.
Bạn phải **copy lại nội dung header này sang tab demo_scores.h trên Wokwi**
và chạy mô phỏng lại. Header trong repo mặc định chỉ có kết quả gốc
(13/32); export từ CSV Compact64 phải hiển thị **19/32** nếu dùng chính
32 ảnh trong báo cáo mà bạn vừa cung cấp.

Muốn đối chiếu bản Compact128 từ cùng CSV:

```powershell
python wokwi/export_scores.py --csv experiments/results/b5_compare_per_image.csv --source roi128
```

Lệnh trên ghi đè `wokwi/demo_scores.h` trên máy. Đừng vô ý commit
dữ liệu điểm số sinh trắc hoặc ảnh thật lên GitHub. `test_photos/` và
`experiments/results/` phải được giữ riêng tư. Xuất score không train model.

## 4. Mạch thật khác Wokwi

Nếu dùng **hai board thật**:
- ESP32-CAM AI Thinker: chụp ảnh, thử gửi bản tin UART.
- ESP32 DevKit: điều khiển OLED/servo/LED/buzzer/công tắc cửa.
- Phải nối **GND chung**.
- Mạch camera trong `firmware/esp32cam_score_replay/` sử dụng GPIO14 TX
  (microSD không dùng) → RX GPIO16 của ESP32 DevKit. **Bỏ dây loopback
  GPIO17→GPIO16** khi đấu 2 board thật. Đường UART trong ví dụ này chưa
  xác thực gói và chỉ dành cho thử nghiệm trên bàn.
- Servo thật cần **nguồn 5V riêng ổn định**, GND chung; không lấy dòng
  motor trực tiếp từ GPIO ESP32. Điện áp tín hiệu/nguồn OLED và buzzer
  phải theo đúng module thực tế.
- Khi mang `wokwi/sketch.ino` sang DevKit thật, cần nối reed switch
  hoặc công tắc tương ứng GPIO23→GND khi cửa đóng; nếu để GPIO23 nổi,
  INPUT_PULLUP đọc HIGH và hệ thống khóa không mở.

**Nếu chỉ có một ESP32-CAM:** không lắp thiết bị theo sơ đồ DevKitC
này vì pin map camera chiếm nhiều chân. Cần xác định đúng board và
thiết kế lại sơ đồ phần cứng hoặc sử dụng thêm ESP32 DevKit.

## 5. Giới hạn

- Mô phỏng trên Wokwi không có OV2640 thật; không chạy MediaPipe,
  CompNet hay PalmBridge trên vi điều khiển.
- Một dự án Wokwi không mô phỏng hai MCU nối dây với nhau
  (https://docs.wokwi.com/faq).
- Bản ghi cosine không phải xác suất xác thực. Ngưỡng 0.85 hiện chỉ
  là ngưỡng demo **chưa được hiệu chỉnh**.
- TEST 95% là công tắc thử nghiệm nguy hiểm nếu dùng cho cửa thật;
  cần bỏ hoàn toàn trước khi triển khai.
- Với lỗi Person01/Person04 và chưa kiểm thử người lạ, **không nên điều
  khiển khóa cửa thật theo dự đoán sinh trắc**.

README đầy đủ gốc: `README_WOKWI.md`.
