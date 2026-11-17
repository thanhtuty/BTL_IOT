# Reference ESP32 runtime — PalmLock V2

**Board thực tế theo esptool của nhóm:** ESP32-D0WDQ6-V3 (ESP32 thường), 4 MB Flash, 2 MB PSRAM; không giả định các kernel ESP32-S3 được hỗ trợ.

Sau khi train/enroll/calibrate và chạy `python -m palm_v2.cli export`:

1. Copy `PalmLiteV2.h`, `PalmLiteV2.cpp`, `generated_palm_v2.h` vào sketch Arduino **riêng** (không thay thế Result.ino V1 trên main).
2. Chụp/cắt chính xác vùng ROI grayscale 64×64, độ sáng và vị trí tay tương đương bộ dữ liệu train. Runtime **không tự dò tay, không tự crop, không đọc JPEG trực tiếp**.
3. Trong setup gọi `palmLiteBegin()` để cấp ~288 KB bộ đệm PSRAM, bắt lỗi nếu trả về false.
4. Khi cần nhận diện: `PalmLitePrediction pred; bool ok = palmLitePredict(roi64, pred);`.
5. Chỉ xét `pred.person` nếu `ok && pred.valid && pred.accepted`. Kết quả **không cấp quyền tự động**; còn cần kiểm tra trạng thái khóa/quyền ở ứng dụng.
6. Ghi log latency, free PSRAM, cosine và đối chiếu output của cùng ROI với Python. Không nối relay thật trước khi hoàn tất test.

```cpp
#include "PalmLiteV2.h"

void setup() {
  Serial.begin(115200);
  if (!palmLiteBegin()) {
    Serial.println("Cannot allocate PSRAM for Palm V2");
  }
}

void testOneRoi(const uint8_t roi64[4096]) {
  PalmLitePrediction result;
  if (palmLitePredict(roi64, result)) {
    Serial.printf("candidate=%d person=%d score=%.4f time=%lu ms\\n",
                  result.candidate, result.person, result.best, result.latencyMs);
  }
}
```

**Hạn chế:** Export hiện tại là float32, chưa lượng tử hóa W8A8, tốc độ trên ESP32 thường cần benchmark. Gabor cố định 4 hướng và CNN compact, chưa có chống giả mạo ảnh in/màn hình. Camera + Web + khóa thực tế chưa tích hợp trong repo này. Nếu muốn so sánh V1/V2, cần chạy trên cùng ROI và board. Đừng nhúng song song hai bộ model trong 4 MB Flash.
