/*
 PalmLock — ESP32-CAM REAL ON-DEVICE BENCH PROTOTYPE
 Board: AI Thinker ESP32-CAM with OV2640; tested chip family: classic ESP32.
 This sketch captures REAL GRAYSCALE frames and computes an inexpensive
 handcrafted gradient descriptor ON the ESP32. Four reference templates
 are enrolled using the web UI and stored in ESP32 NVS.

 IMPORTANT:
 - This DOES NOT run the trained CompNet/PalmBridge checkpoint.
 - NOT a validated biometric identity verification algorithm.
 - Keep servo connected to a demonstration latch, NEVER an occupied door.
 - AUTO_UNLOCK_ON_MATCH is deliberately OFF. Test servo with test button.
 - The Wi-Fi AP / HTTP endpoints are for a closed BENCH, not deployment.
 - Disable microSD: GPIO13 (servo) and GPIO14 (button) are used.
*/
#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <Preferences.h>
#include <ESP32Servo.h>
#include <esp_camera.h>
#include <math.h>
#include <string.h>

static const char *AP_SSID = "PalmLock-CAM-BENCH";
// Change this before anyone outside your team uses the lab hardware.
static const char *AP_PASSWORD = "PalmLock1234";

// Servo external 5V supply, with common GND with camera board.
// GPIO13/GPIO14 are unused if microSD is disabled.
static const int PIN_SERVO = 13;
static const int PIN_SCAN_BUTTON = 14; // momentary button -> GND
static const int SERVO_LOCKED_DEG = 0;
static const int SERVO_UNLOCKED_DEG = 90;
static const unsigned long MOTOR_PULSE_MS = 2000;
static const bool AUTO_UNLOCK_ON_MATCH = false; // SECURITY: keep false

// Fixed centered crop is NOT MediaPipe; frame: 160x120, crop: 96x96.
// The preview page overlays the ROI and helps align a single palm.
static const int FRAME_W = 160, FRAME_H = 120;
static const int ROI_X = 32, ROI_Y = 12, ROI_SIZE = 96;
static const int CELLS = 6, BINS = 5;
static const int FEAT_DIM = CELLS * CELLS * BINS; // 180
static const uint32_t TEMPLATE_MAGIC = 0x504C3031u; // "PL01"
static const uint16_t MAX_ENROLL = 32;
static const int MIN_ENROLL_FOR_SCAN = 5;
static const float COSINE_GATE = 0.94f; // conservative starting point, NOT calibrated
static const float MARGIN_GATE = 0.07f; // nearest - second nearest, NOT calibrated

struct Template {
  uint32_t magic;
  uint16_t count;
  uint16_t dim;
  float sum[FEAT_DIM]; // Sum of normalized descriptors
};

static Template people[4] = {};
WebServer http(80);
Preferences store;
Servo servo;
bool cameraOK = false;
bool pulseActive = false;
unsigned long pulseStarted = 0;
bool buttonWasDown = false;
unsigned long buttonAt = 0;
unsigned long scanNo = 0;

static bool hasTemplate(int i) {
  return i >= 0 && i < 4 &&
         people[i].magic == TEMPLATE_MAGIC &&
         people[i].dim == FEAT_DIM &&
         people[i].count >= MIN_ENROLL_FOR_SCAN;
}

static void motorClose() {
  servo.write(SERVO_LOCKED_DEG);
  pulseActive = false;
}
static void motorBenchPulse() {
  servo.write(SERVO_UNLOCKED_DEG);
  pulseActive = true;
  pulseStarted = millis();
  Serial.println("[BENCH SERVO] 90 degrees, returning to 0 in 2 seconds");
}

static void loadTemplates() {
  store.begin("palmlock", false);
  for (int i=0; i<4; i++) {
    char key[8]; snprintf(key, sizeof(key), "p%d", i+1);
    Template temp = {};
    if (store.getBytesLength(key) == sizeof(Template) &&
        store.getBytes(key, &temp, sizeof(temp)) == sizeof(temp) &&
        temp.magic == TEMPLATE_MAGIC && temp.dim == FEAT_DIM &&
        temp.count > 0 && temp.count <= MAX_ENROLL) {
      people[i] = temp;
    } else {
      memset(&people[i], 0, sizeof(Template));
      people[i].magic = TEMPLATE_MAGIC;
      people[i].dim = FEAT_DIM;
    }
    Serial.printf("[NVS] person_%02d enrolled samples=%u\n", i+1, people[i].count);
  }
}

static bool saveTemplate(int id) {
  char key[8]; snprintf(key, sizeof(key), "p%d", id+1);
  return store.putBytes(key, &people[id], sizeof(Template)) == sizeof(Template);
}

static bool cameraInit() {
  camera_config_t c = {};
  // Camera XCLK uses timer3/channel7 to avoid timer0/channel0 used by servo.
  c.ledc_timer = LEDC_TIMER_3;
  c.ledc_channel = LEDC_CHANNEL_7;
  c.pin_d0 = 5; c.pin_d1 = 18; c.pin_d2 = 19; c.pin_d3 = 21;
  c.pin_d4 = 36; c.pin_d5 = 39; c.pin_d6 = 34; c.pin_d7 = 35;
  c.pin_xclk = 0; c.pin_pclk = 22;
  c.pin_vsync = 25; c.pin_href = 23;
  c.pin_sccb_sda = 26; c.pin_sccb_scl = 27;
  c.pin_pwdn = 32; c.pin_reset = -1;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_GRAYSCALE;
  c.frame_size = FRAMESIZE_QQVGA; // 160 x 120; fits small DRAM buffers
  c.jpeg_quality = 12;
  c.fb_count = 1;
  c.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  c.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  esp_err_t error = esp_camera_init(&c);
  if (error != ESP_OK) {
    Serial.printf("[CAMERA] init FAILED: 0x%x\n", (unsigned)error);
    return false;
  }
  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor) {
    // Keep camera configuration stable across enrollment and verification.
    sensor->set_brightness(sensor, 0);
    sensor->set_contrast(sensor, 0);
  }
  Serial.println("[CAMERA] OV2640 grayscale QQVGA initialized");
  return true;
}

// Gradient orientation histogram + relative cell luminance.
// This lightweight method is different from CompNet+PalmBridge.
// It does not perform hand detection. Palm MUST fill fixed ROI consistently.
static bool makeDescriptor(const camera_fb_t *fb, float out[FEAT_DIM]) {
  if (!fb || fb->format != PIXFORMAT_GRAYSCALE ||
      fb->width != FRAME_W || fb->height != FRAME_H ||
      fb->len < (size_t)(FRAME_W * FRAME_H)) return false;
  memset(out, 0, sizeof(float) * FEAT_DIM);
  const uint8_t *p = fb->buf;
  float mean = 0;
  for (int y = ROI_Y; y < ROI_Y + ROI_SIZE; y++)
    for (int x = ROI_X; x < ROI_X + ROI_SIZE; x++)
      mean += p[y*FRAME_W + x];
  mean /= (ROI_SIZE*ROI_SIZE);
  const int cellW = ROI_SIZE / CELLS;

  for (int gy = 0; gy < CELLS; gy++) {
    for (int gx = 0; gx < CELLS; gx++) {
      int base = (gy * CELLS + gx) * BINS;
      float lum = 0;
      for (int dy = 1; dy < cellW - 1; dy++) {
        for (int dx = 1; dx < cellW - 1; dx++) {
          int x = ROI_X + gx * cellW + dx;
          int y = ROI_Y + gy * cellW + dy;
          int ix = y * FRAME_W + x;
          int dX = (int)p[ix+1] - (int)p[ix-1];
          int dY = (int)p[ix+FRAME_W] - (int)p[ix-FRAME_W];
          int aX = abs(dX), aY = abs(dY), bin = 0;
          if (aX > 2*aY) bin = 0; // vertical texture / x-gradient
          else if (aY > 2*aX) bin = 1; // horizontal texture / y-gradient
          else bin = ((dX>=0) == (dY>=0)) ? 2 : 3;
          out[base+bin] += (float)(aX + aY);
          lum += p[ix];
        }
      }
      const float n = (float)((cellW-2)*(cellW-2));
      for (int b=0; b<4; b++) out[base+b] /= n;
      out[base+4] = ((lum / n) - mean) * 0.2f;
    }
  }

  double norm2 = 0;
  for (int i=0; i<FEAT_DIM; i++) norm2 += (double)out[i]*out[i];
  if (norm2 < 1e-7) return false;
  float inv = 1.0f / sqrtf((float)norm2);
  for (int i=0; i<FEAT_DIM; i++) out[i] *= inv;
  return true;
}

static float templateCosine(const Template &t, const float f[FEAT_DIM]) {
  double dot=0, mag2=0;
  for (int i=0; i<FEAT_DIM; i++) {
    dot += (double)t.sum[i]*f[i];
    mag2 += (double)t.sum[i]*t.sum[i];
  }
  if (mag2 < 1e-9) return -1.0f;
  return (float)(dot / sqrt(mag2));
}

static String reject(const char *message) {
  return String("{\"ok\":false,\"message\":\"") + message + "\"}";
}

// Only use feature vectors from NEW captures. No downloaded/test photo replay.
static String enroll(int id) {
  if (id < 0 || id >= 4) return reject("Bad person ID");
  if (!cameraOK) return reject("Camera not initialized");
  if (people[id].count >= MAX_ENROLL) return reject("Enrollment limit 32 reached");
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return reject("Camera capture failed");
  float f[FEAT_DIM];
  bool ok = makeDescriptor(fb, f);
  esp_camera_fb_return(fb);
  if (!ok) return reject("Grayscale frame/ROI invalid");
  Template prior = people[id];
  for (int i=0; i<FEAT_DIM; i++) people[id].sum[i] += f[i];
  people[id].count++;
  if (!saveTemplate(id)) {
    people[id] = prior; // restore in-memory copy on persistence failure
    return reject("NVS write failed");
  }
  Serial.printf("[ENROLL] person_%02d: %u images\n", id+1, people[id].count);
  return String("{\"ok\":true,\"person\":") + String(id+1) +
         ",\"samples\":" + people[id].count +
         ",\"message\":\"Frame enrolled locally in NVS\"}";
}

static String scanPalm() {
  scanNo++;
  if (!cameraOK) return reject("Camera not initialized");
  uint32_t startMs = millis(); // measure capture + feature + comparison
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return reject("Camera capture failed");
  float f[FEAT_DIM];
  bool ok = makeDescriptor(fb, f);
  esp_camera_fb_return(fb);
  if (!ok) return reject("Grayscale frame/ROI invalid");

  float scores[4];
  float best = -2, second = -2;
  int bestUser = -1, countValid = 0;
  for (int i=0; i<4; i++) {
    scores[i] = hasTemplate(i) ? templateCosine(people[i], f) : -1.0f;
    if (!hasTemplate(i)) continue;
    countValid++;
    if (scores[i] > best) {
      second = best;
      best = scores[i];
      bestUser = i;
    } else if (scores[i] > second) {
      second = scores[i];
    }
  }
  if (countValid < 4) return reject("Enroll >=5 images for EACH of 4 people first");

  float margin = best - second;
  bool matched = best >= COSINE_GATE && margin >= MARGIN_GATE;
  Serial.printf("[ON-CHIP] scan=%lu best=person_%02d cos=%.4f margin=%.4f matched=%s time=%lums\n",
                scanNo, bestUser+1, best, margin, matched?"yes":"no",
                (unsigned long)(millis()-startMs));
  if (matched && AUTO_UNLOCK_ON_MATCH && !pulseActive) {
    motorBenchPulse(); // disabled by default; lab bench only
  }

  String out = "{\"ok\":true,\"prediction\":";
  out += String(bestUser+1);
  out += ",\"matched\":";
  out += matched ? "true" : "false";
  out += ",\"auto_unlock_enabled\":";
  out += AUTO_UNLOCK_ON_MATCH ? "true" : "false";
  out += ",\"cosine\":";
  out += String(best, 5);
  out += ",\"margin\":";
  out += String(margin, 5);
  out += ",\"latency_ms\":";
  out += String((unsigned long)(millis()-startMs));
  out += ",\"scores\":[";
  for (int i=0; i<4; i++) {
    if (i) out += ",";
    out += String(scores[i],5);
  }
  out += "],\"message\":\"Real OV2640 frame; descriptor matched ON ESP32 (not CompNet)\"}";
  return out;
}

static void putLE16(uint8_t *dst, unsigned v) {
  dst[0] = v & 255; dst[1] = (v>>8) & 255;
}
static void putLE32(uint8_t *dst, unsigned long v) {
  for (int i=0; i<4; i++) dst[i] = (v>>(8*i)) & 255;
}
static void sendPreview() {
  if (!cameraOK) { http.send(503, "text/plain", "Camera error"); return; }
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb || fb->format != PIXFORMAT_GRAYSCALE ||
      fb->width != FRAME_W || fb->height != FRAME_H) {
    if (fb) esp_camera_fb_return(fb);
    http.send(503, "text/plain", "Camera capture error");
    return;
  }
  const uint32_t imageBytes = FRAME_W*FRAME_H*3; // 24-bit BMP, stride aligned to 4
  uint8_t header[54] = {};
  header[0] = 'B'; header[1] = 'M';
  putLE32(header+2, 54+imageBytes); putLE32(header+10, 54);
  putLE32(header+14, 40); putLE32(header+18, FRAME_W);
  putLE32(header+22, FRAME_H); putLE16(header+26, 1);
  putLE16(header+28, 24); putLE32(header+34, imageBytes);
  http.setContentLength(54+imageBytes);
  http.send(200, "image/bmp", "");
  WiFiClient client = http.client();
  client.write(header, sizeof(header));
  uint8_t row[FRAME_W*3];
  for (int y=FRAME_H-1; y>=0; y--) {
    for (int x=0; x<FRAME_W; x++) {
      uint8_t c = fb->buf[y*FRAME_W+x];
      row[x*3] = c; row[x*3+1] = c; row[x*3+2] = c;
    }
    if (client.write(row, sizeof(row)) != sizeof(row)) break;
    yield();
  }
  esp_camera_fb_return(fb);
}

static const char HOME[] PROGMEM = R"HTML(
<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PalmLock ESP32-CAM — chip-only bench</title>
<style>
body{font:16px system-ui,Arial,sans-serif;background:#101827;color:#f5f7fa;margin:0}
main{max-width:760px;margin:auto;padding:16px}
section{background:#1d2b40;border:1px solid #375070;border-radius:12px;margin-top:14px;padding:16px}
h1{font-size:24px}p,small{color:#bac8dc}
button{font:inherit;background:#336dd5;color:white;border:0;padding:10px 12px;margin:5px;border-radius:7px}
button:disabled{opacity:.5}.alert{color:#ffc27a}
.preview{position:relative;display:inline-block;width:100%;max-width:640px}
.preview img{width:100%;display:block;image-rendering:pixelated}
.roi{position:absolute;left:20%;top:10%;width:60%;height:80%;border:3px solid #18d6ab;box-sizing:border-box;pointer-events:none}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#111b29;padding:12px;border-radius:8px}
</style></head><body><main>
<h1>PalmLock: camera + feature matching TRÊN ESP32-CAM</h1>
<p class="alert">BẢN THỬ NGHIỆM: xử lý gradient nhẹ trên board, KHÔNG PHẢI mô hình CompNet/PalmBridge đã train. Không dùng để khóa cửa thật.</p>
<section><h3>1. Căn giữa lòng bàn tay</h3>
<p>Giữ một bàn tay phẳng, cùng chiều, vừa khít hình chữ nhật xanh. Chụp nhiều lần ở tư thế hơi khác nhau.</p>
<div class="preview"><img id="pic" alt="Bấm Chụp để xem camera"><div class="roi"></div></div>
<p><button onclick="preview()">Chụp ảnh mới từ OV2640</button></p></section>
<section><h3>2. Đăng ký 4 người (chỉ lần đầu)</h3>
<p>Đăng ký mỗi người ít nhất 5 ảnh thật, tốt hơn 8–12 ảnh. Hệ thống chỉ lưu vector đặc trưng trong flash NVS.</p>
<div><button onclick="enroll(1)">Đăng ký người 01</button><button onclick="enroll(2)">Đăng ký người 02</button>
<button onclick="enroll(3)">Đăng ký người 03</button><button onclick="enroll(4)">Đăng ký người 04</button></div>
<p id="counts"></p></section>
<section><h3>3. Nhận diện và thử servo</h3>
<button onclick="scan()">Chụp ảnh mới + Nhận diện TRÊN BOARD</button>
<button onclick="servo()">Thử servo 2 giây (không phụ thuộc AI)</button>
<small>Vì độ chính xác chưa được xác thực, servo KHÔNG tự mở theo dự đoán. Nút thử servo chỉ dành cho mạch mẫu trên bàn.</small></section>
<section><h3>Kết quả thực tế</h3><pre id="log">Sẵn sàng...</pre>
<button onclick="reset()">Xóa dữ liệu đăng ký (NVS)</button></section>
<script>
const log=document.getElementById('log');
async function call(url){
 try{
  const res=await fetch(url,{method:'POST'});
  const d=await res.json();
  log.textContent=JSON.stringify(d,null,2);
  await status();
 }catch(e){log.textContent='Lỗi: '+e.message}
}
function preview(){document.getElementById('pic').src='/preview.bmp?t='+Date.now()}
function enroll(i){call('/api/enroll?person='+i)}
function scan(){call('/api/scan')}
function servo(){call('/api/test-servo')}
function reset(){if(confirm('Xóa toàn bộ 4 template?')&&confirm('Xác nhận xóa vĩnh viễn?'))call('/api/reset')}
async function status(){
 try{let d=await (await fetch('/api/status')).json();
 document.getElementById('counts').textContent='Ảnh đã đăng ký: '+d.samples.join(' / ')+' (người 01 → 04)';
 }catch(e){document.getElementById('counts').textContent='Chưa lấy được trạng thái.'}
}
status();
</script></main></body></html>
)HTML";

static void jsonResponse(const String &value, int code=200) {
  http.sendHeader("Cache-Control", "no-store");
  http.send(code, "application/json; charset=utf-8", value);
}

static void setupWeb() {
  http.on("/", HTTP_GET, [](){http.send_P(200,"text/html; charset=utf-8",HOME);});
  http.on("/preview.bmp", HTTP_GET, sendPreview);
  http.on("/api/status", HTTP_GET, [](){
    String data = "{\"camera_ok\":";
    data += cameraOK ? "true" : "false";
    data += ",\"psram\":";
    data += psramFound() ? "true" : "false";
    data += ",\"auto_unlock_enabled\":";
    data += AUTO_UNLOCK_ON_MATCH ? "true" : "false";
    data += ",\"samples\":[";
    for (int i=0; i<4; i++) {if (i) data += ","; data += String(people[i].count);}
    data += "]}";
    jsonResponse(data);
  });
  http.on("/api/enroll", HTTP_POST, [](){
    if (!http.hasArg("person")) {jsonResponse(reject("Missing person"),400);return;}
    String s = http.arg("person");
    if (s.length()!=1 || s[0]<'1' || s[0]>'4') {
      jsonResponse(reject("Person must be 1..4"),400); return;
    }
    jsonResponse(enroll(s[0]-'1'));
  });
  http.on("/api/scan", HTTP_POST, [](){jsonResponse(scanPalm());});
  http.on("/api/test-servo", HTTP_POST, [](){
    motorBenchPulse();
    jsonResponse("{\"ok\":true,\"message\":\"Bench servo pulse 2 seconds (no AI grant)\"}");
  });
  http.on("/api/reset", HTTP_POST, [](){
    // Bench only: this endpoint deliberately has no user authentication.
    for (int i=0; i<4; i++) {
      char key[8]; snprintf(key,sizeof(key),"p%d",i+1);
      store.remove(key);
      memset(&people[i],0,sizeof(Template));
      people[i].magic=TEMPLATE_MAGIC; people[i].dim=FEAT_DIM;
    }
    motorClose();
    Serial.println("[NVS] All enrollment templates removed");
    jsonResponse("{\"ok\":true,\"message\":\"All four templates cleared\"}");
  });
  http.onNotFound([](){http.send(404,"text/plain","Not found");});
  http.begin();
}

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("\nPalmLock PHYSICAL ESP32-CAM on-chip feature benchmark");
  Serial.println("This is NOT trained CompNet/PalmBridge. DO NOT use for real security.");
  Serial.printf("[BOARD] PSRAM: %s, free heap: %lu\n",
                psramFound()?"yes":"no",(unsigned long)ESP.getFreeHeap());
  loadTemplates();
  cameraOK = cameraInit();

  // Attach servo AFTER camera to keep camera XCLK LEDC distinct.
  pinMode(PIN_SCAN_BUTTON,INPUT_PULLUP);
  servo.setPeriodHertz(50);
  servo.attach(PIN_SERVO, 500, 2400);
  motorClose();

  WiFi.mode(WIFI_AP);
  if (!WiFi.softAP(AP_SSID,AP_PASSWORD)) {
    Serial.println("[AP] FAIL to start WiFi SoftAP");
  } else {
    Serial.printf("[AP] SSID: %s\n", AP_SSID);
    Serial.printf("[AP] Open http://%s/ (password in sketch, BENCH ONLY)\n",
                  WiFi.softAPIP().toString().c_str());
  }
  setupWeb();
  Serial.println("[READY] On-device camera + descriptor + 4 template storage.");
  Serial.println("[BUTTON] GPIO14 to GND = capture and scan; status in Serial.");
}

void loop() {
  http.handleClient();
  bool down = digitalRead(PIN_SCAN_BUTTON)==LOW;
  if (down && !buttonWasDown && millis()-buttonAt>250) {
    buttonAt=millis();
    String result=scanPalm();
    Serial.printf("[BUTTON SCAN] %s\n", result.c_str());
  }
  buttonWasDown=down;
  if (pulseActive && millis()-pulseStarted >= MOTOR_PULSE_MS) motorClose();
  delay(3);
}
