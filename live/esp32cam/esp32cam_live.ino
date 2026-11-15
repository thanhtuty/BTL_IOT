// PalmLock-AI: REAL OV2640 JPEG capture via ESP32-CAM Wi-Fi HTTP.
// Target: AI-Thinker ESP32-CAM / ESP32-D0WDQ6, camera OV2640.
// Only capture runs here. The original MediaPipe + CompNet + PalmBridge
// run in live/live_web.py on the PC. This is NOT on-device AI inference.
//
// Flash with Arduino IDE using Board "AI Thinker ESP32-CAM".
// Edit WIFI_SSID / WIFI_PASSWORD below and NEVER commit your actual password.
#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"

// Replace before flashing; 2.4 GHz Wi-Fi, same LAN as your computer.
const char *WIFI_SSID = "REPLACE_WITH_WIFI_SSID";
const char *WIFI_PASSWORD = "REPLACE_WITH_WIFI_PASSWORD";

WebServer http(80);

// AI-Thinker ESP32-CAM GPIO mapping. Change if you own a different module.
static camera_config_t makeCameraConfig() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = 5;   c.pin_d1 = 18; c.pin_d2 = 19; c.pin_d3 = 21;
  c.pin_d4 = 36;  c.pin_d5 = 39; c.pin_d6 = 34; c.pin_d7 = 35;
  c.pin_xclk = 0; c.pin_pclk = 22;
  c.pin_vsync = 25; c.pin_href = 23;
  c.pin_sccb_sda = 26; c.pin_sccb_scl = 27;
  c.pin_pwdn = 32; c.pin_reset = -1;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = psramFound() ? FRAMESIZE_VGA : FRAMESIZE_QVGA;
  c.jpeg_quality = 12; // lower number = better quality
  c.fb_count = 1;
  c.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  c.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  return c;
}

void captureJpeg() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    http.send(503, "text/plain; charset=utf-8", "Camera capture failed");
    Serial.println("[CAMERA] Capture failed");
    return;
  }
  if (fb->format != PIXFORMAT_JPEG) {
    esp_camera_fb_return(fb);
    http.send(500, "text/plain", "Unexpected non-JPEG image");
    return;
  }
  // The response body is the ACTUAL camera frame acquired above.
  http.sendHeader("Cache-Control", "no-store, no-cache, must-revalidate");
  http.sendHeader("Content-Disposition", "inline; filename=palm_capture.jpg");
  http.setContentLength(fb->len);
  http.send(200, "image/jpeg", "");
  WiFiClient client = http.client();
  size_t sent = 0;
  uint32_t deadline = millis() + 5000;
  while (sent < fb->len && client.connected() && (int32_t)(deadline - millis()) > 0) {
    const size_t block = min((size_t)1024, fb->len - sent);
    size_t n = client.write(fb->buf + sent, block);
    if (n == 0) { delay(2); continue; }
    sent += n;
  }
  Serial.printf("[CAPTURE] %ux%u, JPEG %u bytes, sent %u bytes\n",
                fb->width, fb->height, (unsigned)fb->len, (unsigned)sent);
  esp_camera_fb_return(fb);
}

static const char INDEX_HTML[] PROGMEM = R"HTML(
<!doctype html><html lang="vi"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PalmLock - ESP32-CAM capture</title>
<style>body{font-family:Arial,sans-serif;max-width:720px;margin:25px auto;padding:12px}
img{width:100%;max-width:640px;background:#ddd;border-radius:12px}
button,a{padding:12px 18px;margin:8px 6px 10px 0;cursor:pointer}
small{color:#555}</style></head>
<body><h2>ESP32-CAM: chup anh THAT</h2>
<p>Moi lan bam <strong>Chup anh</strong> la mot khung JPEG moi tu camera OV2640.</p>
<button onclick="shoot()">Chup anh moi</button>
<a href="/capture" target="_blank">Mo anh JPG goc</a>
<p><img id="photo" alt="Anh camera se hien thi tai day"></p>
<small>Model PalmBridge chay tren PC qua live/live_web.py. Trang nay chi chup anh.</small>
<script>
function shoot(){document.getElementById("photo").src="/capture?t="+Date.now();}
</script></body></html>
)HTML";

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\nPalmLock ESP32-CAM LIVE CAPTURE (NO ON-BOARD AI)");
  Serial.printf("[BOARD] PSRAM: %s\n", psramFound() ? "yes" : "no");
  camera_config_t config = makeCameraConfig();
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[ERROR] esp_camera_init: 0x%x\n", (unsigned)err);
    Serial.println("Check AI-Thinker camera pinout, power 5V and board selection.");
    return;
  }
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("[WIFI] Connecting to 2.4GHz network");
  for (int i=0; i<40 && WiFi.status()!=WL_CONNECTED; i++) {
    delay(500);
    Serial.print(".");
  }
  if (WiFi.status()!=WL_CONNECTED) {
    Serial.println("\n[ERROR] Wi-Fi not connected. Check SSID/password/network.");
    return;
  }
  Serial.printf("\n[READY] Browser: http://%s/\n", WiFi.localIP().toString().c_str());
  Serial.printf("[READY] Direct JPEG: http://%s/capture\n", WiFi.localIP().toString().c_str());
  http.on("/", HTTP_GET, [](){ http.send(200, "text/html; charset=utf-8", INDEX_HTML); });
  http.on("/health", HTTP_GET, [](){ http.send(200, "text/plain", "PalmLock camera OK"); });
  http.on("/capture", HTTP_GET, captureJpeg);
  http.onNotFound([](){ http.send(404, "text/plain", "Not found"); });
  http.begin();
}

void loop() {
  if (WiFi.status()==WL_CONNECTED) http.handleClient();
  delay(2);
}
