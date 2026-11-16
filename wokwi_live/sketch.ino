// PalmLock REAL CAPTURE BENCH -> WOKWI ACTUATOR SIMULATOR
// Camera and MediaPipe + CompNet + PalmBridge run on the PC, NOT on ESP32.
// The ESP32 RECEIVES fresh AI predictions over MQTT and drives virtual hardware.
//
// IMPORTANT: simulation-only firmware. NEVER flash this bench sketch on a
// real door controller: public MQTT has NO authentication and the servo
// pulse below may be triggered by any AI result, even WRONG identity.
#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ESP32Servo.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// Set this to the same --topic used by python live/live_wokwi.py
static const char *TOPIC = "palmlock/demo/REPLACE_WITH_SAME_RANDOM_ID";
static const char *BROKER = "test.mosquitto.org";
static const uint16_t MQTT_PORT = 1883;

// For demonstrating motor operation on Wokwi ONLY. This deliberately
// moves the motor for any fresh model result. NOT an authentication grant.
// Turn false to apply the 0.85 score gate. Scores from the current model
// are mostly lower than 0.85, so the motor normally stays still.
static const bool BENCH_PULSE_EVERY_RESULT = true;
static const int AUTH_GATE_SCORE_X10000 = 8500;
static const unsigned long AUTO_CLOSE_MS = 2500;

static const int PIN_SERVO = 18;
static const int PIN_GREEN = 26;
static const int PIN_RED = 27;
static const int PIN_BUZZER = 19;
static const int PIN_DOOR = 23;  // LOW: door closed, HIGH: door open

WiFiClient network;
PubSubClient mqtt(network);
Adafruit_SSD1306 oled(128, 64, &Wire, -1);
Servo motor;
bool oledReady = false;
bool motorPulsing = false;
bool lastDoorOpen = false;
unsigned long startedPulse = 0;
unsigned long lastConnectAttempt = 0;
unsigned long wifiStartedAt = 0;
unsigned long lastWiFiLogAt = 0;
int previousWiFiStatus = -1;
static const unsigned long WIFI_RETRY_MS = 30000;
static const unsigned long WIFI_LOG_MS = 5000;

bool doorOpen() {
  return digitalRead(PIN_DOOR) == HIGH;
}

void screen(const char* a, const char* b, const char* c, const char* d) {
  if (!oledReady) return;
  oled.clearDisplay();
  oled.setTextSize(1);
  oled.setTextColor(SSD1306_WHITE);
  oled.setCursor(0, 0); oled.println("PALMLOCK REAL-AI BENCH");
  oled.drawLine(0, 11, 127, 11, SSD1306_WHITE);
  oled.setCursor(0, 16); oled.println(a);
  oled.setCursor(0, 28); oled.println(b);
  oled.setCursor(0, 40); oled.println(c);
  oled.setCursor(0, 52); oled.println(d);
  oled.display();
}

void lockClosed() {
  motor.write(0);
  digitalWrite(PIN_GREEN, LOW);
  digitalWrite(PIN_RED, HIGH);
  motorPulsing = false;
}

void benchMotorPulse() {
  motor.write(90);
  digitalWrite(PIN_GREEN, HIGH);
  digitalWrite(PIN_RED, LOW);
  tone(PIN_BUZZER, 1700, 130);
  motorPulsing = true;
  startedPulse = millis();
}

void deny(const char *reason) {
  lockClosed();
  tone(PIN_BUZZER, 540, 180);
  Serial.printf("[BLOCKED] %s\n", reason);
  screen("DOOR LOCKED", reason, "No servo pulse", "BENCH MQTT ONLY");
}

void onMessage(char *topic, byte *payload, unsigned int len) {
  if (strcmp(topic, TOPIC) != 0 || len == 0 || len >= 72) {
    deny("Bad topic / length");
    return;
  }
  char text[72];
  memcpy(text, payload, len);
  text[len] = '\0';

  int person = 0, score = 0;
  unsigned long seq = 0;
  char extra = 0;
  if (sscanf(text, "P,%d,%d,%lu%c", &person, &score, &seq, &extra) != 3 ||
      person < 1 || person > 4 || score < -10000 || score > 10000 ||
      seq == 0) {
    deny("Invalid AI packet");
    return;
  }

  Serial.printf("[REAL PC AI] image seq=%lu predicted person_%02d, cosine=%.4f\n",
                seq, person, score / 10000.0f);
  if (doorOpen()) {
    deny("DOOR OPEN");
    return;
  }

  bool passed = score >= AUTH_GATE_SCORE_X10000;
  char who[26], conf[26];
  snprintf(who, sizeof(who), "Pred: person_%02d", person);
  snprintf(conf, sizeof(conf), "Cosine: %.4f", score / 10000.0f);

  if (BENCH_PULSE_EVERY_RESULT) {
    // REAL inference result, but NOT valid authorization!
    Serial.printf("[BENCH ACTUATOR] pulse; biometric gate=%s -- NOT AUTHORIZED\n",
                  passed ? "score>=0.85" : "DENIED");
    benchMotorPulse();
    screen(who, conf, passed ? "Gate score: PASS" : "Gate score: FAIL",
           "SERVO BENCH TEST");
  } else if (passed) {
    Serial.println("[DEMO GATE] score >= 0.85 (NOT validated)");
    benchMotorPulse();
    screen(who, conf, "Demo gate: PASS", "MOTOR PULSE 2.5s");
  } else {
    Serial.println("[DEMO GATE] score <0.85: no motor pulse");
    deny("Score below 0.85");
    screen(who, conf, "Demo gate: DENY", "LOCKED");
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_GREEN, OUTPUT);
  pinMode(PIN_RED, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_DOOR, INPUT_PULLUP);
  Wire.begin(21, 22);
  oledReady = oled.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  motor.setPeriodHertz(50);
  motor.attach(PIN_SERVO, 500, 2400);
  lockClosed();
  lastDoorOpen = doorOpen();

  screen("Connecting WiFi...", "Wokwi-GUEST", "PC AI -> MQTT", "No camera on chip");
  Serial.println("PalmLock REAL PHOTO + REAL AI (on PC) -> virtual ESP32 hardware");
  Serial.println("CHECK: change TOPIC to the same random value as Python --topic.");
  Serial.println("WARNING: open public MQTT! BENCH ONLY; DO NOT FLASH TO REAL LOCK.");
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  Serial.println("[WIFI] Connecting to Wokwi-GUEST, channel 6...");
  WiFi.begin("Wokwi-GUEST", "", 6);
  wifiStartedAt = millis();
  lastWiFiLogAt = millis();
  mqtt.setServer(BROKER, MQTT_PORT);
  mqtt.setCallback(onMessage);
}

void loop() {
  unsigned long now = millis();
  wl_status_t wifiStatus = WiFi.status();
  if ((int)wifiStatus != previousWiFiStatus) {
    previousWiFiStatus = (int)wifiStatus;
    Serial.printf("[WIFI] status=%d (%s)\\n", (int)wifiStatus,
                  wifiStatus == WL_CONNECTED ? "CONNECTED" : "NOT CONNECTED");
    if (wifiStatus == WL_CONNECTED) {
      Serial.print("[WIFI] Connected! IP: ");
      Serial.println(WiFi.localIP());
      screen("WiFi connected", "Connecting MQTT...", "AI running on PC", "Wokwi hardware");
    }
  }
  if (wifiStatus != WL_CONNECTED && now - lastWiFiLogAt >= WIFI_LOG_MS) {
    lastWiFiLogAt = now;
    Serial.printf("[WIFI] Waiting for Wokwi-GUEST... status=%d\\n", (int)wifiStatus);
    screen("WiFi not connected", "SSID: Wokwi-GUEST", "Check IoT Gateway", "Serial Monitor");
  }
  // Do not reset the WiFi connection every 6 seconds: Wokwi's
  // virtual gateway can take some time. Retry only after 30 seconds.
  if (wifiStatus != WL_CONNECTED && now - wifiStartedAt >= WIFI_RETRY_MS) {
    wifiStartedAt = now;
    Serial.println("[WIFI] Retry after 30s without connection...");
    WiFi.disconnect();
    WiFi.begin("Wokwi-GUEST", "", 6);
  }
  if (wifiStatus == WL_CONNECTED) {
    if (!mqtt.connected() && now - lastConnectAttempt > 4000) {
      lastConnectAttempt = now;
      String id = "palm-wokwi-" + String((uint32_t)ESP.getEfuseMac(), HEX);
      Serial.printf("[MQTT] connect to %s:%d...\n", BROKER, MQTT_PORT);
      if (mqtt.connect(id.c_str())) {
        if (mqtt.subscribe(TOPIC)) {
          Serial.printf("[MQTT] CONNECTED, subscribed: %s\n", TOPIC);
          screen("Wokwi MQTT: READY", "Shoot REAL photo", "in PC web browser",
                 doorOpen() ? "DOOR OPEN" : "DOOR CLOSED");
        } else Serial.println("[MQTT] Subscribe failed");
      } else Serial.printf("[MQTT] connect failed rc=%d\n", mqtt.state());
    }
    if (mqtt.connected()) mqtt.loop();
  }
  bool currentDoorOpen = doorOpen();
  if (lastDoorOpen != currentDoorOpen) {
    lastDoorOpen = currentDoorOpen;
    Serial.println(currentDoorOpen ? "[DOOR] OPEN" : "[DOOR] CLOSED");
    if (currentDoorOpen && motorPulsing) {
      lockClosed();
      screen("DOOR OPEN", "MOTOR STOPPED", "Bench simulation", "Close the switch");
    }
  }
  if (motorPulsing && now - startedPulse >= AUTO_CLOSE_MS) {
    lockClosed();
    Serial.println("[LOCK] BENCH MOTOR PULSE FINISHED");
    screen("MOTOR CLOSED", "Ready for next shot", "PC camera -> PC AI", "MQTT -> ESP32");
  }
  delay(10);
}
