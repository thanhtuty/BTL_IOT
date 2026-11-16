/*
 PalmLock ESP32-CAM ON-CHIP ALGORITHM BENCH — WOKWI VERSION
 ----------------------------------------------------------
 REAL ESP32 executable logic (NO Python/MQTT in this simulation):
 - Generate synthetic 160x120 grayscale "camera" frame in ESP32 RAM
 - Run the SAME 180-d gradient / cell-brightness descriptor math as
   firmware/esp32cam_onchip_bench/esp32cam_onchip_bench.ino
 - Enroll four profiles, save normalized descriptor sums to NVS
 - Compare cosine, calculate margin, apply demo thresholds
 - Control servo, OLED, LEDs and buzzer via virtual ESP32 GPIO

 WOKWI CANNOT ACQUIRE A REAL OV2640 FRAME. The frames here are
 PROCEDURAL SYNTHETIC TEXTURES, NOT real palm photos.
 This is NOT the trained PyTorch CompNet/PalmBridge network.
 These scores demonstrate math and wiring ONLY, not biometric accuracy.
 Do NOT use these thresholds, templates or this firmware on a real door.
*/
#include <Arduino.h>
#include <Preferences.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ESP32Servo.h>
#include <math.h>
#include <string.h>

// Wokwi virtual ESP32 DevKitC pins, NOT AI-Thinker ESP32-CAM pinout.
static const int PIN_NEXT = 32;   // N: next input profile (01..04, UNKNOWN)
static const int PIN_ENROLL = 33; // E: enroll 4 synthetic profiles locally
static const int PIN_SCAN = 25;   // S: generate new frame + run algorithm
static const int PIN_TEST = 14;   // T: 2s motor-only bench test
static const int PIN_SERVO = 18;
static const int PIN_BUZZER = 19;
static const int PIN_GREEN = 26;
static const int PIN_RED = 27;
static const int PIN_DOOR = 23;   // LOW=closed, HIGH=open

static const int FRAME_W = 160, FRAME_H = 120;
static const int ROI_X = 32, ROI_Y = 12, ROI_SIZE = 96;
static const int CELLS = 6, BINS = 5;
static const int FEAT_DIM = CELLS * CELLS * BINS;
static const int ENROLL_SAMPLES = 8;
static const uint32_t TEMPLATE_MAGIC = 0x504C3031u;
static const float COSINE_GATE = 0.94f; // NOT calibrated on real palms
static const float MARGIN_GATE = 0.07f; // NOT calibrated on real palms
static const bool PULSE_ON_SYNTHETIC_MATCH = true; // WOKWI TEST ONLY!
static const unsigned long OPEN_MS = 2000;

struct Template {
  uint32_t magic;
  uint16_t count;
  uint16_t dim;
  float sum[FEAT_DIM];
};

static Template templates[4] = {};
static uint8_t frame[FRAME_W * FRAME_H] = {};
Preferences store;
Adafruit_SSD1306 oled(128, 64, &Wire, -1);
Servo lockServo;
bool oledOk = false, unlocked = false, registered = false;
unsigned long openedAt = 0, scanNumber = 0;
int selected = 0; // 0..3 enrolled profiles, 4 unknown input

static bool doorOpen() {
  return digitalRead(PIN_DOOR) == HIGH;
}
static void lockClosed() {
  lockServo.write(0);
  digitalWrite(PIN_GREEN, LOW);
  digitalWrite(PIN_RED, HIGH);
  unlocked = false;
}
static void unlockBench() {
  lockServo.write(90);
  digitalWrite(PIN_GREEN, HIGH);
  digitalWrite(PIN_RED, LOW);
  tone(PIN_BUZZER, 1600, 160);
  unlocked = true;
  openedAt = millis();
}
static void beepDeny() { tone(PIN_BUZZER, 490, 170); }

static void screen(const char *a, const char *b, const char *c, const char *d) {
  if (!oledOk) return;
  oled.clearDisplay();
  oled.setTextSize(1);
  oled.setTextColor(SSD1306_WHITE);
  oled.setCursor(0, 0); oled.println("PALMLOCK ON-CHIP TEST");
  oled.drawLine(0, 11, 127, 11, SSD1306_WHITE);
  oled.setCursor(0, 16); oled.println(a);
  oled.setCursor(0, 28); oled.println(b);
  oled.setCursor(0, 40); oled.println(c);
  oled.setCursor(0, 52); oled.println(d);
  oled.display();
}

static void showSelected() {
  char line[32];
  if (selected == 4) snprintf(line, sizeof(line), "Input: UNKNOWN");
  else snprintf(line, sizeof(line), "Input: person_%02d", selected + 1);
  screen(line, registered ? "E=reset enrollment" : "E=enroll 4 profiles",
         "N=next S=scan T=servo",
         doorOpen() ? "DOOR: OPEN" : "DOOR: CLOSED");
  Serial.printf("[INPUT] %s; images are SYNTHETIC, not real camera captures\n", line);
}

// Deterministic procedural textures representing 4 distinguishable ridge
// orientations, plus an unrelated UNKNOWN class. Only image transport is
// simulated; ALL feature extraction/classification below runs on ESP32.
static int modPositive(int n, int m) {
  int r = n % m;
  return r < 0 ? r + m : r;
}
static void generateFrame(int person, int variant) {
  int shift = (variant % 5) - 2;
  for (int y = 0; y < FRAME_H; y++) {
    for (int x = 0; x < FRAME_W; x++) {
      int phase;
      switch (person) {
        case 0: phase = x + shift + ((y / 29) % 2); break;
        case 1: phase = y + shift + ((x / 29) % 2); break;
        case 2: phase = x + y + shift; break;
        case 3: phase = x - y + shift; break;
        default: phase = 2*x + 3*y + (x*y/13) + shift; break;
      }
      int q = modPositive(phase, 18);
      int triangle = min(q, 18-q);
      int noise = modPositive(x*13 + y*7 + variant*19, 9) - 4;
      int gray = 128 + (triangle * 2 - 9) * 13 / 2 + noise / 2;
      frame[y * FRAME_W + x] = (uint8_t)constrain(gray, 0, 255);
    }
  }
}

// Adapted DIRECTLY from makeDescriptor() in
// firmware/esp32cam_onchip_bench/esp32cam_onchip_bench.ino
// Input buffer differs from OV2640 camera_fb_t but descriptor formula is same.
static bool makeDescriptor(const uint8_t *p, float out[FEAT_DIM]) {
  if (!p) return false;
  memset(out, 0, sizeof(float) * FEAT_DIM);
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
          if (aX > 2*aY) bin = 0;
          else if (aY > 2*aX) bin = 1;
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
static void clearInMemory(Template &t) {
  memset(&t, 0, sizeof(t));
  t.magic = TEMPLATE_MAGIC;
  t.dim = FEAT_DIM;
}
static void loadStoredTemplates() {
  store.begin("palmlock", false);
  registered = true;
  for (int i=0; i<4; i++) {
    char key[8]; snprintf(key,sizeof(key),"p%d",i+1);
    Template temp = {};
    if (store.getBytesLength(key) == sizeof(temp) &&
        store.getBytes(key,&temp,sizeof(temp)) == sizeof(temp) &&
        temp.magic == TEMPLATE_MAGIC && temp.dim == FEAT_DIM &&
        temp.count == ENROLL_SAMPLES) {
      templates[i] = temp;
    } else {
      clearInMemory(templates[i]);
      registered = false;
    }
  }
  Serial.printf("[NVS] templates loaded; all 4 enrolled=%s\n",
                registered ? "yes" : "no");
}

static void enrollFourProfiles() {
  Serial.println("[ENROLL] Generate synthetic grayscale frames inside ESP32...");
  unsigned long started=millis();
  Template updated[4];
  for (int person=0; person<4; person++) {
    clearInMemory(updated[person]);
    for (int variant=0; variant<ENROLL_SAMPLES; variant++) {
      generateFrame(person,variant);
      float features[FEAT_DIM];
      if (!makeDescriptor(frame,features)) {
        Serial.println("[ENROLL] Feature extraction failed");
        screen("ENROLL FAILED", "Algorithm error", "See Serial Monitor", "");
        return;
      }
      for (int i=0; i<FEAT_DIM; i++) updated[person].sum[i] += features[i];
      updated[person].count++;
    }
  }
  bool saved=true;
  for (int person=0; person<4; person++) {
    char key[8]; snprintf(key,sizeof(key),"p%d",person+1);
    if (store.putBytes(key,&updated[person],sizeof(Template)) != sizeof(Template))
      saved=false;
    templates[person] = updated[person];
  }
  registered=saved;
  Serial.printf("[ENROLL] 4 x %d synthetic frames; NVS=%s; %lu ms\n",
                ENROLL_SAMPLES, saved?"OK":"FAILED",
                (unsigned long)(millis()-started));
  screen("SYNTH ENROLLED", "4 people x 8 frames", saved?"NVS saved":"NVS ERROR",
         "N=choose S=scan");
}

static void scanSyntheticFrame() {
  if (!registered) {
    Serial.println("[SCAN] Enroll first: press E");
    beepDeny();
    screen("NOT REGISTERED", "Press E to enroll", "4 synthetic profiles", "");
    return;
  }
  uint32_t started = millis();
  int person = selected;
  generateFrame(person, (int)(8 + (++scanNumber % 20)));
  float f[FEAT_DIM];
  if (!makeDescriptor(frame,f)) {
    beepDeny(); screen("SCAN FAILED", "Descriptor error", "", "");
    return;
  }
  float scores[4], best=-2, second=-2;
  int bestId=-1;
  for (int i=0;i<4;i++) {
    scores[i]=templateCosine(templates[i],f);
    if (scores[i]>best) {
      second=best;best=scores[i];bestId=i;
    } else if (scores[i]>second) {
      second=scores[i];
    }
  }
  float margin=best-second;
  bool matched=best>=COSINE_GATE && margin>=MARGIN_GATE;
  unsigned long elapsed=(unsigned long)(millis()-started);
  Serial.printf("[SIM CAMERA] source=%s, frame=160x120 in ESP32 RAM (SYNTHETIC)\n",
                person==4?"UNKNOWN":("person_"+String(person+1)).c_str());
  Serial.printf("[ON-ESP32 GRADIENT] pred=person_%02d cos=%.5f "
                "margin=%.5f matched=%s time=%lums\n",
                bestId+1,best,margin,matched?"YES":"NO",elapsed);
  Serial.printf("[COSINE] p1=%.4f p2=%.4f p3=%.4f p4=%.4f\n",
                scores[0],scores[1],scores[2],scores[3]);
  Serial.println("[IMPORTANT] This is handcrafted gradient matching, NOT CompNet/PalmBridge AI!");
  char a[26],b[26],c[26];
  snprintf(a,sizeof(a),"Input: %s",person==4?"UNKNOWN":("person_"+String(person+1)).c_str());
  snprintf(b,sizeof(b),"Pred: person_%02d",bestId+1);
  snprintf(c,sizeof(c),"Cos %.3f Mar %.3f",best,margin);
  if (matched && !doorOpen() && PULSE_ON_SYNTHETIC_MATCH) {
    unlockBench();
    screen(a,b,c,"SYNTH MATCH->SERVO");
    Serial.println("[BENCH LOCK] PULSE for 2s (SIMULATED match, NOT security)");
  } else {
    lockClosed();
    beepDeny();
    screen(a,b,c,matched?"DOOR OPEN / DENY":"NO MATCH / DENY");
    Serial.println(matched?"[BENCH LOCK] DOOR OPEN / no unlock":"[BENCH LOCK] DENIED");
  }
}

static bool edgePressed(int pin,int index) {
  static bool wasDown[4]={false,false,false,false};
  static unsigned long lastPressed[4]={0,0,0,0};
  bool down=digitalRead(pin)==LOW;
  bool fire=down && !wasDown[index] && millis()-lastPressed[index]>=100;
  if (fire) lastPressed[index]=millis();
  wasDown[index]=down;
  return fire;
}
void setup() {
  Serial.begin(115200);
  pinMode(PIN_NEXT, INPUT_PULLUP);
  pinMode(PIN_ENROLL, INPUT_PULLUP);
  pinMode(PIN_SCAN, INPUT_PULLUP);
  pinMode(PIN_TEST, INPUT_PULLUP);
  pinMode(PIN_DOOR, INPUT_PULLUP);
  pinMode(PIN_GREEN, OUTPUT);
  pinMode(PIN_RED, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  Wire.begin(21,22);
  oledOk=oled.begin(SSD1306_SWITCHCAPVCC,0x3C);
  lockServo.setPeriodHertz(50);
  lockServo.attach(PIN_SERVO,500,2400);
  lockClosed();
  loadStoredTemplates();
  Serial.println("\n=== PalmLock ON-CHIP WOKWI DEMO ===");
  Serial.println("NO MQTT, NO Python, NO live camera, NO trained CompNet/PalmBridge.");
  Serial.println("Image source is SYNTHETIC grayscale patterns in ESP32 RAM.");
  Serial.println("E=enroll 4 synthetic persons, N=next, S=scan, T=servo test.");
  showSelected();
}
void loop() {
  if (edgePressed(PIN_NEXT,0)) {
    selected=(selected+1)%5;
    showSelected();
  }
  if (edgePressed(PIN_ENROLL,1)) enrollFourProfiles();
  if (edgePressed(PIN_SCAN,2)) scanSyntheticFrame();
  if (edgePressed(PIN_TEST,3)) {
    if (!doorOpen()) {
      unlockBench();
      screen("SERVO BENCH TEST", "2 sec pulse", "NOT AI INFERENCE",
             "DOOR CLOSED");
      Serial.println("[BENCH SERVO] manual 2s pulse, not biometric grant");
    } else {
      beepDeny();
      Serial.println("[BENCH SERVO] blocked because door is OPEN");
      showSelected();
    }
  }
  if (unlocked && millis()-openedAt>=OPEN_MS) {
    lockClosed();
    Serial.println("[BENCH LOCK] Auto-closed");
    showSelected();
  }
  delay(5);
}
