// PalmLock Wokwi: ESP32 + UART camera-data LOOPBACK + servo + OLED
// DEMO ONLY! Does not run MediaPipe, CompNet, PalmBridge, or capture OV2640.
// Replays real scores previously obtained from PalmBridge on the PC.
// Connect GPIO17 (TX2) -> GPIO16 (RX2) for one-board Wokwi simulation.
// REMOVE loopback wire on real hardware and send frames from a second MCU.
#include <Arduino.h>
#include <Wire.h>
#include <ESP32Servo.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "demo_scores.h"

static const int PIN_NEXT = 32;
static const int PIN_SCAN = 33;
static const int PIN_TEST = 25;
static const int PIN_GREEN = 26;
static const int PIN_RED = 27;
static const int PIN_SERVO = 18;
static const int PIN_UART_RX = 16;
static const int PIN_UART_TX = 17;
static const int THRESHOLD_X10000 = 8500; // 0.85. Lock remains fail-closed.
static const unsigned long OPEN_MS = 2500;
Adafruit_SSD1306 oled(128, 64, &Wire, -1);
Servo lockServo;
bool oledOk = false;
int selected = 0;
unsigned long openedAt = 0;
bool lockOpened = false;
char rxLine[96];
size_t rxSize = 0;

void closeLock() {
  lockServo.write(0);
  digitalWrite(PIN_GREEN, LOW);
  digitalWrite(PIN_RED, HIGH);
  lockOpened = false;
}
void openLock() {
  lockServo.write(90);
  digitalWrite(PIN_GREEN, HIGH);
  digitalWrite(PIN_RED, LOW);
  lockOpened = true;
  openedAt = millis();
}
void screen(const char *line1, const char *line2, const char *line3, const char *line4) {
  if (!oledOk) return;
  oled.clearDisplay();
  oled.setTextSize(1);
  oled.setTextColor(SSD1306_WHITE);
  oled.setCursor(0, 0); oled.println("PALMLOCK / WOKWI DEMO");
  oled.drawLine(0, 11, 127, 11, SSD1306_WHITE);
  oled.setCursor(0, 16); oled.println(line1);
  oled.setCursor(0, 28); oled.println(line2);
  oled.setCursor(0, 40); oled.println(line3);
  oled.setCursor(0, 52); oled.println(line4);
  oled.display();
}
void showSelected() {
  char a[36], b[36];
  snprintf(a,sizeof(a),"Image %d/%d",selected+1,(int)DEMO_COUNT);
  snprintf(b,sizeof(b),"Expected: person_%02d",DEMO_SAMPLES[selected].expected);
  screen(a,b,"NEXT=change  SCAN=run","TEST=synthetic accept");
  Serial.printf("[SELECT] %d: %s\n",selected+1,DEMO_SAMPLES[selected].file);
}

// Protocol: S,index,p1,p2,p3,p4\n  integer scores = cosine*10000
// Example recorded score: S,1,1842,1730,1133,2371
void sendRecordedSample() {
  const DemoSample &d=DEMO_SAMPLES[selected];
  Serial.printf("[CAM-REPLAY] %s (not a live camera capture)\n",d.file);
  Serial2.printf("S,%d,%d,%d,%d,%d\n",selected,
                 d.score[0],d.score[1],d.score[2],d.score[3]);
}
void sendSyntheticAllowed() {
  // Explicit positive test only, not a biometric result.
  Serial.println("[SYNTHETIC] Injecting artificial 95% match to test servo path");
  Serial2.print("S,999,9500,1100,800,1000\n");
}
void consumeFrame(const char *text) {
  int index, score[4];
  if (sscanf(text,"S,%d,%d,%d,%d,%d",&index,
             &score[0],&score[1],&score[2],&score[3])!=5) {
    Serial.println("[UART] Invalid frame, fail-closed"); closeLock(); return;
  }
  if (!((index>=0 && index<(int)DEMO_COUNT) || index==999)) {
    Serial.println("[UART] Unknown sample index, fail-closed"); closeLock(); return;
  }
  for (int i=0;i<4;i++) if (score[i]<-10000 || score[i]>10000) {
    Serial.println("[UART] Score out of range, fail-closed"); closeLock(); return;
  }
  int best=0;
  for (int i=1;i<4;i++) if (score[i]>score[best]) best=i;
  const int predicted=best+1;
  const bool synthetic=(index==999);
  const bool accepted=score[best]>=THRESHOLD_X10000;
  const int actual=synthetic ? 0 : DEMO_SAMPLES[index].expected;
  Serial.printf("[INFERENCE REPLAY] %s | predicted=person_%02d | score=%.2f%% | %s\n",
                synthetic ? "SYNTHETIC" : DEMO_SAMPLES[index].file, predicted,
                score[best]/100.0f, synthetic ? "ARTIFICIAL" : (predicted==actual?"TOP1_CORRECT":"TOP1_WRONG"));
  Serial.printf("[SCORES] P1=%.2f P2=%.2f P3=%.2f P4=%.2f\n",
                score[0]/100.0f,score[1]/100.0f,score[2]/100.0f,score[3]/100.0f);
  char a[32],b[32],c[32];
  snprintf(a,sizeof(a),"Predicted: person_%02d",predicted);
  snprintf(b,sizeof(b),"Cosine: %.2f%%",score[best]/100.0f);
  snprintf(c,sizeof(c),"%s",synthetic?"TEST FRAME":(predicted==actual?"TOP1 CORRECT":"TOP1 WRONG"));
  if (accepted) {
    Serial.println("[LOCK] OPEN -- accepted score (synthetic demo only with current replay set)");
    openLock();
    screen(a,b,c,"LOCK: OPEN (2.5s)");
  } else {
    Serial.println("[LOCK] DENIED -- score below 0.85, servo stays closed");
    closeLock();
    screen(a,b,c,"LOCK: DENIED");
  }
}
void pollUART() {
  while (Serial2.available()) {
    char ch=(char)Serial2.read();
    if (ch=='\n') {
      rxLine[rxSize]=0;
      consumeFrame(rxLine);
      rxSize=0;
    } else if (ch!='\r') {
      if (rxSize<sizeof(rxLine)-1) rxLine[rxSize++]=ch;
      else {rxSize=0;closeLock();Serial.println("[UART] Frame overflow");}
    }
  }
}
bool edgePressed(int pin, int ix) {
  static bool last[3]={HIGH,HIGH,HIGH};
  static unsigned long lastMs[3]={0,0,0};
  const bool now=(digitalRead(pin)==LOW);
  const bool fire=now && !last[ix] && millis()-lastMs[ix]>80;
  if(fire)lastMs[ix]=millis();
  last[ix]=now;
  return fire;
}
void setup() {
  Serial.begin(115200);
  pinMode(PIN_NEXT,INPUT_PULLUP);
  pinMode(PIN_SCAN,INPUT_PULLUP);
  pinMode(PIN_TEST,INPUT_PULLUP);
  pinMode(PIN_GREEN,OUTPUT);
  pinMode(PIN_RED,OUTPUT);
  Wire.begin(21,22);
  oledOk=oled.begin(SSD1306_SWITCHCAPVCC,0x3C);
  lockServo.setPeriodHertz(50);
  lockServo.attach(PIN_SERVO,500,2400);
  closeLock();
  Serial2.begin(115200,SERIAL_8N1,PIN_UART_RX,PIN_UART_TX);
  Serial.println("PalmLock Wokwi hardware integration / RECORDED-SCORES REPLAY ONLY");
  Serial.println("Wire GPIO17->GPIO16 creates UART loopback for one-MCU Wokwi.");
  Serial.println("NEXT: select photo; SCAN: replay model scores; TEST: artificial 95% unlock.");
  showSelected();
}
void loop() {
  if (edgePressed(PIN_NEXT,0)) {
    selected=(selected+1)%DEMO_COUNT;
    showSelected();
  }
  if (edgePressed(PIN_SCAN,1)) sendRecordedSample();
  if (edgePressed(PIN_TEST,2)) sendSyntheticAllowed();
  pollUART();
  if (lockOpened && millis()-openedAt>=OPEN_MS) {
    closeLock();
    Serial.println("[LOCK] AUTO-CLOSED");
    showSelected();
  }
  delay(5);
}
