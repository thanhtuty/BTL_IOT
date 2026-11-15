// Physical AI-Thinker ESP32-CAM bench test: CAMERA CAPTURE + REPLAY UART.
// VERY IMPORTANT: CompNet/PalmBridge/MediaPipe DO NOT run in this sketch.
// Capture is real on physical hardware; cosine scores are PRE-RECORDED,
// NOT computed from the new capture. Do NOT use to authorize a real door.
// No microSD; GPIO13 button to GND; GPIO14 TX -> controller GPIO16 RX.
// Ground pins of ESP32-CAM and controller MUST be connected.
#include <Arduino.h>
#include "esp_camera.h"
#include <HardwareSerial.h>

static const int TRIGGER_PIN=13;
static const int CAM_TX_PIN=14;
HardwareSerial linkToController(1);
int selected=0;
bool lastButton=HIGH;
unsigned long lastMs=0;

struct RecordedCase {
  int index; const char* label; int score[4];
};
// These are exact rounded values from user logs (cosine * 10000).
static const RecordedCase CASES[]={
  {1,  "person_01/palm_02", {1842,1730,1133,2371}}, // wrong
  {11, "person_02/palm_02", {629,5052,-1693,1565}}, // right
  {12, "person_03/palm_01", {218,-1275,3390,297}}, // right
  {29, "person_04/palm_08", {3892,-210,1942,4009}}  // right
};

bool initCamera() {
  camera_config_t config = {};
  config.ledc_channel=LEDC_CHANNEL_0;
  config.ledc_timer=LEDC_TIMER_0;
  config.pin_d0=5;
  config.pin_d1=18;
  config.pin_d2=19;
  config.pin_d3=21;
  config.pin_d4=36;
  config.pin_d5=39;
  config.pin_d6=34;
  config.pin_d7=35;
  config.pin_xclk=0;
  config.pin_pclk=22;
  config.pin_vsync=25;
  config.pin_href=23;
  config.pin_sccb_sda=26;
  config.pin_sccb_scl=27;
  config.pin_pwdn=32;
  config.pin_reset=-1;
  config.xclk_freq_hz=20000000;
  config.pixel_format=PIXFORMAT_JPEG;
  config.frame_size=FRAMESIZE_QVGA;
  config.jpeg_quality=15;
  config.fb_count=1;
  esp_err_t err=esp_camera_init(&config);
  Serial.printf("Camera init: 0x%x\n",(unsigned)err);
  return err==ESP_OK;
}

bool cameraReady=false;
void setup(){
  Serial.begin(115200);
  pinMode(TRIGGER_PIN,INPUT_PULLUP);
  linkToController.begin(115200,SERIAL_8N1,-1,CAM_TX_PIN);
  cameraReady=initCamera();
  Serial.println("ESP32-CAM BENCH: a live JPEG is captured, but AI scores are replayed from PC logs.");
}
void loop(){
  bool b=(digitalRead(TRIGGER_PIN)==LOW);
  if(b && !lastButton && millis()-lastMs>180){
    lastMs=millis();
    if(cameraReady){
      camera_fb_t* frame=esp_camera_fb_get();
      if(frame){
        Serial.printf("LIVE CAMERA: JPEG %u bytes; not analyzed by PalmBridge\n",(unsigned)frame->len);
        // Controller rejects C frames: no computed recognition result.
        linkToController.printf("C,%u\n",(unsigned)frame->len);
        esp_camera_fb_return(frame);
      }else Serial.println("Capture failed; no matching data transmitted");
    }
    // REPLAY from the OLD images (NOT the image just captured).
    const RecordedCase &s=CASES[selected];
    Serial.printf("REPLAY %s -> UART\n",s.label);
    linkToController.printf("S,%d,%d,%d,%d,%d\n",s.index,
      s.score[0],s.score[1],s.score[2],s.score[3]);
    selected=(selected+1)%4;
  }
  lastButton=b;
  delay(10);
}
