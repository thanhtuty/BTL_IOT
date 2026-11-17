#pragma once
#include <stdint.h>

// Reference CPU inference for ESP32-D0WDQ6-V3 with external PSRAM.
// This module does NOT operate a relay or authorize a door by itself.
struct PalmLitePrediction {
  int person = 0;       // 0 = unknown, 1..4 = accepted identity
  int candidate = 0;    // top candidate even when rejected
  float best = -2.0f;
  float margin = -2.0f;
  float scores[4] = {0, 0, 0, 0};
  unsigned long latencyMs = 0;
  bool accepted = false;
  bool valid = false;
};

bool palmLiteBegin();
void palmLiteEnd();
bool palmLitePredict(const uint8_t roi64[4096], PalmLitePrediction &out);
