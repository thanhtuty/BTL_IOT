#include "PalmLiteV2.h"
#include <Arduino.h>
#include <esp_heap_caps.h>
#include <math.h>
#include <string.h>
#include "generated_palm_v2.h"  // generated locally by export; intentionally not committed

namespace w = palm_v2_weights;

// Largest feature map: 24 channels * 32 * 32 = 24576 floats.
// Three buffers use approximately 288 KB external RAM.
static constexpr int kBufferFloats = 24 * 32 * 32;
static float *a = nullptr, *b = nullptr, *c = nullptr;

bool palmLiteBegin() {
  if (a && b && c) return true;
  palmLiteEnd();
  a = (float*)heap_caps_malloc(kBufferFloats * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  b = (float*)heap_caps_malloc(kBufferFloats * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  c = (float*)heap_caps_malloc(kBufferFloats * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!a || !b || !c) {
    palmLiteEnd();
    return false;
  }
  return true;
}

void palmLiteEnd() {
  if (a) heap_caps_free(a);
  if (b) heap_caps_free(b);
  if (c) heap_caps_free(c);
  a = b = c = nullptr;
}

static bool normalize(const uint8_t *roi, float *dst) {
  double sum = 0.0;
  int n = 0;
  for (int i = 0; i < 4096; ++i) if (roi[i]) {
    sum += (double)roi[i] / 255.0;
    ++n;
  }
  if (n < 4) return false;
  double mean = sum / n, variance = 0;
  for (int i = 0; i < 4096; ++i) if (roi[i]) {
    double d = (double)roi[i] / 255.0 - mean;
    variance += d * d;
  }
  float stdv = sqrtf((float)(variance / (n - 1)));
  for (int i = 0; i < 4096; ++i) {
    dst[i] = roi[i] ? (float)(((double)roi[i] / 255.0 - mean) / (stdv + 1e-6f)) : 0.0f;
  }
  return true;
}

static void gabor(const float *in, float *out) {
  for (int oc = 0; oc < 4; ++oc) {
    const float *kernel = w::gabor + oc * 49;
    for (int y = 0; y < 64; ++y) {
      for (int x = 0; x < 64; ++x) {
        float acc = 0.0f;
        for (int ky = 0; ky < 7; ++ky) {
          int iy = y + ky - 3;
          if (iy < 0 || iy >= 64) continue;
          for (int kx = 0; kx < 7; ++kx) {
            int ix = x + kx - 3;
            if (ix < 0 || ix >= 64) continue;
            acc += in[iy * 64 + ix] * kernel[ky * 7 + kx];
          }
        }
        out[oc * 4096 + y * 64 + x] = acc > 0 ? acc : 0;
      }
      if ((y & 15) == 0) yield();
    }
  }
}

// Float32 reference: same convolution ordering as torch Conv2d, fused BN.
static void stem(const float *in, float *out) {
  for (int oc = 0; oc < 16; ++oc)
    for (int y = 0; y < 32; ++y) {
      for (int x = 0; x < 32; ++x) {
        float acc = w::stem_b[oc];
        for (int ic = 0; ic < 4; ++ic) {
          const float *kernel = w::stem_w + (oc * 4 + ic) * 9;
          const float *plane = in + ic * 4096;
          for (int ky = 0; ky < 3; ++ky) {
            int iy = y * 2 + ky - 1;
            if (iy < 0 || iy >= 64) continue;
            for (int kx = 0; kx < 3; ++kx) {
              int ix = x * 2 + kx - 1;
              if (ix < 0 || ix >= 64) continue;
              acc += plane[iy * 64 + ix] * kernel[ky * 3 + kx];
            }
          }
        }
        out[oc * 1024 + y * 32 + x] = acc > 0 ? acc : 0;
      }
      if (!(y & 15)) yield();
    }
}

static void depthwise(const float *in, float *out,
                      int channels, int side, int stride, const float *weights) {
  int next = (side + 1) / stride;
  for (int channel = 0; channel < channels; ++channel) {
    const float *kernel = weights + channel * 9;
    const float *plane = in + channel * side * side;
    float *dst = out + channel * next * next;
    for (int y = 0; y < next; ++y) {
      for (int x = 0; x < next; ++x) {
        float acc = 0;
        for (int ky = 0; ky < 3; ++ky) {
          int iy = y * stride + ky - 1;
          if (iy < 0 || iy >= side) continue;
          for (int kx = 0; kx < 3; ++kx) {
            int ix = x * stride + kx - 1;
            if (ix < 0 || ix >= side) continue;
            acc += plane[iy * side + ix] * kernel[ky * 3 + kx];
          }
        }
        dst[y * next + x] = acc;
      }
    }
    if (!(channel & 7)) yield();
  }
}

static void pointwise(const float *in, float *out, int inputChannels,
                      int outputChannels, int side, const float *weights, const float *biases) {
  int pixels = side * side;
  for (int oc = 0; oc < outputChannels; ++oc) {
    const float *row = weights + oc * inputChannels;
    float *dst = out + oc * pixels;
    for (int p = 0; p < pixels; ++p) {
      float acc = biases[oc];
      for (int ic = 0; ic < inputChannels; ++ic)
        acc += in[ic * pixels + p] * row[ic];
      dst[p] = acc > 0 ? acc : 0;
    }
    if (!(oc & 7)) yield();
  }
}

static void block(int inChannels, int outChannels, int &side, int stride,
                  const float *dw, const float *pw, const float *bias) {
  depthwise(b, c, inChannels, side, stride, dw);
  side = (side + 1) / stride;
  pointwise(c, b, inChannels, outChannels, side, pw, bias);
}

bool palmLitePredict(const uint8_t roi64[4096], PalmLitePrediction &out) {
  out = PalmLitePrediction{};
  if (!roi64 || !a || !b || !c) return false;
  unsigned long start = millis();
  if (!normalize(roi64, b)) return false;
  gabor(b, a);
  stem(a, b);
  int side = 32;
  block(16, 24, side, 1, w::dw0_w, w::pw0_w, w::pw0_b);
  block(24, 32, side, 2, w::dw1_w, w::pw1_w, w::pw1_b);
  block(32, 48, side, 2, w::dw2_w, w::pw2_w, w::pw2_b);
  block(48, 64, side, 2, w::dw3_w, w::pw3_w, w::pw3_b);
  block(64, 64, side, 1, w::dw4_w, w::pw4_w, w::pw4_b);
  if (side != 4) return false;
  float pooled[64];
  for (int channel = 0; channel < 64; ++channel) {
    float sum = 0;
    for (int p = 0; p < 16; ++p) sum += b[channel * 16 + p];
    pooled[channel] = sum / 16.0f;
  }
  float z[128], sumSquares = 0;
  for (int oc = 0; oc < 128; ++oc) {
    float value = w::fc_b[oc];
    for (int ic = 0; ic < 64; ++ic)
      value += w::fc_w[oc * 64 + ic] * pooled[ic];
    z[oc] = value;
    sumSquares += value * value;
  }
  if (sumSquares < 1e-24f) return false;
  float norm = sqrtf(sumSquares);
  for (float &value : z) value /= norm;
  int best = 0;
  for (int person = 0; person < 4; ++person) {
    float cosine = 0;
    for (int j = 0; j < 128; ++j)
      cosine += z[j] * w::centroids[person * 128 + j];
    out.scores[person] = cosine;
    if (person == 0 || cosine > out.scores[best]) best = person;
  }
  float second = -2.0f;
  for (int person = 0; person < 4; ++person)
    if (person != best && out.scores[person] > second) second = out.scores[person];
  out.best = out.scores[best];
  out.margin = out.best - second;
  out.candidate = best + 1;
  out.accepted = out.best >= w::kThreshold;
  out.person = out.accepted ? best + 1 : 0;
  out.latencyMs = millis() - start;
  out.valid = true;
  return true;
}
