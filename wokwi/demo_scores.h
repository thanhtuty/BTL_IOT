#pragma once
#include <Arduino.h>

// Recorded PalmBridge/CompNet output from the user's 32 held-out images.
// NOT neural-network inference and NOT live camera frames.
// Score units: cosine_similarity * 10000, e.g. 23.71% -> 2371.
// These scores are replayed over UART for Wokwi hardware integration tests.
struct DemoSample {
  const char *file;
  uint8_t expected;
  int16_t score[4];
};

static const DemoSample DEMO_SAMPLES[] = {
  {"person_01/palm_01.jpg", 1, {1464, 1805, 1332, 2205}},
  {"person_01/palm_02.jpg", 1, {1842, 1730, 1133, 2371}},
  {"person_01/palm_03.jpg", 1, {1953, 1591, 1026, 2531}},
  {"person_01/palm_04.jpg", 1, {2024, 1688, 758, 2511}},
  {"person_01/palm_05.jpg", 1, {2250, 1640, 833, 2516}},
  {"person_01/palm_06.jpg", 1, {1931, 1625, 797, 2445}},
  {"person_01/palm_07.jpg", 1, {2070, 1580, 588, 2535}},
  {"person_01/palm_08.jpg", 1, {1911, 1707, 714, 2632}},
  {"person_01/palm_09.jpg", 1, {2310, 1532, 464, 2591}},
  {"person_01/palm_10.jpg", 1, {1986, 1732, 930, 2438}},
  {"person_02/palm_01.jpg", 2, {1634, 3750, -781, 4145}},
  {"person_02/palm_02.jpg", 2, {629, 5052, -1693, 1565}},
  {"person_03/palm_01.jpg", 3, {218, -1275, 3390, 297}},
  {"person_03/palm_02.jpg", 3, {71, -1024, 3512, 215}},
  {"person_03/palm_03.jpg", 3, {114, -1254, 3516, 239}},
  {"person_03/palm_04.jpg", 3, {241, -1310, 3545, 331}},
  {"person_03/palm_05.jpg", 3, {240, -1499, 3542, 455}},
  {"person_03/palm_06.jpg", 3, {286, -1224, 3585, 268}},
  {"person_03/palm_07.jpg", 3, {290, -1181, 3546, 313}},
  {"person_03/palm_08.jpg", 3, {342, -1225, 3631, 303}},
  {"person_03/palm_09.jpg", 3, {85, -1088, 3711, 311}},
  {"person_03/palm_10.jpg", 3, {24, -1142, 3831, 286}},
  {"person_04/palm_01.jpg", 4, {3766, -1234, 2803, 3315}},
  {"person_04/palm_02.jpg", 4, {3972, -1087, 2738, 3391}},
  {"person_04/palm_03.jpg", 4, {3828, -1204, 2736, 3509}},
  {"person_04/palm_04.jpg", 4, {3954, -1326, 2760, 3569}},
  {"person_04/palm_05.jpg", 4, {3858, -1043, 2627, 3602}},
  {"person_04/palm_06.jpg", 4, {4200, -73, 2096, 3173}},
  {"person_04/palm_07.jpg", 4, {3900, -162, 2204, 3810}},
  {"person_04/palm_08.jpg", 4, {3892, -210, 1942, 4009}},
  {"person_04/palm_09.jpg", 4, {4023, -211, 1975, 3936}},
  {"person_04/palm_10.jpg", 4, {3881, -84, 1911, 4005}}
};
static const size_t DEMO_COUNT = sizeof(DEMO_SAMPLES)/sizeof(DEMO_SAMPLES[0]);
