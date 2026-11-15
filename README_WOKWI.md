# PalmLock-AI - Wokwi hardware integration (separate Git branch)

Branch: `feat/wokwi-palm-lock-simulation`. The original Python / PyTorch AI files are **unchanged**.

## What actually runs
- Wokwi ESP32: button selection, UART RX/TX frame parsing via **loopback**, OLED, score ranking, threshold decision, red/green LEDs, servo control, auto-close.
- Dataset replay: **32 recorded cosine score vectors** from the user's `test_scores.py` results; expected top-1 accuracy **13/32 = 40.625%** on those vectors.
- Synthetic test: a push button injects 95% top cosine only to demonstrate the positive path of the controller; it is **NOT** biometric inference.

## What does NOT run on Wokwi
- Wokwi currently supports **one microcontroller per project**, so you cannot connect and run the ESP32 and ESP32-CAM together in one simulation: https://docs.wokwi.com/faq
- Wokwi's usual ESP32 part does **not** provide a live OV2640 image stream. This is a data replay, not a camera simulator.
- No code here runs PyTorch, MediaPipe, CompNet or PalmBridge inference **on the ESP32**. A `.pt` model cannot be directly copied into these boards; the original full model is too resource-intensive for the known ESP32 4MB Flash / 2MB PSRAM setup.

## Run directly in the browser
1. Visit **https://wokwi.com/projects/new/esp32**.
2. Use files from this branch's `wokwi/` folder:
   - `sketch.ino` → paste into sketch tab.
   - `diagram.json` → paste into diagram tab.
   - `demo_scores.h` → add a new file with exact same name, paste code.
   - `libraries.txt` → add or configure those three libraries through Wokwi Library Manager.
3. Click **Start Simulation**. OLED displays current test image.
4. Press **NEXT** (keyboard N) to move through test images, **SCAN** (S) to replay actual cosine scores over simulated UART, **TEST 95%** (T) to demonstrate servo opening with an *artificial* accepted frame. Red LED = closed, green LED = open. Auto-close after 2.5 seconds.
5. Watch Serial Monitor for every predicted user, 4 cosine percentages, comparison to known label, and lock status.

### Wiring in diagram.json
| Device | Pins |
|---|---|
| ESP32 DevKitC | Main controller |
| UART simulated camera node | GPIO17 TX2 connected directly to GPIO16 RX2 (loopback) |
| Button NEXT | GPIO32 → button → GND |
| Button SCAN | GPIO33 → button → GND |
| Button TEST | GPIO25 → button → GND |
| OLED SSD1306 | SDA21, SCL22, VCC5V, GND |
| Servo | PWM18, V+5V (simulation only), GND |
| Green LED | GPIO26 → 220Ω → LED → GND |
| Red LED | GPIO27 → 220Ω → LED → GND |

## Two physical boards (bench only)
A separate example `firmware/esp32cam_score_replay/esp32cam_score_replay.ino` uses AI-Thinker ESP32-CAM pinout. It can actually capture a JPEG (just to test hardware) and then transmits **previously recorded** scores, unrelated to that new image. Only the existing controller code ranks cosine similarities.

- Common GND between both physical boards.
- ESP32-CAM GPIO14 (UART1 TX, with SD card **unused**) → ESP32 controller GPIO16 (UART2 RX).
- ESP32-CAM GPIO13 → button → GND triggers capture + replay; its `Serial` uses the programming UART for debug.
- ESP32 controller: upload `wokwi/sketch.ino` along with `wokwi/demo_scores.h`, but remove the loopback wire **GPIO17→GPIO16**. With no loopback, a press of `SCAN` merely sends to an unused TX17; incoming score messages from camera board are received on RX16.
- All firmware is a **bench demonstration**; the synthetic TEST button must not be wired to a real lock.

ESP32-CAM physical caveats: correct camera pin mapping is board-specific; example targets AI-Thinker OV2640. Servo must have external regulated supply with common ground. To actually get real palm recognition from new captures, you still need **a compatible on-device inference pipeline**, which is NOT delivered here. The original full PalmBridge/MediaPipe model remains running on PC via `python test_scores.py`.

## Regenerate from fresh model results
Run `python test_scores.py` on PC, then:

```powershell
python wokwi/export_scores.py --csv test_results/score_per_image.csv
```

This updates the replay table with the latest model scores. It **does not** update AI model weights, require training, or turn scores into probability.

## Test expectations and limits
1. Press SCAN with a person_01 image: person_04 will likely be top candidate, model's actual recorded error is reproduced.
2. Select a person_03 image: person_03 should rank highest, but 0.85 gate will **deny** the low similarity.
3. Press TEST: artificial person_01 score 95.00% passes gate; green LED and servo move, then auto-close.
4. No real unknown-person inference: without genuine on-device AI there is no FAR/FRR measurement in Wokwi.
5. UART frames are unauthenticated; not suitable for a physical access-control system.
