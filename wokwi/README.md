# PalmLock-AI Wokwi (one-board UART loopback)

**Purpose:** simulate ESP32 control electronics and UART score protocol, *not* live ESP32-CAM image recognition. Wokwi officially does not support running two wired MCUs in one project: https://docs.wokwi.com/faq

## Run on Wokwi.com

1. Open https://wokwi.com/projects/new/esp32 and create an ESP32 Arduino project.
2. Copy this folder's `sketch.ino` into the Wokwi `sketch.ino` tab.
3. Create a new tab/file `demo_scores.h` and copy its content.
4. Replace `diagram.json` with this folder's diagram.
5. Add `libraries.txt` containing the 3 dependencies shipped here (Wokwi's library manager may auto-create the file).
6. Press the green Start simulation button.
7. Press `N` (NEXT) to choose one of 32 actual recorded test cases; `S` (SCAN) to send its cosine scores over UART loopback; `T` (TEST) to send a **synthetic** 95% vector to test actual servo opening, then auto-close after 2.5 seconds.

**Expected behavior:** Most real replay scores are low (max <= 50.52%); with demo threshold 85%, scans are DENIED. These replays are truthful to user logs. TEST generates artificial high score and opens servo, solely for actuator testing. Green LED = open, red LED = locked. OLED and Serial Monitor show every decision. The recorded sample table has 13/32 top-1 correct results (40.625%).

### Wiring (virtual board)

| Part | Pin | ESP32 |
|---|---|---|
| UART loopback | TX2 | RX2, GPIO17 → GPIO16 |
| NEXT | input + GND | GPIO32 |
| SCAN | input + GND | GPIO33 |
| TEST high-score frame | input + GND | GPIO25 |
| Servo PWM | PWM | GPIO18 |
| OLED | SDA / SCL | GPIO21 / GPIO22 |
| Green LED via 220-ohm resistor | anode | GPIO26 |
| Red LED via 220-ohm resistor | anode | GPIO27 |

Both LEDs' cathodes go to GND. Servo V+ is connected to 5 V *in the simulation*. On real hardware, supply servo with an **external regulated power supply** and shared ground; never power it from an ESP32 GPIO.

### Protocol

`S,index,score1,score2,score3,score4\n` ; scores are integers in units cosine similarity * 10000. E.g. `S,1,1842,1730,1133,2371` replays palm_02 of person_01, where the top score is person_04 (23.71%). `index=999` means synthetic test only. No real feature extraction is performed in this firmware.

### Security and scientific limits

- The label and image filename in a demo case are used only for testing/logging, not to choose the predicted user.
- The score-to-user and actuator logic uses a fixed threshold of 0.85 solely as a safe, *unvalidated* demo choice; do not use it as a validated real lock policy.
- The absence of a camera feed means this cannot measure latency, RAM consumption, or true inference on ESP32-CAM.
- This UART protocol has **no authentication or integrity** and is for a closed bench simulation only; do not connect an actual door lock or rely on the TEST button in a real deployment.
- To verify original AI accuracy, run `python test_scores.py` on PC using `weights/palmbridge_esp32_finetuned.pt` and the held-out images.
- For two real physical boards, see `firmware/esp32cam_score_replay/` and wiring notes in `README_WOKWI.md`.
