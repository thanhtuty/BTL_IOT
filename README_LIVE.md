# PalmLock-AI LIVE - real photos + original PalmBridge

This branch adds a REAL ESP32-CAM capture path. The physical OV2640 takes
a new JPEG. The local Python app downloads that JPEG, saves it, crops the palm
via original MediaPipe, runs the original CompNet/PalmBridge checkpoint on PC,
and reports cosine scores for 4 previously enrolled users.

**Wokwi does not provide a real simulated OV2640 camera stream.**
For a real image use an actual ESP32-CAM or the browser's webcam.
No on-chip inference is claimed. No door/servo is actuated.

## Upload firmware to AI-Thinker ESP32-CAM

Open Arduino IDE and open:

    live/esp32cam/esp32cam_live.ino

Change WIFI_SSID and WIFI_PASSWORD to your own 2.4GHz network credentials.
Choose Board: AI Thinker ESP32-CAM (Espressif). Use correct COM port.
For USB-UART programmer use adapter TX->U0R, RX->U0T, GND->GND.
Feed a stable regulated 5V to 5V pin (check adapter ability). GPIO0 to GND
during bootloader upload, disconnect GPIO0 and reset for normal running.
Do not put 5V UART levels into 3.3V signal pins.

Serial Monitor at 115200 will show:

    [READY] Browser: http://192.168.1.123/
    [READY] Direct JPEG: http://192.168.1.123/capture

Open the printed camera IP on the same LAN and press "Chup anh moi".
That is a new JPEG from the OV2640, NOT replayed results.

## Real AI from ESP32-CAM to PC (same LAN)

In VS Code PowerShell at project root with your virtualenv activated:

    python -m pip install -r live/requirements_live.txt
    python live/live_web.py --camera-url http://192.168.1.123/capture

Change IP to the one in your serial monitor.

Open http://127.0.0.1:8765 in PC browser. Press "Chụp từ ESP32-CAM".
The page shows actual source JPEG, actual ROI used for inference,
prediction, four cosine scores (cosine x100, not probabilities) and image
download link. You may optionally select a real person label to see
whether top-1 matches the known label.

Your pre-existing local files must exist for AI:

    weights/palmbridge_esp32_finetuned.pt
    weights/hand_landmarker.task
    database/test_4people_12epoch.json

Every image and ROI is stored under live_captures/ on the PC. It is Git-ignored.

## Real camera capture without weights / AI

    python live/live_web.py --camera-url http://192.168.1.123/capture --capture-only

This still downloads and saves freshly captured camera JPEGs, but skips AI.

## No ESP32-CAM available? Use laptop webcam now

    python live/live_web.py

Open http://127.0.0.1:8765 and click "Bật webcam trình duyệt", allow access,
then "Chụp webcam và nhận diện". This takes a new real webcam frame;
an image upload option is also included. Use --capture-only if weights
are not available yet.

## Actual processing flow

    ESP32-CAM OV2640 -> HTTP /capture -> PC Flask server -> local JPEG
    -> MediaPipe ROI -> grayscale + resize 128x128 + normalization
    -> original CompNet + PalmBridge checkpoint -> embedding 512
    -> cosine similarity to 4 enrolled user vectors -> web page results

This preserves the original AI implementation but **runs it on the PC**,
not on ESP32. Existing Wokwi examples still work as a separate lock
electronics demo, but are not connected to live camera data.

## Testing & security notes

- The model identified 13/32 held-out photos correctly by top cosine
  (~40.6%). It is not reliable enough for real door access.
- The live app deliberately has NO actuator interface, no automatic unlock.
- Wokwi hardware replay does not prove on-device camera inference.
- The camera /capture endpoint is not authenticated. Use a trusted private
  LAN only, do not expose it via port forwarding or internet.
- Never commit actual Wi-Fi passwords, face/palm images, templates or weights.
- If Arduino upload reports "Invalid head of packet (0x65)", that is a
  serial upload issue rather than an AI error: check proper power, GPIO0
  bootloader mode, stable USB-UART, correct COM and 115200 upload speed.
