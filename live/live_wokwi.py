"""PC camera -> real PalmBridge inference -> Wokwi MQTT actuator demo.

PC does MediaPipe ROI + PyTorch inference. Wokwi NEVER performs vision/AI.
Broker receives only predicted person index, top cosine and sequence, not
images/embeddings. Plaintext public MQTT is FOR SIMULATION ONLY.
"""
import argparse
import io
import json
import secrets
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from flask import Flask, jsonify, request, send_from_directory
import paho.mqtt.client as mqtt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DEFAULT_HAND_LANDMARKER
from experiments.benchmark import loader, centroids
from models.checkpoint import load_encoder
from preprocessing import build_transform
from roi import HandPalmROICropper

CAPTURES = ROOT / "live_captures"
MAX_BYTES = 8 * 1024 * 1024

PAGE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PalmLock - camera that + AI that + Wokwi</title>
<style>
body{background:#121826;color:#edf2f9;font:16px/1.5 system-ui,Arial,sans-serif;margin:0}
main{max-width:1000px;margin:auto;padding:24px}
section{background:#1c273b;border:1px solid #364b66;border-radius:12px;padding:16px;margin-bottom:14px}
button{background:#477dea;color:white;border:0;border-radius:8px;padding:10px 14px;margin:6px 4px 6px 0;cursor:pointer;font:inherit}
button:disabled{opacity:.5}video,img{width:100%;max-width:600px;border-radius:8px;background:#101522}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
small,p{color:#b3c1d3}.ok{color:#91e8b0}.bad{color:#ffd199}
</style></head><body><main>
<h1>PalmLock: anh THAT → AI THAT → Wokwi</h1>
<p>Webcam may tinh / ESP32-CAM that → MediaPipe → Compact CompNet + PalmBridge tren PC → MQTT → Wokwi servo.</p>
<section>
<button onclick="startCamera()">1. Bat webcam</button>
<button onclick="captureWebcam()">2. Chup anh va chay AI</button>
<button onclick="captureESP()">Chup tu ESP32-CAM (neu co URL)</button>
<label>Hoac chon anh: <input id="up" type="file" accept="image/*"></label>
<button onclick="upload()">Phan tich anh</button>
<video id="video" autoplay playsinline muted style="display:none"></video>
<p id="status" class="bad">San sang. Hay mo Wokwi va kiem tra MQTT CONNECTED truoc.</p>
</section>
<div class="grid">
<section><h3>Anh moi chup</h3><img id="raw" alt="Chua co anh"></section>
<section><h3>ROI do MediaPipe cat</h3><img id="roi" alt="Chua co ROI"></section>
</div>
<section><h3>Ket qua mo hinh that</h3>
<strong id="pred">Chua co du doan</strong><div id="scores"></div>
<p id="bridge"></p>
<small>Cosine khong phai xac suat. MQTT demo khong duoc dung de mo cua that.
ESP32 trong Wokwi chi nhan ket qua, khong chay PyTorch/MediaPipe.</small>
</section>
<script>
let stream=null;
const statusEl=document.getElementById('status');
async function startCamera(){
 try {
  if(!stream)stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'},audio:false});
  const v=document.getElementById('video');v.srcObject=stream;v.style.display='block';await v.play();
  statusEl.textContent='Webcam dang hoat dong. Dua long ban tay ro vao giua khung.';
 }catch(e){statusEl.textContent='Loi webcam: '+e.message}
}
async function processResponse(res) {
 let data=await res.json();
 if(!res.ok)throw Error(data.error||('HTTP '+res.status));
 if(data.image_url)document.getElementById('raw').src=data.image_url+'?t='+Date.now();
 const roi=document.getElementById('roi');
 if(data.roi_url)roi.src=data.roi_url+'?t='+Date.now();else roi.removeAttribute('src');
 document.getElementById('pred').textContent=data.predicted?('Du doan '+data.predicted+'; cosine cao nhat '+data.best_cosine.toFixed(4)):'Chua nhan dien';
 document.getElementById('scores').textContent=data.scores?data.scores.map(x=>x.user+': '+x.cosine.toFixed(4)).join(' | '):'';
 document.getElementById('bridge').textContent=data.mqtt_published?'Da gui ket qua that sang Wokwi qua MQTT':('MQTT: '+(data.mqtt_error||'khong gui'));
 statusEl.textContent=data.error||data.message||'Da xu ly anh.';
}
async function submitBlob(blob,name){
 const fd=new FormData();fd.append('file',blob,name);
 const res=await fetch('/api/image',{method:'POST',body:fd});await processResponse(res);
}
async function captureWebcam(){
 try{
  const v=document.getElementById('video');
  if(!stream||!v.videoWidth){await startCamera();if(!v.videoWidth)return;}
  const c=document.createElement('canvas');c.width=v.videoWidth;c.height=v.videoHeight;
  c.getContext('2d').drawImage(v,0,0);
  const b=await new Promise(resolve=>c.toBlob(resolve,'image/jpeg',0.92));
  if(!b)throw Error('Khong tao duoc JPEG');
  statusEl.textContent='Anh moi da chup. Dang chay MediaPipe va PalmBridge...';
  await submitBlob(b,'webcam.jpg');
 }catch(e){statusEl.textContent='Loi: '+e.message}
}
async function captureESP(){
 try{statusEl.textContent='Dang chup anh moi tu ESP32-CAM...';
 const res=await fetch('/api/esp32cam',{method:'POST'});await processResponse(res);
 }catch(e){statusEl.textContent='Loi camera: '+e.message}
}
async function upload(){
 const f=document.getElementById('up').files[0];
 if(!f){statusEl.textContent='Chon anh truoc.';return;}
 try{await submitBlob(f,f.name)}catch(e){statusEl.textContent='Loi: '+e.message}
}
</script></main></body></html>"""


def protocol_message(index, cosine, seq):
    """Only index and similarity leave the PC; never transmit images."""
    if index not in range(1, 5):
        raise ValueError("Invalid predicted index.")
    if not -1.0 <= cosine <= 1.0:
        raise ValueError("Cosine outside [-1,1].")
    if not 0 < seq < 2**32:
        raise ValueError("Sequence out of range.")
    score = min(10000, max(-10000, round(cosine * 10000)))
    return f"P,{index},{score},{seq}"


class Predictor:
    def __init__(self, weights, prepared, hand_model):
        self.model, self.kind = load_encoder(weights, "cpu")
        self.input_size = getattr(self.model, "input_size", 128)
        self.transform = build_transform(self.input_size)
        train_dir = Path(prepared) / "train"
        if not train_dir.is_dir():
            raise FileNotFoundError(f"Khong tim thay train ROI: {train_dir}")
        ds = ImageFolder(train_dir, transform=self.transform, loader=loader)
        self.labels = ds.classes
        if self.labels != [f"person_{i:02d}" for i in range(1, 5)]:
            raise ValueError(f"Can du 4 thu muc person_01..04, nhan {self.labels}")
        with torch.inference_mode():
            self.center = centroids(
                self.model, DataLoader(ds, batch_size=4, shuffle=False, num_workers=0),
                len(self.labels), torch.device("cpu")
            )
        self.cropper = HandPalmROICropper(hand_model)

    @torch.inference_mode()
    def predict(self, photo):
        roi, _ = self.cropper.crop(photo)
        x = self.transform(roi.convert("L")).unsqueeze(0)
        z = self.model.get_embedding(x)
        sims = (z @ self.center.T)[0].cpu().tolist()
        i = max(range(len(sims)), key=lambda idx: sims[idx])
        return roi, i+1, float(sims[i]), [
            {"user": self.labels[k], "cosine": round(float(v), 6)}
            for k, v in enumerate(sims)
        ]

    def close(self):
        self.cropper.close()


class MqttBridge:
    def __init__(self, host, port, topic):
        self.topic = topic
        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="palmlock-pc-" + secrets.token_hex(4),
        )
        self.error = None
        try:
            self.client.connect(host, port, keepalive=30)
            self.client.loop_start()
        except (OSError, ValueError) as exc:
            self.error = str(exc)

    def send(self, message):
        if self.error:
            return False, self.error
        try:
            info = self.client.publish(self.topic, message, qos=1, retain=False)
            info.wait_for_publish(timeout=4)
            if not info.is_published():
                return False, "MQTT publish timeout"
            return True, None
        except (OSError, RuntimeError, ValueError) as exc:
            return False, str(exc)

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()


def read_image(raw):
    if len(raw) > MAX_BYTES:
        raise ValueError("Anh vuot qua 8 MiB.")
    try:
        with Image.open(io.BytesIO(raw)) as check:
            check.verify()
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 12_000_000:
                raise ValueError("Anh qua 12 megapixel.")
            return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("Khong doc duoc anh JPEG/PNG hop le") from exc


def build_app(args, predictor=None, bridge=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES
    CAPTURES.mkdir(parents=True, exist_ok=True)
    predictor = predictor or Predictor(args.weights, args.prepared, args.hand_model)
    bridge = bridge or MqttBridge(args.broker, args.broker_port, args.topic)
    lock = threading.Lock()
    state = {"seq": 0}

    def process(image_bytes, source):
        photo = read_image(image_bytes)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = stamp + "_" + source + ".jpg"
        photo.save(CAPTURES / name, "JPEG", quality=94)
        result = {
            "image_url": "/photos/" + name, "roi_url": None,
            "predicted": None, "scores": None,
            "mqtt_published": False, "mqtt_error": None,
        }
        try:
            roi, pred, best, scores = predictor.predict(photo)
            roi_name = stamp + "_roi.png"
            roi.save(CAPTURES / roi_name, "PNG")
            state["seq"] += 1
            payload = protocol_message(pred, best, state["seq"])
            published, error = bridge.send(payload)
            result.update({
                "roi_url": "/photos/" + roi_name,
                "predicted": f"person_{pred:02d}",
                "best_cosine": best,
                "scores": scores,
                "mqtt_published": published,
                "mqtt_error": error,
                "message": ("Da chup anh moi, da chay AI that tren PC. "
                            "MQTT chi gui index + cosine, KHONG gui anh/embedding."),
            })
            print(f"[REAL AI] {source}: person_{pred:02d} cosine={best:.4f} "
                  f"MQTT={'sent' if published else error}", flush=True)
        except (ValueError, RuntimeError, OSError) as exc:
            result["error"] = str(exc)
            result["message"] = "Anh moi da luu, nhung crop/AI loi. Khong gui MQTT."
            print("[AI ERROR]", exc, flush=True)
        return result

    @app.get("/")
    def home():
        return PAGE

    @app.get("/photos/<path:filename>")
    def photo(filename):
        return send_from_directory(str(CAPTURES), filename, max_age=0)

    @app.post("/api/image")
    def upload_image():
        f = request.files.get("file")
        if f is None:
            return jsonify(error="Thieu file"), 400
        data = f.stream.read(MAX_BYTES + 1)
        with lock:
            try:
                return jsonify(process(data, "webcam"))
            except ValueError as exc:
                return jsonify(error=str(exc)), 400

    @app.post("/api/esp32cam")
    def take_camera():
        if not args.camera_url:
            return jsonify(error="Chua khai bao --camera-url."), 400
        with lock:
            try:
                req = urllib.request.Request(args.camera_url, headers={
                    "Cache-Control": "no-cache", "User-Agent": "PalmLock/2.0"
                })
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.status != 200:
                        raise ValueError(f"Camera returned HTTP {response.status}")
                    data = response.read(MAX_BYTES + 1)
                return jsonify(process(data, "esp32cam"))
            except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
                return jsonify(error=str(exc)), 502

    @app.errorhandler(413)
    def large(_):
        return jsonify(error="Anh >8MiB"), 413

    return app


def parse_args():
    p = argparse.ArgumentParser(description="Real camera+AI on PC -> MQTT -> Wokwi")
    p.add_argument("--topic", required=True,
                   help="Unique MQTT topic. Set same text in wokwi_live/sketch.ino!")
    p.add_argument("--weights", type=Path,
                   default=ROOT / "weights/palmbridge_compact_roi64.pt")
    p.add_argument("--prepared", type=Path, default=ROOT / "dataset/esp32_prepared")
    p.add_argument("--hand-model", type=Path, default=DEFAULT_HAND_LANDMARKER)
    p.add_argument("--camera-url", default=None,
                   help="Optionally use a REAL ESP32-CAM /capture endpoint.")
    p.add_argument("--broker", default="test.mosquitto.org")
    p.add_argument("--broker-port", type=int, default=1883)
    p.add_argument("--port", type=int, default=8766)
    args = p.parse_args()
    if not args.topic.startswith("palmlock/demo/") or len(args.topic) > 128:
        p.error("--topic must begin with palmlock/demo/ and be <=128 chars.")
    if args.camera_url and not args.camera_url.startswith(("http://", "https://")):
        p.error("--camera-url must begin with http:// or https://.")
    return args


def main():
    args = parse_args()
    predictor = Predictor(args.weights, args.prepared, args.hand_model)
    bridge = MqttBridge(args.broker, args.broker_port, args.topic)
    app = build_app(args, predictor=predictor, bridge=bridge)
    print("\nPalmLock LIVE PC -> Wokwi: http://127.0.0.1:%d" % args.port, flush=True)
    print(f"Model: {args.weights.name}, input {predictor.input_size}x{predictor.input_size}", flush=True)
    print(f"Train centroid enroll: {args.prepared / 'train'}", flush=True)
    print(f"MQTT topic: {args.topic}; broker: {args.broker}:{args.broker_port}", flush=True)
    print("WARNING: Public MQTT demo ONLY. No biometric image/embedding is transmitted.", flush=True)
    print("WARNING: This DOES NOT test AI inference on the ESP32.", flush=True)
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False,
                use_reloader=False, threaded=False)
    finally:
        bridge.close()
        predictor.close()


if __name__ == "__main__":
    main()
