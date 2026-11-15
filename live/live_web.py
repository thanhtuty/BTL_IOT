"""PalmLock-AI LIVE: capture REAL images, run ORIGINAL PalmBridge on this PC.

Flow:
 ESP32-CAM (actual OV2640 JPEG) -> HTTP /capture -> Flask -> saved JPEG
 -> MediaPipe HandPalmROICropper -> existing CompNet+PalmBridge weights
 -> cosine against 4 enrolled templates -> results in local web UI.

No on-chip PyTorch/MediaPipe is claimed. No lock actuator is triggered.
Alternative capture: browser webcam or uploaded photo when no ESP32-CAM is available.
"""
import argparse
import io
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DEFAULT_HAND_LANDMARKER
from database import open_database, sha256_file
from inference import PalmEmbedder
from roi import HandPalmROICropper

CAPTURES = ROOT / "live_captures"
MAX_BYTES = 8 * 1024 * 1024

PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PalmLock-AI | Chụp ảnh thật</title>
<style>
body{background:#101728;color:#e8edf7;font:16px/1.5 system-ui,Arial,sans-serif;margin:0}
main{max-width:970px;padding:26px 16px;margin:auto}
h1{font-size:26px;margin-bottom:4px}p{color:#acbbd3}
section{background:#192439;border:1px solid #30415e;border-radius:14px;padding:18px;margin-top:16px}
button,select,input{font:inherit}button{background:#4275e8;border:0;border-radius:9px;color:white;padding:11px 14px;cursor:pointer;margin:4px}
button:disabled{opacity:.4;cursor:wait}select,input{border-radius:6px;padding:8px;background:#0d1523;color:white;border:1px solid #486183}
video,img{width:100%;max-width:640px;border-radius:9px;background:#101927}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
.muted{color:#aab5c7}.note{color:#ffd58b}.result{font-weight:700;font-size:20px}
.score{font-variant-numeric:tabular-nums;border-top:1px solid #36445b;padding:6px 0}
small{color:#aab5c7}
</style></head><body><main>
<h1>PalmLock-AI — chụp ảnh THẬT</h1>
<p>ESP32-CAM hoặc webcam → lưu JPEG → MediaPipe → CompNet → PalmBridge → cosine 4 người.</p>
<section>
<label>Nhãn thật để đối chiếu (tùy chọn):
<select id="expected"><option value="">Không xác định</option>
<option>person_01</option><option>person_02</option>
<option>person_03</option><option>person_04</option></select></label>
<div><button id="esp" onclick="captureESP()">📷 Chụp từ ESP32-CAM</button>
<button onclick="startWebcam()">Bật webcam trình duyệt</button>
<button onclick="captureWebcam()">Chụp webcam và nhận diện</button></div>
<video id="video" autoplay playsinline muted style="display:none"></video>
<div><label>Hoặc chọn ảnh: <input id="file" type="file" accept="image/*"></label>
<button onclick="uploadFile()">Phân tích ảnh đã chọn</button></div>
<p id="status" class="note">Sẵn sàng. Chỉ ESP32-CAM vật lý mới cung cấp ảnh OV2640 thật.</p>
</section>
<div class="grid">
<section><h3>Ảnh JPEG gốc (đã lưu)</h3><img id="raw" alt="Chưa có ảnh"><p><a id="download" download style="color:#8cb5ff"></a></p></section>
<section><h3>Ảnh ROI đã cắt để đưa vào AI</h3><img id="roi" alt="Chưa có ROI"></section>
</div>
<section><h3>Kết quả CompNet + PalmBridge</h3>
<div id="prediction" class="result">Chưa nhận diện</div>
<div id="scores"></div>
<p class="muted">Cosine × 100 được hiển thị dưới dạng %, KHÔNG phải xác suất mô hình đúng.
Không tự mở khóa; mô hình hiện chưa được hiệu chỉnh ngưỡng và kiểm thử người lạ.</p>
</section>
<script>
const statusEl=document.getElementById('status');
let cameraStream=null;
function expected(){return document.getElementById('expected').value;}
function report(d){
 const raw=document.getElementById('raw'),roi=document.getElementById('roi');
 if(d.image_url){
   raw.src=d.image_url+'?t='+Date.now();
   const a=document.getElementById('download');
   a.href=d.image_url;
   a.textContent='Tải ảnh gốc xuống máy';
 }
 if(d.roi_url){roi.src=d.roi_url+'?t='+Date.now();}
 else {roi.removeAttribute('src');}
 const pred=document.getElementById('prediction');
 pred.textContent=d.predicted
   ? ('Dự đoán: '+d.predicted+(d.top1_correct===null?'':(d.top1_correct?' — ĐÚNG':' — SAI')))
   : (d.ai_error?'Chưa trích xuất được đặc trưng':'Chỉ chụp ảnh (capture-only)');
 const scoresEl=document.getElementById('scores');
 scoresEl.textContent='';
 if(d.scores){
   d.scores.forEach(s=>{
     const line=document.createElement('div');
     line.className='score';
     line.textContent=s.user_id+': '+s.similarity_pct.toFixed(2)+'%';
     scoresEl.appendChild(line);
   });
 }
 statusEl.textContent=(d.image_url?'Đã lưu ảnh mới. ':'')+(d.ai_error||d.message||'Đã xử lý xong.');
}
async function handleResponse(response){
 let data;
 try {data=await response.json();}catch(err){throw new Error('Server không trả về JSON');}
 if(!response.ok){throw new Error(data.error||('HTTP '+response.status));}
 report(data);
}
async function captureESP(){
 statusEl.textContent='Đang yêu cầu ESP32-CAM chụp JPEG mới...';
 try {
   const res=await fetch('/api/capture?expected='+encodeURIComponent(expected()),{method:'POST'});
   await handleResponse(res);
 }catch(err){statusEl.textContent='Lỗi ESP32-CAM: '+err.message;}
}
async function startWebcam(){
 try {
   if(!cameraStream){cameraStream=await navigator.mediaDevices.getUserMedia({video:true,audio:false});}
   const vid=document.getElementById('video');
   vid.srcObject=cameraStream;vid.style.display='block';
   await vid.play();
   statusEl.textContent='Webcam đã bật. Đặt bàn tay vào khung rồi bấm Chụp webcam.';
 }catch(err){statusEl.textContent='Không bật được webcam: '+err.message;}
}
async function sendBlob(blob,name){
 const fd=new FormData();
 fd.append('file',blob,name);fd.append('expected',expected());
 const res=await fetch('/api/upload',{method:'POST',body:fd});
 await handleResponse(res);
}
async function captureWebcam(){
 try {
   const v=document.getElementById('video');
   if(!cameraStream||!v.videoWidth){await startWebcam(); if(!v.videoWidth) return;}
   const c=document.createElement('canvas');c.width=v.videoWidth;c.height=v.videoHeight;
   c.getContext('2d').drawImage(v,0,0);
   const blob=await new Promise(resolve=>c.toBlob(resolve,'image/jpeg',0.95));
   if(!blob)throw new Error('Không lấy được JPEG từ webcam');
   statusEl.textContent='Đã chụp khung webcam, đang chạy AI...';
   await sendBlob(blob,'webcam.jpg');
 }catch(err){statusEl.textContent='Lỗi: '+err.message;}
}
async function uploadFile(){
 const f=document.getElementById('file').files[0];
 if(!f){statusEl.textContent='Chọn ảnh trước.';return;}
 try {statusEl.textContent='Đang phân tích ảnh...';await sendBlob(f,f.name);}
 catch(err){statusEl.textContent='Lỗi: '+err.message;}
}
</script></main></body></html>"""


def parse_args():
    p = argparse.ArgumentParser(description="ESP32-CAM -> PC REAL live JPEG + PalmBridge")
    p.add_argument("--camera-url", default=None,
                   help="e.g. http://192.168.1.123/capture ; omit for webcam/upload")
    p.add_argument("--weights", type=Path,
                   default=ROOT / "weights" / "palmbridge_esp32_finetuned.pt")
    p.add_argument("--db", type=Path,
                   default=ROOT / "database" / "test_4people_12epoch.json")
    p.add_argument("--capture-only", action="store_true",
                   help="Only capture/save photo, do NOT load weights or MediaPipe")
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args()


def build_app(args):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES
    CAPTURES.mkdir(parents=True, exist_ok=True)
    working_lock = threading.Lock()

    embedder = cropper = templates = None
    if not args.capture_only:
        import torch
        import torch.nn.functional as F
        db = open_database(args.db, sha256_file(args.weights))
        if not db["users"]:
            raise ValueError("Database không có người. Enroll 4 người trước.")
        embedder = PalmEmbedder(args.weights, device="cpu", input_format="roi")
        cropper = HandPalmROICropper(DEFAULT_HAND_LANDMARKER)
        templates = {
            user: F.normalize(torch.tensor(info["template"], dtype=torch.float32), p=2, dim=0)
            for user, info in db["users"].items()
        }

    def analyze_photo(data, source, expected):
        if len(data) > MAX_BYTES:
            raise ValueError("Ảnh quá lớn (giới hạn 8 MiB).")
        try:
            with Image.open(io.BytesIO(data)) as original:
                original.verify()
            with Image.open(io.BytesIO(data)) as original:
                if original.width * original.height > 12_000_000:
                    raise ValueError("Ảnh quá 12 megapixel.")
                photo = ImageOps.exif_transpose(original).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError("Không đọc được ảnh JPEG/PNG hợp lệ") from exc

        # An image is always saved before attempting ROI/AI.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"{stamp}_{source}.jpg"
        path = CAPTURES / name
        photo.save(path, "JPEG", quality=94)
        result = {
            "image_url": "/photos/" + name,
            "roi_url": None,
            "source": source,
            "predicted": None,
            "top1_correct": None,
            "scores": None,
            "ai_error": None,
            "message": "Ảnh thật đã được chụp và lưu trên máy.",
        }
        if args.capture_only:
            return result
        try:
            _, roi, _ = cropper.crop_file(path)
            roi_name = f"{stamp}_{source}_roi.png"
            roi_path = CAPTURES / roi_name
            roi.save(roi_path, "PNG")
            result["roi_url"] = "/photos/" + roi_name
            embedding = embedder.extract_embedding(roi_path)
            import torch.nn.functional as F
            ranked = [
                {"user_id": user,
                 "similarity_pct": round(float(F.cosine_similarity(
                     embedding.unsqueeze(0), template.unsqueeze(0)).item()) * 100.0, 2)}
                for user, template in templates.items()
            ]
            ranked.sort(key=lambda item: item["similarity_pct"], reverse=True)
            best = ranked[0]["user_id"]
            result["predicted"] = best
            result["scores"] = ranked
            result["top1_correct"] = (best == expected) if expected in templates else None
            result["message"] = "Kết quả AI thật trên ảnh vừa chụp; KHÔNG điều khiển khóa."
        except (OSError, ValueError, RuntimeError) as exc:
            result["ai_error"] = str(exc)
            result["message"] = "Ảnh đã lưu, nhưng bước ROI/AI thất bại."
        return result

    @app.get("/")
    def home():
        return PAGE

    @app.get("/photos/<path:filename>")
    def photo(filename):
        return send_from_directory(str(CAPTURES), filename, max_age=0)

    @app.post("/api/capture")
    def capture_camera():
        if not args.camera_url:
            return jsonify(error="Chưa khai báo --camera-url. Vẫn có thể dùng webcam trình duyệt."), 400
        if not args.camera_url.startswith(("http://", "https://")):
            return jsonify(error="Camera URL phải bắt đầu http:// hoặc https://"), 400
        with working_lock:
            try:
                req = urllib.request.Request(
                    args.camera_url, headers={"User-Agent": "PalmLockLive/1.0", "Cache-Control": "no-cache"}
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.status != 200:
                        raise ValueError(f"ESP32-CAM trả HTTP {response.status}")
                    data = response.read(MAX_BYTES + 1)
                return jsonify(analyze_photo(data, "esp32cam", request.args.get("expected", "")))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                return jsonify(error=str(exc)), 502

    @app.post("/api/upload")
    def upload():
        uploaded = request.files.get("file")
        if uploaded is None:
            return jsonify(error="Thiếu file ảnh."), 400
        data = uploaded.stream.read(MAX_BYTES + 1)
        with working_lock:
            try:
                return jsonify(analyze_photo(data, "webcam", request.form.get("expected", "")))
            except (OSError, ValueError) as exc:
                return jsonify(error=str(exc)), 400

    @app.errorhandler(413)
    def too_big(_error):
        return jsonify(error="Ảnh lớn hơn 8 MiB"), 413

    return app


def main():
    args = parse_args()
    app = build_app(args)
    print("\nPalmLock LIVE - Web giao diện: http://127.0.0.1:%d" % args.port, flush=True)
    print("Camera URL: %s" % (args.camera_url or "không khai báo; dùng webcam/upload"), flush=True)
    print("AI: %s" % ("CAPTURE ONLY" if args.capture_only else "Original CompNet+PalmBridge on PC"), flush=True)
    print("Saved photos: %s" % CAPTURES, flush=True)
    print("No real lock will be opened by this program.", flush=True)
    app.run(host="127.0.0.1", port=args.port, debug=False,
            threaded=False, use_reloader=False)


if __name__ == "__main__":
    main()
