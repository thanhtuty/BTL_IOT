"""Optional offline ROI extraction from phone/full-hand photos.

This reuses the V1 repository's MediaPipe HandPalmROICropper.
ESP32 firmware does NOT perform this landmark detection, so camera framing
must be calibrated separately to match the training ROI before real use.
"""
import hashlib
import json
from pathlib import Path
from .data import IMAGE_EXTENSIONS
from .model import PERSON_IDS


def prepare_full_hand(source, destination, landmarker_path, overwrite=False):
    from roi import HandPalmROICropper
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate independent folders")
    if not source.is_dir():
        raise ValueError(f"Raw photo directory not found: {source}")
    cropper = HandPalmROICropper(landmarker_path, output_size=256)
    report = {"converted": 0, "skipped": 0, "failed": 0, "errors": []}
    try:
        for person in (*PERSON_IDS, "unknown"):
            for session in ("session_01", "session_02", "session_03"):
                folder = source / person / session
                if not folder.exists():
                    continue
                paths = sorted(p for p in folder.rglob("*")
                               if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
                for path in paths:
                    relative = path.relative_to(folder)
                    suffix = hashlib.sha1(str(relative).encode("utf8")).hexdigest()[:8]
                    target = destination / person / session / relative.parent / (relative.stem + "_" + suffix + ".png")
                    if target.exists() and not overwrite:
                        report["skipped"] += 1
                        continue
                    try:
                        _, cropped, _ = cropper.crop_file(path)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        cropped.convert("L").save(target, format="PNG")
                        report["converted"] += 1
                    except Exception as exc:
                        report["failed"] += 1
                        report["errors"].append({"file": str(path), "reason": str(exc)})
    finally:
        cropper.close()
    return report
