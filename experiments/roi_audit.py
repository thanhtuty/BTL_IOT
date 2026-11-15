"""ROI and dataset audit. Does NOT modify source images or checkpoints.
Previews and diagnostics are private/local (experiments/results ignored by Git).
"""
import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT, SUPPORTED_EXTENSIONS, DEFAULT_HAND_LANDMARKER


def images(folder):
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dhash(image):
    gray = image.convert("L").resize((9, 8))
    a = np.asarray(gray, dtype=np.int16)
    bits = a[:, 1:] > a[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def stats(image):
    a = np.asarray(image.convert("L").resize((128, 128)), dtype=np.float32)
    return {
        "mean": round(float(np.mean(a)), 2),
        "std": round(float(np.std(a)), 2),
        "black_fraction": round(float(np.mean(a <= 5)), 4),
        "white_fraction": round(float(np.mean(a >= 250)), 4),
    }


def main():
    p = argparse.ArgumentParser(description="Audit ROI crops, labels and duplicate leakage")
    p.add_argument("--prepared", type=Path, default=ROOT / "dataset/esp32_prepared")
    p.add_argument("--tests", type=Path, default=ROOT / "test_photos")
    p.add_argument("--test-format", choices=["full-hand", "roi"], default="full-hand")
    p.add_argument("--hand-model", type=Path, default=DEFAULT_HAND_LANDMARKER)
    p.add_argument("--max-per-person", type=int, default=5)
    p.add_argument("--near-duplicate-distance", type=int, default=3)
    args = p.parse_args()
    if args.max_per_person < 1:
        raise ValueError("max-per-person >= 1")
    labels = [f"person_{i:02d}" for i in range(1,5)]
    for split in ("train", "val"):
        for label in labels:
            if not (args.prepared / split / label).is_dir():
                raise FileNotFoundError(f"Thiếu thư mục: {args.prepared / split / label}")
    if not args.tests.is_dir():
        raise FileNotFoundError(f"Không có bộ test giữ riêng: {args.tests}")

    from roi import HandPalmROICropper
    cropper = HandPalmROICropper(args.hand_model) if args.test_format == "full-hand" else None
    records, failures, previews = [], [], []
    try:
        for split in ("train", "val", "test"):
            for label in labels:
                folder = args.tests / label if split == "test" else args.prepared / split / label
                files = images(folder)
                for index, path in enumerate(files):
                    try:
                        with Image.open(path) as img:
                            im = ImageOps.exif_transpose(img).convert("RGB")
                        if split == "test" and cropper is not None:
                            roi, quad = cropper.crop(im)
                        else:
                            roi = im
                        info = {
                            "path": str(path), "split": split, "label": label,
                            "sha256": digest(path),
                            "dhash": hex(dhash(roi)),
                            **stats(roi),
                        }
                        # Very low texture or large padding are warnings for manual review.
                        info["suspect_roi"] = (
                            info["std"] < 12 or
                            info["black_fraction"] > 0.20 or
                            info["white_fraction"] > 0.20
                        )
                        records.append(info)
                        if index < args.max_per_person:
                            previews.append((f"{split} / {label} / {path.stem}", roi.copy()))
                    except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as exc:
                        failures.append({"path": str(path), "error": str(exc)})
    finally:
        if cropper is not None:
            cropper.close()

    cross_exact = []
    bysha = defaultdict(list)
    for item in records:
        bysha[item["sha256"]].append(item)
    for same in bysha.values():
        if len({(it["split"],it["label"]) for it in same}) > 1:
            cross_exact.append([{"split": x["split"],"label": x["label"],
                                 "path": x["path"]} for x in same])

    # dHash is only a heuristic: hand images may resemble one another.
    near = []
    prepared = [x for x in records if x["split"] == "train"]
    others = [x for x in records if x["split"] in ("val", "test")]
    for a in prepared:
        for b in others:
            if a["label"] != b["label"]:
                continue
            distance = (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count()
            if distance <= args.near_duplicate_distance:
                near.append({"train": a["path"], "other": b["path"],
                             "distance": distance})
    near = sorted(near, key=lambda it: it["distance"])[:80]

    report = {
        "counts": {split: {label: sum(x["split"]==split and x["label"]==label
                                      for x in records) for label in labels}
                   for split in ("train", "val", "test")},
        "failed": failures,
        "suspect_roi": [
            {"split": x["split"],"label": x["label"],"path": x["path"],
             "std": x["std"],"black_fraction": x["black_fraction"],
             "white_fraction": x["white_fraction"]}
            for x in records if x["suspect_roi"]
        ],
        "exact_duplicates_across_splits_or_labels": cross_exact,
        "possible_near_duplicates_train_vs_val_test": near,
        "interpretation": (
            "dHash flags only POSSIBLE near-duplicates (can be false positives). "
            "Compare preview images manually; correct ROI cannot be proven by statistics."
        ),
        "recommendation": ("Inspect person_01 vs person_04 ROI orientation, "
                           "left/right hand, illumination and label correctness. "
                           "Do not use randomly adjacent frames as independent test."),
    }
    output = ROOT / "experiments/results"
    output.mkdir(parents=True, exist_ok=True)
    (output / "roi_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    if previews:
        thumb, text_h = 160, 37
        columns = 5
        rows = math.ceil(len(previews) / columns)
        sheet = Image.new("RGB", (columns * thumb, rows * (thumb + text_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for i,(label,roi) in enumerate(previews):
            x=(i%columns)*thumb
            y=(i//columns)*(thumb+text_h)
            sheet.paste(roi.convert("RGB").resize((thumb,thumb)), (x,y))
            draw.text((x+2,y+thumb+2),label[:28],fill="black")
        sheet.save(output / "roi_contact_sheet.jpg",quality=88)
    print(json.dumps({
        "counts": report["counts"],
        "failed": len(failures), "suspect_roi": len(report["suspect_roi"]),
        "exact_cross_duplicates": len(cross_exact),
        "possible_near_duplicates": len(near),
        "files": ["experiments/results/roi_audit.json",
                  "experiments/results/roi_contact_sheet.jpg"],
    }, ensure_ascii=False, indent=2))
    print("Hãy MỞ roi_contact_sheet.jpg để so sánh Person 01 và Person 04.")
    print("KHÔNG commit file ảnh đầu ra lên GitHub.")


if __name__ == "__main__":
    main()
