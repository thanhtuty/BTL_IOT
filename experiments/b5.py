"""B5 experiments for PalmLock Custom Compact: streamed Gabor + ROI 64x64.

All operations run on a PC with PyTorch. They do NOT create ESP32 firmware.
All generated checkpoint, biometric image score files remain git-ignored.
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_HAND_LANDMARKER, ROOT, SUPPORTED_EXTENSIONS
from experiments.benchmark import (
    ArcMarginHead, centroids, is_better_val, loader, validation_metrics,
)
from models.checkpoint import load_encoder
from models.streaming_gabor import as_streaming_encoder, streaming_gabor_compete
from preprocessing import build_transform

CHECKPOINT_128 = ROOT / "weights/palmbridge_compact_pool2.pt"
CHECKPOINT_64 = ROOT / "weights/palmbridge_compact_roi64.pt"
PREPARED = ROOT / "dataset/esp32_prepared"
TEST_PHOTOS = ROOT / "test_photos"
RESULTS = ROOT / "experiments/results"
CLASSES = tuple(f"person_{i:02d}" for i in range(1, 5))


def make_dataset(path, size):
    if size not in (64, 128):
        raise ValueError("size must be 64 or 128")
    ds = ImageFolder(path, transform=build_transform(size), loader=loader)
    if tuple(ds.classes) != CLASSES:
        raise ValueError(f"Expected {CLASSES}, found {ds.classes} in {path}")
    return ds


def data_loader(dataset, args, shuffle=False):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=0)


def require_128_model(path, device):
    model, kind = load_encoder(path, device)
    if kind != "custom_compact" or getattr(model, "input_size", 128) != 128:
        raise ValueError(f"Need 128x128 Compact checkpoint, got {kind}, "
                         f"size={getattr(model, 'input_size', None)}")
    return model


def require_64_model(path, device):
    model, kind = load_encoder(path, device)
    if kind != "custom_compact" or getattr(model, "input_size", 128) != 64:
        raise ValueError(f"Need 64x64 Compact checkpoint, got {kind}, "
                         f"size={getattr(model, 'input_size', None)}")
    return model


def save64(model, out, epoch, acc, margin, source):
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "backbone": {k: v.detach().cpu() for k, v in model.backbone.state_dict().items()},
        "palmbridge": {k: v.detach().cpu() for k, v in model.palmbridge.state_dict().items()},
        "backbone_type": "custom_compact",
        "embedding_dim": 512,
        "image_size": 64,
        "pool_size": 2,
        "epoch": epoch,
        "val_centroid_accuracy": float(acc),
        "val_cosine_margin": float(margin),
        "initialized_from": str(source),
        "note": "Experimental 64x64 ROI. Not ESP32 firmware.",
    }, out)


def train64(args):
    """Fine-tune 64 ROI with *already preprocessed ROI* train+val only."""
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    source = require_128_model(args.compact128, device)
    model = source  # identical parameter shapes; initialize from TRAINED Compact128
    model.input_size = 64
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for module in (model.backbone.block4, model.backbone.embed, model.palmbridge):
        for p in module.parameters():
            p.requires_grad_(True)
    head = ArcMarginHead(len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ], weight_decay=1e-4)

    train_ds = make_dataset(args.prepared / "train", 64)
    val_ds = make_dataset(args.prepared / "val", 64)
    train_dl = data_loader(train_ds, args, shuffle=True)
    train_eval_dl = data_loader(train_ds, args)
    val_dl = data_loader(val_ds, args)
    best_acc, best_margin, best_epoch = -1.0, float("-inf"), None
    history = []
    for ep in range(1, args.epochs + 1):
        model.eval()  # keep original BN moving averages fixed
        head.train()
        loss_sum = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            embedding = model.get_embedding(x)
            loss = F.cross_entropy(head(embedding, y), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            loss_sum += loss.item() * len(y)
        center = centroids(model, train_eval_dl, len(CLASSES), device)
        val_acc, val_margin = validation_metrics(model, center, val_dl, device)
        record = {
            "epoch": ep, "train_arc_loss": round(loss_sum / len(train_ds), 5),
            "val_nearest_centroid_acc": round(val_acc, 5),
            "val_cosine_margin": round(val_margin, 6),
        }
        history.append(record)
        print(f"ROI64 Epoch {ep}/{args.epochs}: loss={record['train_arc_loss']:.4f}, "
              f"val={val_acc:.1%}, margin={val_margin:.5f}", flush=True)
        if is_better_val(val_acc, val_margin, best_acc, best_margin):
            best_acc, best_margin, best_epoch = val_acc, val_margin, ep
            save64(model, args.compact64, ep, val_acc, val_margin, args.compact128)

    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(args.compact64),
        "initialized_from": str(args.compact128),
        "train_images": len(train_ds), "val_images": len(val_ds),
        "epochs": args.epochs, "selected_epoch": best_epoch,
        "selected_val_acc": best_acc, "selected_val_margin": best_margin,
        "input_size": 64, "seed": args.seed, "history": history,
        "warning": ("Validation has few samples and may contain same-session images; "
                    "test_photos is never used in training or checkpoint selection."),
    }
    (RESULTS / "b5_train64_history.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved best 64x64 checkpoint:", args.compact64)
    print(f"Selected epoch={best_epoch}, validation={best_acc:.1%}, margin={best_margin:.5f}")


def test_images(root):
    if not root.is_dir():
        raise FileNotFoundError(f"Missing held-out test directory: {root}")
    results = []
    for person in CLASSES:
        directory = root / person
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing test class folder: {directory}")
        for file in sorted(directory.rglob("*")):
            if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
                results.append((person, file))
    if not results:
        raise ValueError("No test images found")
    return results


@torch.inference_mode()
def compare64(args):
    """Evaluate 128 and 64 with identical ROI source; no Test used for training."""
    device = torch.device(args.device)
    m128 = require_128_model(args.compact128, device).eval()
    m64 = require_64_model(args.compact64, device).eval()
    ds128 = make_dataset(args.prepared / "train", 128)
    ds64 = make_dataset(args.prepared / "train", 64)
    c128 = centroids(m128, data_loader(ds128, args), 4, device)
    c64 = centroids(m64, data_loader(ds64, args), 4, device)
    tx128, tx64 = build_transform(128), build_transform(64)

    from roi import HandPalmROICropper
    cropper = (
        HandPalmROICropper(args.hand_model)
        if args.input_format == "full-hand" else None
    )
    samples = test_images(args.tests)
    rows, errors = [], []
    try:
        for expected, file in samples:
            item = {"image": str(file.relative_to(args.tests)),
                    "expected": expected,
                    "pred128": "", "pred64": "",
                    "score128": "", "score64": "", "error": ""}
            try:
                with Image.open(file) as img:
                    im = ImageOps.exif_transpose(img).convert("RGB")
                if cropper is not None:
                    roi, _ = cropper.crop(im)  # SINGLE ROI for BOTH versions
                else:
                    roi = im
                gray = roi.convert("L")
                x128 = tx128(gray).unsqueeze(0).to(device)
                x64 = tx64(gray).unsqueeze(0).to(device)
                s128 = (m128.get_embedding(x128) @ c128.T)[0].cpu()
                s64 = (m64.get_embedding(x64) @ c64.T)[0].cpu()
                i128, i64 = int(s128.argmax()), int(s64.argmax())
                item["pred128"], item["pred64"] = CLASSES[i128], CLASSES[i64]
                item["score128"], item["score64"] = round(float(s128[i128]), 6), round(float(s64[i64]), 6)
                item["correct128"], item["correct64"] = i128 == CLASSES.index(expected), i64 == CLASSES.index(expected)
                for i, person in enumerate(CLASSES):
                    item["s128_" + person] = round(float(s128[i]), 6)
                    item["s64_" + person] = round(float(s64[i]), 6)
                print(f"{item['image']} | expected {expected} | 128 {item['pred128']} "
                      f"({item['score128']:.3f}) | 64 {item['pred64']} ({item['score64']:.3f})")
            except (ValueError, RuntimeError, OSError, UnidentifiedImageError) as exc:
                item["error"] = str(exc)
                errors.append(item["image"])
                print(f"[ERROR] {item['image']}: {exc}")
            rows.append(item)
    finally:
        if cropper is not None:
            cropper.close()

    successful = [x for x in rows if not x["error"]]
    if not successful:
        raise RuntimeError("No successful Test samples")
    report = {"processed": len(successful), "failed": len(errors),
              "accuracy128": round(sum(x["correct128"] for x in successful) / len(successful), 5),
              "accuracy64": round(sum(x["correct64"] for x in successful) / len(successful), 5),
              "per_person": {}, "confusion": {"roi128": {}, "roi64": {}}}
    for person in CLASSES:
        group = [x for x in successful if x["expected"] == person]
        report["per_person"][person] = {
            "total": len(group),
            "roi128_correct": sum(x["pred128"] == person for x in group),
            "roi64_correct": sum(x["pred64"] == person for x in group),
        }
        for mode, key in (("roi128", "pred128"), ("roi64", "pred64")):
            report["confusion"][mode][person] = {
                who: sum(x[key] == who for x in group) for who in CLASSES
            }
    report["note"] = (
        "SAME images, SAME MediaPipe crop, DIFFERENT model input resolution. "
        "Both identities centroids recalculated from train ROI at matching resolution. "
        "Top-1 only; scores are cosine not posterior probabilities. "
        "Repeated use of this Test set for model decisions reduces independence."
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    fields = (["image", "expected", "pred128", "pred64",
               "score128", "score64", "correct128", "correct64"]
              + [f"{mode}_{who}" for mode in ("s128", "s64") for who in CLASSES]
              + ["error"])
    with (RESULTS / "b5_compare_per_image.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "b5_compare_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== COMPACT ROI 128 vs 64 ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))


@torch.inference_mode()
def stream_test(args):
    """Test exact intended layer behavior and embedding parity on a few ROI photos."""
    device = torch.device(args.device)
    base = require_128_model(args.compact128, device).eval()
    stream = as_streaming_encoder(base)
    dataset = make_dataset(args.prepared / "val", 128)
    indices = list(range(0, len(dataset), max(1, len(dataset) // args.samples)))[:args.samples]
    scores, g_errors, times = [], [], []
    for index in indices:
        x, label = dataset[index]
        x = x.unsqueeze(0).to(device)
        t0 = time.perf_counter()
        ref = base.get_embedding(x)
        times.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        out = stream.get_embedding(x)
        times.append(time.perf_counter() - t1)
        regular_g = base.backbone.compete(F.relu(base.backbone.gabor(x)))
        streamed_g = streaming_gabor_compete(base.backbone.gabor, x)
        gerr = float((regular_g - streamed_g).abs().max().cpu())
        cos = float(F.cosine_similarity(ref, out).item())
        scores.append(cos)
        g_errors.append(gerr)
        print(f"Validation ROI #{index} class={CLASSES[label]} | cosine original vs streamed="
              f"{cos:.8f}; Gabor maximum abs diff={gerr:.8g}")
    report = {
        "images": len(indices),
        "embedding_cosine_min": min(scores),
        "embedding_cosine_mean": sum(scores) / len(scores),
        "gabor_max_absolute_difference": max(g_errors),
        "elapsed_standard_seconds": round(sum(times[0::2]), 4),
        "elapsed_streamed_seconds": round(sum(times[1::2]), 4),
        "passed_tolerance": min(scores) > 0.9999 and max(g_errors) < 1e-4,
        "memory_estimate": memory_estimate(),
        "warning": ("Inference-only PyTorch proof of equivalence, NOT measured ESP32 "
                    "RAM or speed. Streaming can be slower on CPU."),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "b5_stream_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed_tolerance"]:
        raise RuntimeError("Streaming Gabor mismatch; do not port until investigated.")


def memory_estimate():
    data = {}
    for res in (128, 64):
        full = 32 * res * res * 4
        stream = 2 * res * res * 4
        data[str(res)] = {
            "full_gabor_activation_bytes": full,
            "streamed_gabor_two_activation_buffers_bytes": stream,
            "estimated_reduction_percent": 100.0 * (full - stream) / full,
        }
    return {
        "gabor_activation_fp32_only": data,
        "not_included": [
            "original image", "Gabor coefficients and kernel-building workspace",
            "convolution scratch", "CNN feature maps", "PalmBridge parameters",
            "MediaPipe detector", "WiFi/camera buffers", "ESP32 heap fragmentation"
        ],
        "note": ("Memory savings are theoretical for the Gabor activation only. "
                 "Two 1-channel buffers needed for running max and current response; "
                 "not an ESP32 whole-model peak RAM measurement."),
    }


def mem_report(_args):
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = memory_estimate()
    (RESULTS / "b5_gabor_memory_estimate.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description="B5 streaming-Gabor and ROI64 experiment")
    p.add_argument("task", choices=["memory", "stream-test", "train64", "compare64"])
    p.add_argument("--compact128", type=Path, default=CHECKPOINT_128)
    p.add_argument("--compact64", type=Path, default=CHECKPOINT_64)
    p.add_argument("--prepared", type=Path, default=PREPARED)
    p.add_argument("--tests", type=Path, default=TEST_PHOTOS)
    p.add_argument("--hand-model", type=Path, default=DEFAULT_HAND_LANDMARKER)
    p.add_argument("--input-format", choices=["full-hand", "roi"], default="full-hand")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.samples < 1:
        p.error("--epochs, --batch-size, --samples must be >= 1")
    {"memory": mem_report, "stream-test": stream_test,
     "train64": train64, "compare64": compare64}[args.task](args)


if __name__ == "__main__":
    main()
