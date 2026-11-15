"""Custom CompNet experiment: train pool2, compare on same held-out photos, simulate INT8.

Run from project root; no biometric images/checkpoints are uploaded.
Do NOT interpret simulated weight-only INT8 as a deployable .espdl firmware.
"""
import argparse
import copy
import csv
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROOT, SUPPORTED_EXTENSIONS, DEFAULT_HAND_LANDMARKER
from finetune import ArcMarginHead
from models.checkpoint import load_encoder
from models.compact import initialize_compact_from_full
from preprocessing import build_transform, preprocess_file

DEFAULT_FULL = ROOT / "weights/palmbridge_esp32_finetuned.pt"
DEFAULT_COMPACT = ROOT / "weights/palmbridge_compact_pool2.pt"
DEFAULT_PREPARED = ROOT / "dataset/esp32_prepared"
DEFAULT_TESTS = ROOT / "test_photos"
RESULTS = ROOT / "experiments/results"


def loader(path):
    with Image.open(path) as im:
        return im.convert("L")


def datasets_for(prepared):
    transform = build_transform()
    ds = {s: ImageFolder(Path(prepared) / s, transform=transform, loader=loader)
          for s in ("train", "val")}
    if ds["train"].classes != ds["val"].classes or len(ds["train"].classes) != 4:
        raise ValueError("Train/val cần cùng 4 nhãn; không gộp ảnh test vào train.")
    return ds


@torch.inference_mode()
def centroids(model, loader, count, device):
    model.eval()
    sums = torch.zeros(count, 512, device=device)
    nums = torch.zeros(count, device=device)
    for x, y in loader:
        z = model.get_embedding(x.to(device))
        sums.index_add_(0, y.to(device), z)
        nums.index_add_(0, y.to(device), torch.ones(len(y), device=device))
    if (nums == 0).any():
        raise ValueError("Thiếu ảnh train trong một lớp.")
    return F.normalize(sums / nums[:, None], dim=1)


@torch.inference_mode()
def score_validation(model, centers, validation, device):
    good = total = 0
    for x, y in validation:
        z = model.get_embedding(x.to(device))
        best = (z @ centers.T).argmax(dim=1).cpu()
        good += int((best == y).sum())
        total += len(y)
    return good / total if total else 0.0


def save_checkpoint(model, path, epoch, val_acc, classes, origin):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "backbone": {k: v.detach().cpu() for k, v in model.backbone.state_dict().items()},
        "palmbridge": {k: v.detach().cpu() for k, v in model.palmbridge.state_dict().items()},
        "embedding_dim": 512, "image_size": 128,
        "backbone_type": "custom_compact", "pool_size": 2,
        "class_names": classes, "epoch": epoch,
        "val_centroid_accuracy": val_acc,
        "initialized_from": str(origin),
    }, path)


def cmd_train(args):
    if args.epochs < 1:
        raise ValueError("--epochs phải >0")
    train = datasets_for(args.prepared)
    device = torch.device(args.device)
    full, kind = load_encoder(args.full, device)
    if kind != "custom":
        raise ValueError(f"Checkpoint gốc phải custom 4x4; nhận {kind}")
    model = initialize_compact_from_full(full).to(device)
    del full
    for p in model.parameters():
        p.requires_grad_(False)
    for module in (model.backbone.block4, model.backbone.embed, model.palmbridge):
        for p in module.parameters():
            p.requires_grad_(True)
    head = ArcMarginHead(4).to(device)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
        {"params": head.parameters(), "lr": args.head_lr}
    ], weight_decay=1e-4)
    train_dl = DataLoader(train["train"], batch_size=args.batch_size,
                          shuffle=True, num_workers=0)
    train_eval = DataLoader(train["train"], batch_size=args.batch_size,
                           shuffle=False, num_workers=0)
    val_dl = DataLoader(train["val"], batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    history = []
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.eval()  # Small dataset: freeze batch-norm moving statistics.
        head.train()
        loss_sum = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            features = model.get_embedding(x)
            logits = head(features, y)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(y)
        centers = centroids(model, train_eval, 4, device)
        val_acc = score_validation(model, centers, val_dl, device)
        record = {"epoch": epoch, "train_arc_loss": round(loss_sum / len(train["train"]), 5),
                  "val_nearest_centroid_acc": round(val_acc, 5)}
        history.append(record)
        print(f"Epoch {epoch}/{args.epochs}: loss={record['train_arc_loss']:.4f}, "
              f"val cosine nearest-centroid={val_acc:.1%}", flush=True)
        if val_acc > best:
            best = val_acc
            save_checkpoint(model, args.compact, epoch, best, train["train"].classes, args.full)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "compact_train_history.json").write_text(
        json.dumps({"history": history, "best_val": best,
                    "selection": "validation nearest train centroid; test_photos NEVER used for checkpoint selection",
                    "warning": "Random-split val may be biased; obtain independent sessions."},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved best compact checkpoint:", args.compact)
    print("Now run: python experiments/benchmark.py compare")


def photo_items(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f"Không có ảnh test: {folder}")
    photos = []
    for person in sorted(p for p in folder.iterdir() if p.is_dir()):
        for p in sorted(person.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                photos.append((person.name, p))
    if not photos:
        raise ValueError(f"Không có ảnh trong {folder}")
    return photos


@torch.inference_mode()
def cmd_compare(args):
    device = torch.device(args.device)
    full, full_kind = load_encoder(args.full, device)
    small, small_kind = load_encoder(args.compact, device)
    if full_kind != "custom" or small_kind != "custom_compact":
        raise ValueError(f"Model kinds phải custom và custom_compact; có {full_kind}/{small_kind}")
    ds = datasets_for(args.prepared)
    train_dl = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=False, num_workers=0)
    classes = ds["train"].classes
    base_centers = centroids(full, train_dl, 4, device)
    small_centers = centroids(small, train_dl, 4, device)
    from roi import HandPalmROICropper
    cropper = (HandPalmROICropper(args.hand_model) if args.input_format == "full-hand" else None)
    rows = []
    counts = {"processed": 0, "failed": 0, "baseline_correct": 0, "compact_correct": 0}
    try:
        for expected, file in photo_items(args.tests):
            if expected not in classes:
                print(f"[SKIP] Không có nhãn này trong train: {expected}")
                continue
            row = {"image": str(file.relative_to(args.tests)), "expected": expected,
                   "baseline_pred": "", "compact_pred": "",
                   "baseline_best": "", "compact_best": "", "error": ""}
            try:
                x = preprocess_file(file, cropper=cropper).to(device)
                b = (full.get_embedding(x) @ base_centers.T)[0].cpu()
                c = (small.get_embedding(x) @ small_centers.T)[0].cpu()
                bi, ci = int(b.argmax()), int(c.argmax())
                row.update({
                    "baseline_pred": classes[bi], "compact_pred": classes[ci],
                    "baseline_best": round(float(b[bi]), 6),
                    "compact_best": round(float(c[ci]), 6),
                    "baseline_correct": classes[bi] == expected,
                    "compact_correct": classes[ci] == expected,
                })
                for i, name in enumerate(classes):
                    row["baseline_"+name] = round(float(b[i]), 6)
                    row["compact_"+name] = round(float(c[i]), 6)
                counts["processed"] += 1
                counts["baseline_correct"] += int(classes[bi] == expected)
                counts["compact_correct"] += int(classes[ci] == expected)
                print(f"{row['image']}: {expected} | goc={classes[bi]} "
                      f"{b[bi]:.3f} | compact={classes[ci]} {c[ci]:.3f}")
            except (ValueError, RuntimeError, OSError) as exc:
                counts["failed"] += 1
                row["error"] = str(exc)
                print(f"[ROI ERROR] {row['image']}: {exc}")
            rows.append(row)
    finally:
        if cropper:
            cropper.close()
    # Per-class breakdown and confusion matrices (critical for person_01/04).
    counts["per_person"] = {}
    counts["confusion"] = {"baseline": {}, "compact": {}}
    for person in classes:
        relevant = [row for row in rows if row["expected"] == person and not row["error"]]
        counts["per_person"][person] = {
            "total": len(relevant),
            "baseline_correct": sum(row["baseline_pred"] == person for row in relevant),
            "compact_correct": sum(row["compact_pred"] == person for row in relevant),
        }
        for mode, pred_key in (("baseline", "baseline_pred"), ("compact", "compact_pred")):
            counts["confusion"][mode][person] = {
                candidate: sum(row[pred_key] == candidate for row in relevant)
                for candidate in classes
            }
    n = counts["processed"]
    counts["baseline_accuracy"] = round(counts["baseline_correct"] / n, 5) if n else None
    counts["compact_accuracy"] = round(counts["compact_correct"] / n, 5) if n else None
    counts["note"] = ("Same held-out photos, same ROI, same enroll protocol (centroid from "
                      "prepared/train). This can differ from historical 13/32 DB-based score.")
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "compare_per_image.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["image", "expected", "baseline_pred", "compact_pred",
                  "baseline_best", "compact_best", "baseline_correct", "compact_correct"]
        fields += [f"{s}_{u}" for s in ("baseline", "compact") for u in classes]
        fields += ["error"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (RESULTS / "compare_summary.json").write_text(
        json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== SAME-IMAGE COMPARISON ===")
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    print("Results: experiments/results/compare_per_image.csv")


def weight_only_int8_sim(model):
    """Symmetric per-tensor quantization of learnable parameters only.

    Values are dequantized back to FLOAT32 for PyTorch execution: this is
    sensitivity testing, not a full INT8 activation runtime or .espdl conversion.
    """
    result = copy.deepcopy(model).eval()
    archive = {}
    spec = {}
    fpbytes = int(sum(p.numel() * p.element_size() for p in model.parameters()))
    qbytes = 0
    with torch.no_grad():
        for name, p in result.named_parameters():
            src = p.detach().float().cpu()
            scale = max(float(src.abs().max()) / 127.0, 1e-9)
            q = torch.clamp(torch.round(src / scale), -127, 127).to(torch.int8)
            p.copy_((q.float() * scale).to(p.device))
            archive[name] = q.numpy()
            spec[name] = {"scale": scale, "shape": list(q.shape)}
            qbytes += q.numel() + 4
    return result, archive, spec, fpbytes, qbytes


@torch.inference_mode()
def cmd_quantize(args):
    device = torch.device(args.device)
    model, kind = load_encoder(args.compact, device)
    qmodel, archive, spec, fpbytes, qbytes = weight_only_int8_sim(model)
    ds = datasets_for(args.prepared)
    train_dl = DataLoader(ds["train"], batch_size=args.batch_size, num_workers=0)
    val_dl = DataLoader(ds["val"], batch_size=args.batch_size, num_workers=0)
    fcenters = centroids(model, train_dl, 4, device)
    qcenters = centroids(qmodel, train_dl, 4, device)
    before = score_validation(model, fcenters, val_dl, device)
    after = score_validation(qmodel, qcenters, val_dl, device)
    all_cos = []
    start = time.perf_counter()
    for x, _ in val_dl:
        a = model.get_embedding(x.to(device))
        b = qmodel.get_embedding(x.to(device))
        all_cos.extend(F.cosine_similarity(a, b).cpu().tolist())
    sec = time.perf_counter() - start
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS / "compact_int8_weights_only.npz", **archive)
    report = {
        "model": kind,
        "parameter_fp32_bytes": fpbytes,
        "int8_parameter_payload_plus_scales_bytes": qbytes,
        "int8_archive_bytes": (RESULTS / "compact_int8_weights_only.npz").stat().st_size,
        "val_accuracy_fp32_centroid": round(before, 6),
        "val_accuracy_weight_only_quant_dequant": round(after, 6),
        "embedding_cosine_mean": float(np.mean(all_cos)),
        "embedding_cosine_min": float(np.min(all_cos)),
        "val_images": len(all_cos),
        "elapsed_seconds_for_pair": round(sec, 3),
        "important": ("NOT a deployed INT8 model: parameters were roundtrip "
                      "quantized, PyTorch activations and compute remain FP32. "
                      "Need ESP-PPQ w8a8, operator compatibility and real-board tests.")
    }
    (RESULTS / "int8_scales.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8")
    (RESULTS / "int8_proxy_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Exported weight-only int8 archive for size/sensitivity study; NOT ESP firmware.")


def cmd_size(args):
    device = torch.device("cpu")
    data = {}
    for name, file in (("original", args.full), ("compact", args.compact)):
        model, kind = load_encoder(file, device)
        params = sum(p.numel() for p in model.parameters())
        float32 = sum(p.numel() * p.element_size() for p in model.parameters())
        buffers = sum(b.numel() * b.element_size() for b in model.buffers())
        # Model's Gabor op makes 32 channels at 128x128 before pooling.
        gabor_bytes = 32 * 128 * 128 * 4
        data[name] = {
            "kind": kind,
            "parameter_count": params,
            "fp32_parameter_bytes": float32,
            "fp32_buffer_bytes": buffers,
            "weight_only_int8_lower_bound_bytes": sum(p.numel() for p in model.parameters()),
            "gabor_output_float32_bytes": gabor_bytes,
            "checkpoint_file_bytes": file.stat().st_size,
        }
    data["board"] = {"flash_bytes": 4 * 1024**2, "psram_bytes": 2 * 1024**2}
    data["limitations"] = [
        "INT8 parameter bytes are a lower bound, not executable firmware footprint.",
        "Gabor output FP32 is already 2 MiB, excluding inputs/other tensors.",
        "Total peak RAM requires ESP-DL memory planner or physical target profiling.",
        "MediaPipe HandLandmarker and dynamic Gabor kernels are not converted here.",
        "No .espdl firmware created; ESP-PPQ compatibility must be verified.",
    ]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "esp32_feasibility.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description="CompNet pool2 experimental benchmark")
    p.add_argument("task", choices=["train", "compare", "quantize", "size"])
    p.add_argument("--full", type=Path, default=DEFAULT_FULL)
    p.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    p.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    p.add_argument("--tests", type=Path, default=DEFAULT_TESTS)
    p.add_argument("--input-format", choices=["full-hand", "roi"], default="full-hand")
    p.add_argument("--hand-model", type=Path, default=DEFAULT_HAND_LANDMARKER)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()
    {"train": cmd_train, "compare": cmd_compare,
     "quantize": cmd_quantize, "size": cmd_size}[args.task](args)


if __name__ == "__main__":
    main()
