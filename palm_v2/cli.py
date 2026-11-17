"""Run with: python -m palm_v2.cli <subcommand>."""
import argparse
import json
from pathlib import Path
from .data import dataset_counts
from .model import TinyPalmNet


def main():
    parser = argparse.ArgumentParser(description="PalmLock V2 four-person toolkit")
    parser.add_argument("--data", default="dataset_v2", help="Dataset root")
    parser.add_argument("--checkpoint", default="palm_v2/artifacts/best_model.pt")
    parser.add_argument("--templates", default="palm_v2/artifacts/templates.npz")
    parser.add_argument("--thresholds", default="palm_v2/artifacts/thresholds.json")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("inspect-data", help="Count session photos for each person and unknown")
    train_parser = sub.add_parser("train", help="Train the 4-class tiny model")
    train_parser.add_argument("--epochs", type=int, default=40)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--lr", type=float, default=0.002)
    train_parser.add_argument("--device", default=None, help="cpu or cuda")
    sub.add_parser("enroll", help="Build four mean embeddings from session_01")
    sub.add_parser("calibrate", help="Calibrate cosine threshold using session_02")
    verification = sub.add_parser("verify", help="Verify one new ROI image")
    verification.add_argument("--image", required=True)
    sub.add_parser("evaluate", help="Blind test on session_03")
    exporter = sub.add_parser("export", help="Export floats for ESP32 C++ runtime")
    exporter.add_argument("--output", default="palm_v2/firmware/generated_palm_v2.h")
    args = parser.parse_args()

    if args.action == "inspect-data":
        result = dataset_counts(args.data)
    elif args.action == "train":
        from .training import train
        result = train(args.data, args.checkpoint, epochs=args.epochs,
                       batch_size=args.batch_size, learning_rate=args.lr, device=args.device)
    elif args.action == "enroll":
        from .pipeline import enrolled_templates
        result = enrolled_templates(args.checkpoint, args.data, args.templates)
    elif args.action == "calibrate":
        from .pipeline import calibration
        result = calibration(args.checkpoint, args.templates, args.data, args.thresholds)
    elif args.action == "verify":
        from .pipeline import verify_image
        result = verify_image(args.image, args.checkpoint, args.templates, args.thresholds)
    elif args.action == "evaluate":
        from .pipeline import evaluate_test
        result = evaluate_test(args.checkpoint, args.templates, args.thresholds, args.data)
    elif args.action == "export":
        from .export_c import export_header
        result = export_header(args.checkpoint, args.templates, args.thresholds, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
