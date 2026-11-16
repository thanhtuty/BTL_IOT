"""Convert measured PalmLock cosine similarities into Wokwi UART replay samples.

Supported CSV:
  - experiments/results/b5_compare_per_image.csv (ROI 64 or 128)
  - experiments/results/compare_per_image.csv (Compact or baseline)
  - test_results/score_per_image.csv (legacy similarity percentages)

This is REPLAY of PC scores, never real camera recognition or ESP32 AI.
No raw palm images are copied to the Wokwi sketch.
"""
import argparse
import csv
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PEOPLE = [f"person_{i:02d}" for i in range(1, 5)]
SOURCE_FIELDS = {
    "roi64": ("s64_", "cosine"),
    "roi128": ("s128_", "cosine"),
    "compact": ("compact_", "cosine"),
    "baseline": ("baseline_", "cosine"),
    "legacy": ("", "percent"),
}


def choose_source(headers, requested):
    headers = set(headers or [])
    candidates = [requested] if requested != "auto" else [
        "roi64", "compact", "legacy", "roi128", "baseline"
    ]
    for source in candidates:
        prefix, unit = SOURCE_FIELDS[source]
        cols = [f"{prefix}{person}" if unit == "cosine"
                else f"{person}_similarity_pct" for person in PEOPLE]
        if all(col in headers for col in cols):
            return source, cols, unit
    raise ValueError(
        f"Could not find score columns for source={requested}. "
        "Use --source roi64/roi128/compact/baseline/legacy. "
        f"CSV columns: {sorted(headers)}"
    )


def quantize_score(text, unit):
    try:
        value = Decimal(str(text))
        if not value.is_finite():
            raise ValueError("NaN or infinite score")
        if unit == "percent":
            value /= 100
        if value < -1 or value > 1:
            raise ValueError(f"Cosine outside [-1,1]: {value}")
        return int((value * 10000).to_integral_value(rounding=ROUND_HALF_UP))
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"Invalid cosine score {text!r}") from exc


def convert(source: Path, output: Path, score_source="auto"):
    rows = []
    with source.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        chosen, columns, unit = choose_source(reader.fieldnames, score_source)
        for row in reader:
            if row.get("error"):
                continue
            expected = row.get("expected", "")
            if expected not in PEOPLE:
                continue
            label = str(row.get("image", ""))
            if not re.fullmatch(r"[a-zA-Z0-9_./\\-]{1,120}", label):
                raise ValueError(f"Unsafe file name: {label!r}")
            if any(not row.get(col, "").strip() for col in columns):
                raise ValueError("Missing score columns for " + label)
            scores = [quantize_score(row[col], unit) for col in columns]
            rows.append((label.replace("\\", "/"), int(expected[-2:]), scores))
    if not rows:
        raise ValueError("No labeled valid test rows.")

    items = []
    for name, expected, scores in rows:
        items.append("  {\"" + name + "\", " + str(expected) + ", {" +
                     ", ".join(str(i) for i in scores) + "}}")
    output_text = (
        "#pragma once\n#include <Arduino.h>\n"
        "// REPLAY of real PC cosine scores, NOT camera or on-device AI inference.\n"
        "// Source: " + chosen + ". Scores: cosine * 10000.\n"
        "struct DemoSample { const char *file; uint8_t expected; int16_t score[4]; };\n"
        "static const DemoSample DEMO_SAMPLES[] = {\n" +
        ",\n".join(items) + "\n};\n"
        "static const size_t DEMO_COUNT = sizeof(DEMO_SAMPLES)/sizeof(DEMO_SAMPLES[0]);\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(output_text, encoding="utf-8")
    correct = sum(scores.index(max(scores)) + 1 == expected
                  for _, expected, scores in rows)
    print(f"Source: {chosen}")
    print(f"Exported {len(rows)} measured score vectors into {output}")
    print(f"Nearest-centroid top-1 on replay data: {correct}/{len(rows)} = {correct/len(rows):.1%}")
    print("REPLAY ONLY! Not neural inference or a camera simulation.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path,
                        default=ROOT / "test_results/score_per_image.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "wokwi/demo_scores.h")
    parser.add_argument("--source", choices=["auto", *SOURCE_FIELDS],
                        default="auto",
                        help="auto prefers ROI64 when present; select roi128 to compare.")
    args = parser.parse_args()
    convert(args.csv, args.output, args.source)


if __name__ == "__main__":
    main()
