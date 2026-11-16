"""Offline regression tests for Wokwi score exporter (no real palm photos)."""
import csv
import tempfile
import unittest
from pathlib import Path

from wokwi.export_scores import choose_source, convert, quantize_score


class WokwiExporterTests(unittest.TestCase):
    def test_score_units(self):
        self.assertEqual(quantize_score("0.545821", "cosine"), 5458)
        self.assertEqual(quantize_score("54.5821", "percent"), 5458)
        self.assertEqual(quantize_score("-0.01", "cosine"), -100)
        with self.assertRaises(ValueError):
            quantize_score("1.2", "cosine")

    def test_chooses_roi64_on_b5_csv(self):
        columns = ["image", "expected", "error"]
        columns += [f"s64_person_{i:02d}" for i in range(1, 5)]
        columns += [f"s128_person_{i:02d}" for i in range(1, 5)]
        self.assertEqual(choose_source(columns, "auto")[0], "roi64")
        self.assertEqual(choose_source(columns, "roi128")[0], "roi128")

    def test_exports_b5_scores_without_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "b5.csv"
            output = Path(tmp) / "demo_scores.h"
            fieldnames = ["image", "expected", "error"]
            fieldnames += [f"s64_person_{i:02d}" for i in range(1, 5)]
            with source.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({
                    "image": "person_01\\palm_01.jpg",
                    "expected": "person_01",
                    "error": "",
                    "s64_person_01": "0.01",
                    "s64_person_02": "0.02",
                    "s64_person_03": "-0.03",
                    "s64_person_04": "0.325247",
                })
            convert(source, output, "roi64")
            text = output.read_text(encoding="utf-8")
            self.assertIn('"person_01/palm_01.jpg", 1, {100, 200, -300, 3252}', text)
            self.assertIn("Source: roi64", text)


if __name__ == "__main__":
    unittest.main()
