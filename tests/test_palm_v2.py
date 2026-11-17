import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from palm_v2.model import TinyPalmNet, PERSON_IDS, make_gabor_bank
from palm_v2.data import normalize_roi, samples_for_split
from palm_v2.export_c import export_header
from palm_v2.pipeline import sha256


class PalmV2Tests(unittest.TestCase):
    def test_forward_and_unit_embeddings(self):
        model = TinyPalmNet().eval()
        with torch.no_grad():
            embedding, logits = model(torch.randn(2, 1, 64, 64))
        self.assertEqual(tuple(embedding.shape), (2, 128))
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertTrue(torch.allclose(torch.linalg.norm(embedding, dim=1),
                                       torch.ones(2), atol=1e-5))

    def test_gabor_and_roi(self):
        self.assertEqual(tuple(make_gabor_bank().shape), (4, 1, 7, 7))
        roi = np.full((64, 64), 127, dtype=np.uint8)
        self.assertTrue(np.isfinite(normalize_roi(roi)).all())
        with self.assertRaises(ValueError):
            normalize_roi(np.zeros((64, 64), dtype=np.uint8))

    def test_explicit_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "person_01" / "session_03"
            folder.mkdir(parents=True)
            (folder / "sample.png").write_bytes(b"dummy")
            self.assertEqual(len(samples_for_split(tmp, "train")), 0)
            self.assertEqual(len(samples_for_split(tmp, "test")), 1)

    def test_export_header(self):
        with tempfile.TemporaryDirectory() as folder:
            d = Path(folder)
            checkpoint, templates = d / "best.pt", d / "centroids.npz"
            thresholds, output = d / "thresholds.json", d / "generated.h"
            model = TinyPalmNet().eval()
            torch.save({"model_state": model.state_dict(), "people": list(PERSON_IDS),
                        "input_side": 64, "embedding_dim": 128}, checkpoint)
            centers = np.eye(4, 128, dtype=np.float32)
            np.savez_compressed(templates, centroids=centers,
                                people=np.asarray(PERSON_IDS),
                                model_sha256=np.array(sha256(checkpoint)))
            thresholds.write_text(json.dumps({
                "checkpoint_sha256": sha256(checkpoint),
                "templates_sha256": sha256(templates),
                "threshold": 0.5,
            }), encoding="utf8")
            result = export_header(checkpoint, templates, thresholds, output)
            self.assertTrue(output.is_file())
            self.assertIn("dw4_w", output.read_text(encoding="utf8"))
            self.assertGreater(result["size_bytes"], 1000)


if __name__ == "__main__":
    unittest.main()
