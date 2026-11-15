"""Smoke tests for experimental pool2 architecture. No private dataset required."""
import tempfile
import unittest
from pathlib import Path

import torch
from models.backbone import CustomCompNetBackbone
from models.palmbridge import PalmBridgeEncoder
from models.compact import CustomCompactBackbone, initialize_compact_from_full
from models.checkpoint import load_encoder


class TestCustomCompact(unittest.TestCase):
    def test_fc_reduction_and_checkpoint_compatibility(self):
        original = PalmBridgeEncoder(CustomCompNetBackbone()).eval()
        compact = initialize_compact_from_full(original).eval()
        self.assertEqual(tuple(original.backbone.embed[1].weight.shape), (512, 4096))
        self.assertEqual(tuple(compact.backbone.embed[1].weight.shape), (512, 1024))
        self.assertEqual(
            original.backbone.embed[1].weight.numel() -
            compact.backbone.embed[1].weight.numel(),
            1572864,
        )
        self.assertTrue(torch.equal(
            original.palmbridge.P, compact.palmbridge.P
        ))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic_compact.pt"
            torch.save({
                "backbone": compact.backbone.state_dict(),
                "palmbridge": compact.palmbridge.state_dict(),
                "embedding_dim": 512, "image_size": 128,
            }, path)
            restored, kind = load_encoder(path)
            self.assertEqual(kind, "custom_compact")
            self.assertTrue(torch.equal(
                restored.backbone.embed[1].weight, compact.backbone.embed[1].weight
            ))

    def test_pool2_shape(self):
        torch.manual_seed(5)
        model = PalmBridgeEncoder(CustomCompactBackbone()).eval()
        with torch.inference_mode():
            vec = model(torch.rand(1, 1, 128, 128))
        self.assertEqual(tuple(vec.shape), (1, 512))
        self.assertAlmostEqual(float(vec.norm()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
