"""B5 regression checks. Only randomly initialized synthetic models/images.

No private palm photos/checkpoints are required. Full integration comparison
must be run locally with --prepared and --tests.
"""
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from models.compact import CustomCompactBackbone
from models.palmbridge import PalmBridgeEncoder
from models.checkpoint import load_encoder
from models.streaming_gabor import as_streaming_encoder, streaming_gabor_compete
from preprocessing import build_transform
from experiments.b5 import memory_estimate, is_better_val


class B5Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gabor_streaming_matches_regular_competitive_relu(self):
        torch.manual_seed(11)
        model = CustomCompactBackbone().eval()
        x = torch.randn(1, 1, 16, 16)
        with torch.inference_mode():
            reference = model.compete(F.relu(model.gabor(x)))
            streamed = streaming_gabor_compete(model.gabor, x)
        self.assertEqual(tuple(streamed.shape), (1, 1, 16, 16))
        torch.testing.assert_close(reference, streamed, rtol=1e-4, atol=1e-5)

    def test_end_to_end_embedding_parity(self):
        torch.manual_seed(12)
        regular = PalmBridgeEncoder(CustomCompactBackbone()).eval()
        streamed = as_streaming_encoder(regular).eval()
        with torch.inference_mode():
            x = torch.rand(1, 1, 64, 64)
            a = regular.get_embedding(x)
            b = streamed.get_embedding(x)
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
        self.assertEqual(tuple(b.shape), (1, 512))

    def test_64_checkpoint_round_trip_without_damaging_128(self):
        regular = PalmBridgeEncoder(CustomCompactBackbone()).eval()
        with tempfile.TemporaryDirectory() as folder:
            checkpoint64 = Path(folder) / "b5_64.pt"
            checkpoint128 = Path(folder) / "b5_128.pt"
            for size, path in ((64, checkpoint64), (128, checkpoint128)):
                torch.save({
                    "backbone": regular.backbone.state_dict(),
                    "palmbridge": regular.palmbridge.state_dict(),
                    "embedding_dim": 512,
                    "image_size": size,
                }, path)
                model, kind = load_encoder(path)
                self.assertEqual(kind, "custom_compact")
                self.assertEqual(model.input_size, size)
        self.assertEqual(tuple(build_transform(64).transforms[0].size), (64, 64))
        self.assertEqual(tuple(build_transform().transforms[0].size), (128, 128))

    def test_gabor_buffer_memory_estimate(self):
        estimate = memory_estimate()["gabor_activation_fp32_only"]
        self.assertEqual(estimate["128"]["full_gabor_activation_bytes"], 2097152)
        self.assertEqual(estimate["128"]["streamed_gabor_two_activation_buffers_bytes"], 131072)
        self.assertEqual(estimate["64"]["full_gabor_activation_bytes"], 524288)
        self.assertEqual(estimate["64"]["streamed_gabor_two_activation_buffers_bytes"], 32768)


if __name__ == "__main__":
    unittest.main()
