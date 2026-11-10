"""Unit tests KHÔNG cần checkpoint thật; kiểm thử kiến trúc/tiền xử lý."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from config import IMAGE_SIZE
from models.backbone import CustomCompNetBackbone, OfficialCompNetBackbone
from models.palmbridge import PalmBridgeEncoder
from models.checkpoint import load_encoder
from preprocessing import preprocess_file


class TestPalmLock(unittest.TestCase):
    def test_preprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'sample.png'
            a=np.full((160,180),80,dtype=np.uint8)
            a[30:130,30:150]=np.linspace(30,210,120).astype('uint8')[None,:]
            Image.fromarray(a).save(p)
            x=preprocess_file(p)
            self.assertEqual(tuple(x.shape),(1,1,IMAGE_SIZE,IMAGE_SIZE))
            self.assertTrue(torch.isfinite(x).all().item())

    def _check_backbone(self, backbone):
        encoder=PalmBridgeEncoder(backbone).eval()
        x=torch.randn(1,1,128,128)
        with torch.inference_mode():
            z=encoder(x)
        self.assertEqual(tuple(z.shape),(1,512))
        self.assertAlmostEqual(z.norm().item(),1.0,places=4)

    def test_custom(self):
        self._check_backbone(CustomCompNetBackbone())

    def test_official(self):
        self._check_backbone(OfficialCompNetBackbone())

    def test_checkpoint_custom(self):
        encoder=PalmBridgeEncoder(CustomCompNetBackbone()).eval()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'TEST_ONLY_RANDOM.pt'
            torch.save({'backbone_type':'custom','image_size':128,'embedding_dim':512,
                        'backbone':encoder.backbone.state_dict(),
                        'palmbridge':encoder.palmbridge.state_dict()},p)
            loaded,kind=load_encoder(p)
            self.assertEqual(kind,'custom')
            with torch.inference_mode():
                z=loaded(torch.randn(1,1,128,128))
            self.assertEqual(tuple(z.shape),(1,512))


if __name__=='__main__':
    unittest.main()
