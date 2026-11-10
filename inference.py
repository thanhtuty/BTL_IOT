"""API trích xuất đặc trưng: PC/Kaggle, không chạy trên ESP32-S3."""
from pathlib import Path
import torch
from config import DEFAULT_HAND_LANDMARKER
from preprocessing import build_transform, preprocess_file
from models.checkpoint import load_encoder


class PalmEmbedder:
    def __init__(self, weights_path, device=None, input_format="roi",
                 hand_model_path=DEFAULT_HAND_LANDMARKER,
                 roi_width_scale=1.30, roi_height_scale=1.50):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model, self.backbone_type = load_encoder(weights_path, self.device)
        self.transform = build_transform()
        self.weights_path = Path(weights_path)
        if input_format not in {"roi", "full-hand"}:
            raise ValueError("input_format chỉ nhận 'roi' hoặc 'full-hand'.")
        self.roi_cropper = None
        if input_format == "full-hand":
            from roi import HandPalmROICropper
            self.roi_cropper = HandPalmROICropper(
                hand_model_path,
                width_scale=roi_width_scale,
                height_scale=roi_height_scale,
            )

    @torch.inference_mode()
    def extract_embedding(self, image_path):
        x = preprocess_file(
            image_path,
            self.transform,
            cropper=self.roi_cropper,
        ).to(self.device)
        embedding = self.model.get_embedding(x).squeeze(0).cpu()
        if tuple(embedding.shape) != (512,):
            raise RuntimeError(f'Embedding phải có 512 chiều, nhận {tuple(embedding.shape)}')
        return embedding

    def close(self):
        if self.roi_cropper is not None:
            self.roi_cropper.close()
