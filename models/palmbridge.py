"""PalmBridge theo notebook (nguyên lớp) và wrapper inference-only.
Không chứa ArcMargin vì nhận diện bằng vector không cần classifier.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_PB_VECTORS, FEATURE_DIM, W_MAP, W_ORI

class PalmBridge(nn.Module):
    def __init__(self):
        super().__init__()

        self.K = NUM_PB_VECTORS

        self.P = nn.Parameter(
            F.normalize(
                torch.randn(
                    NUM_PB_VECTORS,
                    FEATURE_DIM,
                ),
                p=2,
                dim=1,
            )
        )

    def _nearest_vector(self, z):
        dists = (
            z.pow(2).sum(
                1,
                keepdim=True,
            )
            + self.P.pow(2).sum(1).unsqueeze(0)
            - 2.0
            * (
                z @ self.P.t()
            )
        )

        indices = dists.argmin(dim=1)
        z_tilde = self.P[indices]

        return z_tilde, indices

    def _blend(self, z, z_tilde):
        return (
            W_ORI * z
            + W_MAP * z_tilde
        )

    def forward(self, z):
        z_tilde, indices = self._nearest_vector(z)
        z_hat = self._blend(z, z_tilde)

        return z_hat, z_tilde, indices


class PalmBridgeEncoder(nn.Module):
    """Đưa ảnh ROI qua backbone + PalmBridge để tạo embedding 512 chiều."""
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.palmbridge = PalmBridge()

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone.getFeatureCode(x)
        z = F.normalize(z, p=2, dim=1)
        z_hat, _, _ = self.palmbridge(z)
        return F.normalize(z_hat, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_embedding(x)
