"""B5A: inference-only streaming Gabor + ReLU + Competitive Pool.

Original: ReLU(conv2d(x, 32 Gabor kernels)) -> channel-wise abs max.
Since ReLU outputs >= 0, the result is a pixel-wise maximum over the
32 ReLU-filtered maps. Process ONE kernel at a time; do not materialize
the [N, 32, H, W] output tensor.

This is a proof of principle in PyTorch, NOT a C/ESP-DL operator. Actual
MCU RAM savings must be verified after porting and memory planning.
"""
import torch
import torch.nn.functional as F

from models.compact import CustomCompactBackbone
from models.palmbridge import PalmBridgeEncoder


def streaming_gabor_compete(gabor, x):
    """Exactly the same intended math as relu(gabor(x)) -> CompetitivePool.

    Inference only. Uses TWO [N,1,H,W] activation buffers (current response
    and running max), plus conv workspaces/kernel params, rather than the
    full [N,32,H,W] response. No absolute value is needed after ReLU.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("Streaming Gabor chi duoc dung cho inference voi torch.inference_mode().")
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError("Gabor expected tensor [N,1,H,W].")
    filters = gabor._build_filters()
    pooled = None
    for i in range(filters.shape[0]):
        response = F.relu(F.conv2d(x, filters[i:i + 1], padding=gabor.ks // 2))
        if pooled is None:
            pooled = response
        else:
            # In-place update: do not keep 32 activation maps.
            torch.maximum(pooled, response, out=pooled)
    if pooled is None:
        raise ValueError("No Gabor filters.")
    return pooled


class StreamingCompactBackbone(CustomCompactBackbone):
    """Compact CompNet using streaming Gabor at inference only.

    State dict keys and every trainable parameter match CustomCompactBackbone.
    """

    def getFeatureCode(self, x):
        g = self.gbn(streaming_gabor_compete(self.gabor, x))
        f = self.pool1(self.block1(g))
        f = self.pool2(self.block2(f))
        f = self.pool3(self.block3(f))
        z = self.embed(self.gap(self.block4(f)))
        return F.normalize(z, p=2, dim=1)


@torch.no_grad()
def as_streaming_encoder(compact_encoder):
    if not isinstance(compact_encoder.backbone, CustomCompactBackbone):
        raise TypeError("Chi nhan CustomCompactBackbone checkpoint da train.")
    encoder = PalmBridgeEncoder(StreamingCompactBackbone())
    encoder.load_state_dict(compact_encoder.state_dict(), strict=True)
    encoder.input_size = getattr(compact_encoder, "input_size", 128)
    device = next(compact_encoder.parameters()).device
    return encoder.to(device).eval()
