"""Experimental Custom CompNet: keep Gabor, CNN, PalmBridge and 512-D embedding.
Only change AdaptiveAvgPool2d(4x4) -> (2x2), so FC shrinks by 75%.
This is a NEW architecture: old 4x4 FC weights are projected, not reused verbatim.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FEATURE_DIM
from models.backbone import CustomCompNetBackbone


class CustomCompactBackbone(CustomCompNetBackbone):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((2, 2))
        self.embed[1] = nn.Linear(256 * 2 * 2, FEATURE_DIM, bias=False)


@torch.no_grad()
def initialize_compact_from_full(full_encoder):
    """Initialize compact using matching original weights and a spatial FC projection.

    The 4x4 FC weights are summed within non-overlapping 2x2 spatial groups.
    If each group's four feature values coincide, compact output matches full
    pre-normalization FC. This does NOT guarantee equivalent predictions.
    """
    from models.palmbridge import PalmBridgeEncoder

    original = full_encoder.backbone
    if not isinstance(original, CustomCompNetBackbone) or isinstance(original, CustomCompactBackbone):
        raise ValueError("Cần checkpoint có CustomCompNetBackbone gốc (FC 4096→512).")
    old = original.state_dict()
    old_w = old["embed.1.weight"]
    if tuple(old_w.shape) != (FEATURE_DIM, 256 * 4 * 4):
        raise ValueError(f"FC gốc không đúng kích thước: {tuple(old_w.shape)}")
    new_w = (
        F.avg_pool2d(
            old_w.view(FEATURE_DIM * 256, 1, 4, 4),
            kernel_size=2, stride=2,
        ).view(FEATURE_DIM, 256 * 2 * 2) * 4.0
    )
    state = dict(old)
    state["embed.1.weight"] = new_w
    compact = PalmBridgeEncoder(CustomCompactBackbone())
    compact.backbone.load_state_dict(state, strict=True)
    compact.palmbridge.load_state_dict(full_encoder.palmbridge.state_dict(), strict=True)
    return compact
