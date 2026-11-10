"""Nạp checkpoint PalmBridge theo định dạng của cell 2 notebook đã gửi.

Dùng torch.load(weights_only=True): KHÔNG nạp pickle tùy ý.
Chỉ nạp file .pt từ nguồn bạn tin tưởng.
"""
from collections.abc import Mapping
from pathlib import Path
import torch
from config import EMBED_DIM, IMAGE_SIZE
from models.backbone import CustomCompNetBackbone, OfficialCompNetBackbone
from models.palmbridge import PalmBridgeEncoder


def _strip_module(state):
    result = {}
    for key, value in state.items():
        while key.startswith("module."):
            key = key[len("module."):]
        result[key] = value
    return result


def _is_state_dict(item):
    return isinstance(item, Mapping) and any(torch.is_tensor(v) for v in item.values())


def extract_parts(ckpt):
    """Nhận dạng checkpoint embedding, best-model, hoặc full state_dict.

    Hỗ trợ các định dạng đã được lưu trong palmbridge.ipynb cell 2:
      {'backbone': state, 'palmbridge': state}
      {'model_state_dict': {'backbone.*':..., 'palmbridge.*':...}}
      {'backbone_state_dict': state, 'palmbridge_state_dict': state}
    """
    if not isinstance(ckpt, Mapping):
        raise ValueError("Checkpoint không phải dictionary.")
    if (isinstance(ckpt.get('backbone'), Mapping) and
            isinstance(ckpt.get('palmbridge'), Mapping)):
        return _strip_module(ckpt['backbone']), _strip_module(ckpt['palmbridge'])
    if (isinstance(ckpt.get('backbone_state_dict'), Mapping) and
            isinstance(ckpt.get('palmbridge_state_dict'), Mapping)):
        return _strip_module(ckpt['backbone_state_dict']), _strip_module(ckpt['palmbridge_state_dict'])
    state = None
    for key in ('model_state_dict','state_dict','model','net'):
        if _is_state_dict(ckpt.get(key)):
            state = ckpt[key]
            break
    if state is None and _is_state_dict(ckpt):
        state = ckpt
    if state is None:
        raise ValueError('Không tìm thấy model_state_dict/backbone/palmbridge trong checkpoint.')
    state = _strip_module(state)
    backbone = {}
    palmbridge = {}
    backbone_prefixes = ('cb1.','cb2.','cb3.','fc.','block1.','block2.',
                         'block3.','block4.','embed.','gabor.','gbn.','pool1.',
                         'pool2.','pool3.','gap.')
    for key, val in state.items():
        if key.startswith('backbone.'):
            backbone[key[len('backbone.'):]] = val
        elif key.startswith('palmbridge.'):
            palmbridge[key[len('palmbridge.'):]] = val
        elif key.startswith(backbone_prefixes):
            backbone[key] = val
        elif key == 'P':
            palmbridge[key] = val
    if not backbone or not palmbridge:
        raise ValueError('Thiếu backbone hoặc PalmBridge vector P trong checkpoint.')
    return backbone, palmbridge


def detect_backbone_type(state):
    keys = state.keys()
    if any(k.startswith('cb1.') for k in keys):
        return 'official'
    if any(k.startswith(('block1.', 'gabor.')) for k in keys):
        return 'custom'
    raise ValueError('Không nhận diện được loại backbone (official/custom).')


def load_encoder(weights_path, device='cpu'):
    """Nạp model và trọng số. Không nạp một encoder ngẫu nhiên để nhận diện."""
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(
            f'Không có checkpoint: {path}\n'
            'Tải best_palmbridge_V1_embedding.pt từ Kaggle (output của cell 2), '
            'đặt trong weights/ rồi chạy lại.'
        )
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f'Không đọc được checkpoint an toàn: {path}. '
            'Kiểm tra định dạng, phiên bản PyTorch và nguồn file. '
            'Không tắt weights_only cho file không tin cậy.'
        ) from exc
    if isinstance(checkpoint, Mapping):
        emb = checkpoint.get('embedding_dim')
        img = checkpoint.get('image_size')
        if emb is not None and int(emb) != EMBED_DIM:
            raise ValueError(f'Checkpoint embedding_dim={emb}, cần {EMBED_DIM}.')
        if img is not None and int(img) != IMAGE_SIZE:
            raise ValueError(f'Checkpoint image_size={img}, cần {IMAGE_SIZE}.')
    backbone_state, pb_state = extract_parts(checkpoint)
    kind = detect_backbone_type(backbone_state)
    backbone = OfficialCompNetBackbone() if kind == 'official' else CustomCompNetBackbone()
    model = PalmBridgeEncoder(backbone)
    try:
        model.backbone.load_state_dict(backbone_state, strict=True)
        model.palmbridge.load_state_dict(pb_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f'Trọng số không khớp backbone {kind}; hãy dùng checkpoint đúng '
            'với notebook PalmBridge được gửi. Chi tiết: ' + str(exc)
        ) from exc
    model = model.to(device)
    model.eval()
    return model, kind
