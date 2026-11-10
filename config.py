"""Cấu hình dùng chung, lấy từ cell 2 của palmbridge.ipynb."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "weights" / "best_palmbridge_V1_embedding.pt"
DEFAULT_DB = ROOT / "database" / "users.json"
DEFAULT_HAND_LANDMARKER = ROOT / "weights" / "hand_landmarker.task"
DEFAULT_VERIFY_THRESHOLD = 0.85
IMAGE_SIZE = 128
EMBED_DIM = 512
FEATURE_DIM = 512
NUM_PB_VECTORS = 512
W_MAP = 0.3
W_ORI = 0.7
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
