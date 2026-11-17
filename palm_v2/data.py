"""Image loading and explicit session split; prevents train/test image leakage.

Each of four people uses session_01 for training, session_02 for calibration,
and session_03 for final held-out testing. Unknown people use analogous
session_02 and session_03 folders.
"""
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch.utils.data import Dataset
from .model import INPUT_SIDE, PERSON_IDS

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SESSION_FOR_SPLIT = {"train": "session_01", "val": "session_02", "test": "session_03"}


def image_files(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def samples_for_split(root, split, include_unknown=False):
    if split not in SESSION_FOR_SPLIT:
        raise ValueError("Split must be train, val or test")
    root = Path(root)
    session = SESSION_FOR_SPLIT[split]
    result = []
    for index, person in enumerate(PERSON_IDS):
        result.extend((path, index) for path in image_files(root / person / session))
    if include_unknown and split != "train":
        result.extend((path, -1) for path in image_files(root / "unknown" / session))
    return result


def dataset_counts(root):
    return {
        split: {person: len(image_files(Path(root) / person / session))
                for person in (*PERSON_IDS, "unknown")}
        for split, session in SESSION_FOR_SPLIT.items()
    }


def load_roi(path, augment=False):
    """Central square grayscale ROI; consistent with the firmware reference.

    Frame the central palm region during capture. Do NOT feed unrelated
    full-hand photos expecting automatic landmark detection here.
    """
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("L")
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    if augment:
        image = image.rotate(random.uniform(-7.0, 7.0), resample=Image.Resampling.BILINEAR, fillcolor=0)
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
    image = image.resize((INPUT_SIDE, INPUT_SIDE), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def normalize_roi(roi):
    """Replicate V1 positive-mask mean/std (sample std, epsilon 1e-6).

    Any grayscale pixel >0 is considered part of the normalization mask.
    """
    if roi.shape != (INPUT_SIDE, INPUT_SIDE):
        raise ValueError("ROI must be 64 x 64")
    values = roi.astype(np.float32) / 255.0
    valid = roi > 0
    if int(valid.sum()) < 4:
        raise ValueError("ROI contains fewer than four nonzero pixels")
    selected = values[valid]
    mean = selected.mean(dtype=np.float64)
    std = selected.std(ddof=1, dtype=np.float64)
    values[valid] = (values[valid] - mean) / (std + 1e-6)
    values[~valid] = 0.0
    return values.astype(np.float32)


class PalmDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = list(samples)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        roi = load_roi(path, augment=self.augment)
        tensor = torch.from_numpy(normalize_roi(roi)[None, :, :].copy())
        return tensor, int(label), str(path)


def require_training_data(root, minimum_train=5, minimum_val=2):
    counts = dataset_counts(root)
    errors = []
    for person in PERSON_IDS:
        if counts["train"][person] < minimum_train:
            errors.append(f"{person}/session_01: at least {minimum_train} training photos")
        if counts["val"][person] < minimum_val:
            errors.append(f"{person}/session_02: at least {minimum_val} validation photos")
    if errors:
        raise ValueError("Missing data:\n - " + "\n - ".join(errors))
