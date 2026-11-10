"""Đăng ký template lòng bàn tay, KHÔNG cập nhật trọng số AI."""
from pathlib import Path
import torch
import torch.nn.functional as F
from config import SUPPORTED_EXTENSIONS
from database import open_database, save_database, sha256_file, validate_user_id


def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'Không tìm thấy thư mục ảnh: {folder}')
    return sorted(p for p in folder.rglob('*')
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def enroll_user(embedder, user_id, images_dir, db_path):
    validate_user_id(user_id)
    paths = list_images(images_dir)
    if not paths:
        raise ValueError('Thư mục không có ảnh; cần ảnh ROI lòng bàn tay đã crop.')
    embeddings = [embedder.extract_embedding(p) for p in paths]
    template = F.normalize(torch.stack(embeddings).mean(dim=0),p=2,dim=0)
    db = open_database(db_path, sha256_file(embedder.weights_path))
    db['users'][user_id] = {'template':template.tolist(),'image_count':len(paths)}
    save_database(db_path,db)
    return len(paths)
