"""Lưu template sinh trắc học dạng JSON; coi dữ liệu này là nhạy cảm."""
from pathlib import Path
import hashlib
import json
import re
from config import DEFAULT_DB


def sha256_file(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_user_id(user_id):
    if re.fullmatch(r'[a-zA-Z0-9_-]{1,40}', user_id) is None:
        raise ValueError('user ID chỉ được dùng chữ/số/_/- và tối đa 40 ký tự')
    return user_id


def open_database(db_path, model_sha256):
    path = Path(db_path)
    if not path.exists():
        return {'schema':1, 'model_sha256':model_sha256, 'embedding_dim':512,
                'users':{}}
    obj = json.loads(path.read_text(encoding='utf-8'))
    if obj.get('schema') != 1 or obj.get('embedding_dim') != 512:
        raise ValueError('Database không tương thích. Hãy tạo DB đúng định dạng.')
    if obj.get('model_sha256') != model_sha256:
        raise ValueError('Checkpoint khác với checkpoint dùng khi đăng ký. '
                         'Không so sánh embedding giữa hai model khác nhau.')
    if not isinstance(obj.get('users'), dict):
        raise ValueError('Database thiếu users.')
    return obj


def save_database(db_path, db):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(db,indent=2,ensure_ascii=False),encoding='utf-8')
    temp.replace(path)
