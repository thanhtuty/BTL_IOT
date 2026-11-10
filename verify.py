"""So sánh cosine 1:N. Ngưỡng cần hiệu chỉnh với tập kiểm thử độc lập."""
import torch
import torch.nn.functional as F
from database import open_database, sha256_file
from config import DEFAULT_VERIFY_THRESHOLD


def verify_embedding(embedder, embedding, db_path,
                     threshold=DEFAULT_VERIFY_THRESHOLD):
    db = open_database(db_path, sha256_file(embedder.weights_path))
    if not db['users']:
        raise ValueError('Database chưa có người. Chạy enroll trước.')
    scores = []
    for user_id, item in db['users'].items():
        template = torch.tensor(item['template'],dtype=torch.float32)
        if tuple(template.shape) != (512,):
            raise ValueError(f'Template của {user_id} không đủ 512 số.')
        score = float(F.cosine_similarity(embedding.unsqueeze(0),
                                          template.unsqueeze(0)).item())
        scores.append({'user_id':user_id,'score':round(score,6)})
    scores.sort(key=lambda x:x['score'],reverse=True)
    top = scores[0]
    if threshold is None:
        decision = 'CHUA_HIEU_CHINH_NGUONG'
        accepted_user = None
    else:
        if not (-1.0 <= threshold <= 1.0):
            raise ValueError('Cosine threshold nằm trong [-1,1].')
        accepted_user = top['user_id'] if top['score'] >= threshold else None
        decision = 'CHAP_NHAN' if accepted_user is not None else 'TU_CHOI'
    return {'decision':decision,'accepted_user':accepted_user,
            'candidate':top['user_id'],'best_score':top['score'],'scores':scores,
            'threshold':threshold}


def verify_image(embedder, image_path, db_path,
                 threshold=DEFAULT_VERIFY_THRESHOLD):
    embedding = embedder.extract_embedding(image_path)
    return verify_embedding(embedder, embedding, db_path, threshold)
