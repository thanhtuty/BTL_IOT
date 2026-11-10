"""Test cuối cùng bằng tập ảnh giữ riêng, không dùng để enroll.

Bố cục tests/: person_01, person_02,..., outsider_01, outsider_02,...
Tên thư mục trùng ID đã đăng ký => genuine; tên khác => outsider.
"""
from enroll import list_images
from database import open_database, sha256_file
from verify import verify_image


def evaluate(embedder, tests_dir, db_path, threshold):
    db = open_database(db_path, sha256_file(embedder.weights_path))
    result = {'genuine_total':0,'genuine_rejected':0,'genuine_misidentified':0,
              'outsider_total':0,'outsider_accepted':0}
    base = __import__('pathlib').Path(tests_dir)
    if not base.is_dir():
        raise FileNotFoundError(f'Không có thư mục test: {base}')
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        paths=list_images(folder)
        for img in paths:
            out=verify_image(embedder,img,db_path,threshold)
            if folder.name in db['users']:
                result['genuine_total']+=1
                if out['accepted_user'] is None:
                    result['genuine_rejected']+=1
                elif out['accepted_user'] != folder.name:
                    result['genuine_misidentified']+=1
            else:
                result['outsider_total']+=1
                if out['accepted_user'] is not None:
                    result['outsider_accepted']+=1
    npos=result['genuine_total']; nneg=result['outsider_total']
    result['FRR'] = round(result['genuine_rejected']/npos,4) if npos else None
    result['FAR'] = round(result['outsider_accepted']/nneg,4) if nneg else None
    result['misidentification_rate'] = round(result['genuine_misidentified']/npos,4) if npos else None
    result['note']='Chỉ mang tính tham khảo nếu tập test nhỏ hoặc dùng để điều chỉnh ngưỡng.'
    return result
