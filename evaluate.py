"""Evaluate held-out palm photos; never train or change model weights.

Directory convention:
  tests/person_01/*.jpg ... tests/person_04/*.jpg -> enrolled members
  tests/outsider_01/*.jpg                     -> unknown people

If MediaPipe cannot detect a palm in one photo, record the failure and
continue with the other photos. For enrolled people, a preprocessing
failure counts as a false rejection (fail closed), not a successful test.
"""

from pathlib import Path

from enroll import list_images
from database import open_database, sha256_file
from verify import verify_image


def evaluate(embedder, tests_dir, db_path, threshold):
    db = open_database(db_path, sha256_file(embedder.weights_path))
    base = Path(tests_dir)
    if not base.is_dir():
        raise FileNotFoundError(f'Không có thư mục test: {base}')

    result = {
        'genuine_total': 0,
        'genuine_correct': 0,
        'genuine_rejected': 0,
        'genuine_misidentified': 0,
        'outsider_total': 0,
        'outsider_accepted': 0,
        'processed_images': 0,
        'failed_images': 0,
        'per_person': {},
        'details': [],
    }

    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        images = list_images(folder)
        known = folder.name in db['users']
        person = {
            'total': 0,
            'correct': 0,
            'rejected': 0,
            'misidentified': 0,
            'failed': 0,
        }
        result['per_person'][folder.name] = person

        for img in images:
            person['total'] += 1
            if known:
                result['genuine_total'] += 1
            else:
                result['outsider_total'] += 1

            detail = {
                'image': str(img.relative_to(base)),
                'expected_user': folder.name if known else 'OUTSIDER',
            }
            try:
                output = verify_image(embedder, img, db_path, threshold)
            except (OSError, ValueError, RuntimeError) as exc:
                # MediaPipe or image decode errors affect only this image.
                # Do not conceal systemic database errors; DB is opened above.
                result['failed_images'] += 1
                person['failed'] += 1
                if known:
                    result['genuine_rejected'] += 1
                    person['rejected'] += 1
                detail.update({
                    'status': 'KHONG_XU_LY_DUOC_ANH',
                    'candidate': None,
                    'accepted_user': None,
                    'best_score': None,
                    'error': str(exc),
                })
                result['details'].append(detail)
                continue

            result['processed_images'] += 1
            accepted = output['accepted_user']
            detail.update({
                'candidate': output['candidate'],
                'accepted_user': accepted,
                'best_score': output['best_score'],
                'scores': output['scores'],
            })
            if known:
                if accepted is None:
                    result['genuine_rejected'] += 1
                    person['rejected'] += 1
                    detail['status'] = 'TU_CHOI_NHAM'
                elif accepted != folder.name:
                    result['genuine_misidentified'] += 1
                    person['misidentified'] += 1
                    detail['status'] = 'NHAN_NHAM_NGUOI'
                else:
                    result['genuine_correct'] += 1
                    person['correct'] += 1
                    detail['status'] = 'NHAN_DIEN_DUNG'
            elif accepted is not None:
                result['outsider_accepted'] += 1
                detail['status'] = 'CHAP_NHAN_NHAM_NGUOI_LA'
            else:
                detail['status'] = 'TU_CHOI_DUNG_NGUOI_LA'
            result['details'].append(detail)

    npos = result['genuine_total']
    nneg = result['outsider_total']
    result['FRR'] = round(result['genuine_rejected'] / npos, 4) if npos else None
    result['FAR'] = round(result['outsider_accepted'] / nneg, 4) if nneg else None
    result['misidentification_rate'] = (
        round(result['genuine_misidentified'] / npos, 4) if npos else None
    )
    result['genuine_success_rate'] = (
        round(result['genuine_correct'] / npos, 4) if npos else None
    )
    result['threshold'] = threshold
    result['note'] = (
        'Test only: checkpoint KHONG thay doi. '
        'Anh loi xu ly duoc tinh la tu choi (neu la thanh vien). '
        'Nguong demo chua duoc hieu chinh; FAR/FRR can kiem tra '
        'trong cac luot chup doc lap, voi anh nguoi la.'
    )
    return result
