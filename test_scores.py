"""PalmLock: test all local images and show cosine similarity as percentages.

NO TRAINING. NO threshold. NO rejection/acceptance decision.
Each displayed % = cosine_similarity * 100, NOT probability/accuracy.
Run: python test_scores.py
"""
import argparse
import csv
import sys
from pathlib import Path

from config import ROOT
from database import open_database, sha256_file
from enroll import list_images
from inference import PalmEmbedder
from verify import verify_image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test PalmBridge: print % cosine similarity of each image to every enrolled user; no threshold.'
    )
    parser.add_argument('--tests', type=Path, default=ROOT / 'test_photos')
    parser.add_argument('--weights', type=Path,
                        default=ROOT / 'weights' / 'palmbridge_esp32_finetuned.pt')
    parser.add_argument('--db', type=Path,
                        default=ROOT / 'database' / 'test_4people_12epoch.json')
    parser.add_argument('--input-format', choices=['full-hand', 'roi'],
                        default='full-hand')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'test_results' / 'score_per_image.csv')
    return parser.parse_args()


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    except (AttributeError, OSError):
        pass
    args = parse_args()
    if not args.tests.is_dir():
        raise FileNotFoundError(f'Khong tim thay thu muc test: {args.tests}')
    if not args.weights.is_file():
        raise FileNotFoundError(f'Khong tim thay checkpoint: {args.weights}')
    if not args.db.is_file():
        raise FileNotFoundError(f'Chua enroll: khong co database: {args.db}')

    db = open_database(args.db, sha256_file(args.weights))
    users = sorted(db['users'])
    if not users:
        raise ValueError('Database chua co nguoi da enroll.')
    photos = [(folder.name, image)
              for folder in sorted(p for p in args.tests.iterdir() if p.is_dir())
              for image in list_images(folder)]
    if not photos:
        raise ValueError(f'Khong co anh test trong {args.tests}')

    fields = ['image', 'expected', 'predicted', 'top1_correct',
              'best_similarity_pct'] + [f'{u}_similarity_pct' for u in users] + ['error']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = correct = failed = 0
    known_processed = 0
    print('=== PALMLOCK - DIEM TUONG DONG TUNG ANH (KHONG NGUONG) ===', flush=True)
    print('Luu y: cosine*100 la diem tuong dong, KHONG PHAI xac suat AI dung.\n', flush=True)

    embedder = PalmEmbedder(args.weights, input_format=args.input_format)
    try:
        with args.output.open('w', newline='', encoding='utf-8-sig') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for expected, image in photos:
                try:
                    relative = str(image.relative_to(args.tests))
                except ValueError:
                    relative = str(image)
                row = {'image': relative, 'expected': expected,
                       'predicted': '', 'top1_correct': '', 'best_similarity_pct': '',
                       'error': ''}
                try:
                    # None disables threshold. The highest-cosine candidate is
                    # always reported, even if its score is very low.
                    result = verify_image(embedder, image, args.db, threshold=None)
                    predicted = result['candidate']
                    values = {s['user_id']: s['score'] * 100.0
                              for s in result['scores']}
                    known = expected in db['users']
                    matched = known and (predicted == expected)
                    processed += 1
                    if known:
                        known_processed += 1
                        if matched:
                            correct += 1
                    row.update({
                        'predicted': predicted,
                        'top1_correct': 'DUNG' if matched else ('SAI' if known else 'KHONG_CO_NHAN'),
                        'best_similarity_pct': f"{result['best_score'] * 100:.2f}",
                    })
                    for u in users:
                        row[f'{u}_similarity_pct'] = f"{values[u]:.2f}"
                    verdict = row['top1_correct']
                    parts = ' | '.join(f'{u}={values[u]:.2f}%' for u in users)
                    print(f'{relative}: du doan {predicted} | cao nhat {row["best_similarity_pct"]}% '
                          f'| {verdict}\n    {parts}', flush=True)
                except (ValueError, OSError, RuntimeError) as exc:
                    failed += 1
                    row['error'] = str(exc)
                    row['top1_correct'] = 'LOI_ANH'
                    print(f'{relative}: LOI ANH - {exc}', flush=True)
                writer.writerow(row)
    finally:
        embedder.close()

    print('\n=== TONG KET ===', flush=True)
    print(f'So anh: {len(photos)} | Xu ly duoc: {processed} | Loi xu ly: {failed}', flush=True)
    if known_processed:
        print(f'Du doan dung danh tinh gan nhat: {correct}/{known_processed} '
              f'({correct / known_processed:.1%} tren anh da xu ly, co nhan)', flush=True)
    print(f'Chi tiet CSV: {args.output}', flush=True)
    print('Khong thay doi checkpoint / khong train lai.', flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'LOI: {exc}', file=sys.stderr)
        sys.exit(1)
