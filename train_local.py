"""Fine-tune PalmBridge from local ESP32-CAM images (no browser or uploads).

Run in repository root:
    python train_local.py                     # full-hand photos, auto ROI
    python train_local.py --input-format roi  # already-cropped palm ROIs

Put images or ESP32 ZIP batches under dataset/esp32_photos/person_01..person_04.
Optional separate session folders improve train/validation/test independence.
"""
import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from config import DEFAULT_HAND_LANDMARKER, DEFAULT_WEIGHTS, ROOT, SUPPORTED_EXTENSIONS

PEOPLE = tuple(f'person_{i:02d}' for i in range(1, 5))
DEFAULT_PHOTOS = ROOT / 'dataset' / 'esp32_photos'
STAGING = ROOT / 'dataset' / 'esp32_staging'
PREPARED = ROOT / 'dataset' / 'esp32_prepared'
OUTPUT = ROOT / 'weights' / 'palmbridge_esp32_finetuned.pt'
LANDMARKER_URL = ('https://storage.googleapis.com/mediapipe-models/'
                  'hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task')
MAX_MEMBER = 16 * 1024 * 1024
MAX_ZIP_IMAGES = 2000


def image_files(folder):
    return sorted(f for f in folder.iterdir() if f.is_file()
                  and f.suffix.lower() in SUPPORTED_EXTENSIONS)


def ensure_folders(root):
    root.mkdir(parents=True, exist_ok=True)
    for name in PEOPLE:
        (root / name).mkdir(exist_ok=True)
    return [root / name for name in PEOPLE]


def safe_copy_images(paths, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths, 1):
        shutil.copyfile(path, destination / f'{index:05d}{path.suffix.lower()}')
    return len(paths)


def extract_zip(zip_path, destination):
    """Extract only image members, never archive paths (prevents ZIP traversal)."""
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as archive:
        photos = [info for info in archive.infolist()
                  if not info.is_dir()
                  and Path(info.filename).suffix.lower() in SUPPORTED_EXTENSIONS]
        if len(photos) > MAX_ZIP_IMAGES:
            raise ValueError(f'ZIP co qua nhieu anh: {zip_path}')
        for info in photos:
            if info.file_size > MAX_MEMBER:
                raise ValueError(f'Anh trong ZIP qua 16 MB: {zip_path} / {info.filename}')
            suffix = Path(info.filename).suffix.lower()
            target = destination / f'{count + 1:05d}{suffix}'
            with archive.open(info) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def collect_sessions(photos_root, staging):
    """Stage photos as 4 person directories, with archives treated as sessions."""
    expected = ensure_folders(photos_root)
    if staging.resolve() == photos_root.resolve() or staging.resolve() in photos_root.resolve().parents or photos_root.resolve() in staging.resolve().parents:
        raise ValueError('Thu muc staging phai rieng voi anh nguon.')
    # Remove only the dedicated, Git-ignored staging directory. Never touch raw photos.
    if staging.exists():
        shutil.rmtree(staging)
    sessions_per_person = {}
    total_per_person = {}
    for person in expected:
        out = staging / person.name
        out.mkdir(parents=True, exist_ok=True)
        session_number = 0
        count = 0
        direct = image_files(person)
        if direct:
            session_number += 1
            count += safe_copy_images(direct, out / f'session_{session_number:03d}_loose')
        for child in sorted(person.iterdir()):
            if child.is_dir():
                group = image_files(child)
                if group:
                    session_number += 1
                    count += safe_copy_images(group, out / f'session_{session_number:03d}_folder')
            elif child.is_file() and child.suffix.lower() == '.zip':
                session_number += 1
                target = out / f'session_{session_number:03d}_zip'
                n = extract_zip(child, target)
                count += n
                if not n:
                    shutil.rmtree(target)
                    session_number -= 1
        sessions_per_person[person.name] = session_number
        total_per_person[person.name] = count
    return sessions_per_person, total_per_person


def get_hand_landmarker(path):
    if path.is_file():
        return
    print('Dang tai MediaPipe hand_landmarker.task (chi tai model, khong gui anh)...', flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix('.download')
    try:
        with urllib.request.urlopen(LANDMARKER_URL, timeout=60) as response, temp_path.open('wb') as dest:
            shutil.copyfileobj(response, dest)
        if temp_path.stat().st_size < 100_000:
            raise ValueError('Model MediaPipe tai ve khong hop le')
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Train PalmLock from local ESP32-CAM photos, no UI')
    parser.add_argument('--photos', type=Path, default=DEFAULT_PHOTOS)
    parser.add_argument('--input-format', choices=['full-hand', 'roi'], default='full-hand')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    photos_root = args.photos.resolve()

    folders = ensure_folders(photos_root)
    print('Thu muc nhan anh:', photos_root, flush=True)
    print('Moi nguoi copy JPG/PNG hoac ZIP vao:', flush=True)
    for folder in folders:
        print(' -', folder, flush=True)

    sessions, counts = collect_sessions(photos_root, STAGING)
    for person in PEOPLE:
        print(f' {person}: {counts[person]} anh / {sessions[person]} dot chup', flush=True)
    if any(counts[p] < 6 for p in PEOPLE):
        print('\nChua du anh. Can it nhat 6 anh cho moi nguoi. '
              'Hay copy anh vao 4 thu muc o tren roi chay lai.', flush=True)
        return 1

    demo = any(sessions[p] < 3 for p in PEOPLE)
    if demo:
        print('\nCANH BAO: Co nguoi duoi 3 dot chup doc lap; '
              'se chia ngau nhien de TRAIN THU. '
              'Ket qua validation/test co the qua lac quan do anh chup lien tiep giong nhau.\n', flush=True)
    if not DEFAULT_WEIGHTS.is_file():
        print('THIEU CHECKPOINT:', DEFAULT_WEIGHTS, flush=True)
        return 2
    if args.input_format == 'full-hand':
        get_hand_landmarker(DEFAULT_HAND_LANDMARKER)

    from finetune import prepare_data, finetune
    print('Dang cat ROI va chia tap du lieu...', flush=True)
    result = prepare_data(STAGING, PREPARED, input_format=args.input_format,
                          allow_random_split=demo)
    print('So anh hop le:', json.dumps(result['counts'], ensure_ascii=False), flush=True)
    if result['skipped']:
        print(f"Da bo qua {len(result['skipped'])} anh loi, vi du:", flush=True)
        for message in result['skipped'][:5]:
            print('  ', message, flush=True)
    if args.prepare_only:
        print('Da chuan bi anh. Hay xem dataset/esp32_prepared roi chay lai khong co --prepare-only')
        return 0

    print('Bat dau fine-tune PalmBridge...', flush=True)
    def on_epoch(info, epoch, epochs):
        print(f"Epoch {epoch}/{epochs}: train_acc={info['train_acc']:.1%} "
              f"val_acc={info['val_acc']:.1%} val_loss={info['val_loss']:.4f}", flush=True)
    report = finetune(PREPARED, weights=DEFAULT_WEIGHTS, output=OUTPUT,
                      epochs=args.epochs, batch_size=args.batch_size,
                      device=args.device, progress=on_epoch)
    print('\nTHANH CONG! Checkpoint moi:', report['checkpoint'], flush=True)
    print('Ket qua:', ROOT / 'weights' / 'finetune_report.json', flush=True)
    if demo:
        print('Luu y: train thu voi anh chia ngau nhien, chua du tin cay de mo khoa that.', flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('LOI:', exc, file=sys.stderr)
        sys.exit(1)
