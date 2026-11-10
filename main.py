"""PalmLock-AI CLI: crop-roi / inspect / embedding / enroll / verify / evaluate."""
import argparse
import json
from pathlib import Path
import sys

from config import (
    DEFAULT_HAND_LANDMARKER,
    DEFAULT_WEIGHTS,
    DEFAULT_DB,
    DEFAULT_VERIFY_THRESHOLD,
    ROOT,
)


def print_json(value):
    rendered = json.dumps(value, ensure_ascii=False, indent=2)
    try:
        print(rendered)
    except UnicodeEncodeError:
        print(json.dumps(value, ensure_ascii=True, indent=2))


def configure_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
        except (AttributeError, OSError):
            pass


def add_roi_options(parser):
    parser.add_argument(
        '--input-format', choices=['roi', 'full-hand'], default='roi',
        help='roi: ảnh đã crop; full-hand: tự dò và crop lòng bàn tay',
    )
    parser.add_argument(
        '--hand-model', default=str(DEFAULT_HAND_LANDMARKER),
        help='Đường dẫn hand_landmarker.task của MediaPipe',
    )
    parser.add_argument(
        '--roi-width-scale', type=float, default=1.30,
        help='Hệ số mở rộng chiều ngang ROI khi input-format=full-hand',
    )
    parser.add_argument(
        '--roi-height-scale', type=float, default=1.50,
        help='Hệ số mở rộng chiều dọc ROI khi input-format=full-hand',
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description='PalmLock-AI: suy luận trên PC, không phải ESP32 firmware'
    )
    parser.add_argument(
        '--weights', default=str(DEFAULT_WEIGHTS),
        help='Đường dẫn checkpoint PalmBridge .pt',
    )
    parser.add_argument(
        '--device', default=None, choices=['cpu', 'cuda'],
        help='Thiết bị suy luận; mặc định tự chọn',
    )
    commands = parser.add_subparsers(dest='command', required=True)

    crop = commands.add_parser(
        'crop-roi', help='Cắt thử ROI và lưu ảnh xem khung đã chọn',
    )
    crop.add_argument('--image', required=True, help='Ảnh chụp cả bàn tay')
    crop.add_argument('--output', default=None, help='Ảnh ROI đầu ra')
    crop.add_argument('--debug', default=None, help='Ảnh gốc có vẽ khung ROI')
    crop.add_argument(
        '--no-debug', action='store_true',
        help='Không lưu ảnh gốc có vẽ khung',
    )
    crop.add_argument('--hand-model', default=str(DEFAULT_HAND_LANDMARKER))
    crop.add_argument('--roi-width-scale', type=float, default=1.30)
    crop.add_argument('--roi-height-scale', type=float, default=1.50)

    commands.add_parser('inspect', help='Kiểm tra checkpoint và kiến trúc')

    embedding = commands.add_parser('embedding', help='In vector 512-D của một ảnh')
    embedding.add_argument('--image', required=True)
    add_roi_options(embedding)

    enroll = commands.add_parser('enroll', help='Đăng ký từ thư mục ảnh')
    enroll.add_argument('--user', required=True)
    enroll.add_argument('--images', required=True)
    enroll.add_argument('--db', default=str(DEFAULT_DB))
    add_roi_options(enroll)

    verify = commands.add_parser('verify', help='Tìm ứng viên và xác thực ảnh')
    verify.add_argument('--image', required=True)
    verify.add_argument('--db', default=str(DEFAULT_DB))
    verify.add_argument(
        '--threshold', type=float, default=DEFAULT_VERIFY_THRESHOLD,
        help=f'Ngưỡng cosine; mặc định demo {DEFAULT_VERIFY_THRESHOLD}',
    )
    add_roi_options(verify)

    evaluate = commands.add_parser('evaluate', help='Đánh giá ảnh test giữ riêng')
    evaluate.add_argument('--tests', required=True)
    evaluate.add_argument('--db', default=str(DEFAULT_DB))
    evaluate.add_argument('--threshold', required=True, type=float)
    add_roi_options(evaluate)
    return parser


def crop_roi_command(args):
    from PIL import ImageDraw
    from roi import HandPalmROICropper

    cropper = HandPalmROICropper(
        args.hand_model,
        width_scale=args.roi_width_scale,
        height_scale=args.roi_height_scale,
    )
    try:
        image, roi, quad = cropper.crop_file(args.image)
        source = Path(args.image)
        preview_dir = ROOT / 'roi_previews'
        output_path = Path(args.output) if args.output else preview_dir / (
            f'{source.stem}_roi.png'
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        roi.save(output_path)

        debug_path = None
        if not args.no_debug:
            if args.debug:
                debug_path = Path(args.debug)
            elif args.output:
                debug_path = output_path.with_name(
                    f'{source.stem}_roi_frame.jpg'
                )
            else:
                debug_path = preview_dir / f'{source.stem}_roi_frame.jpg'
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug = image.copy()
            points = [(round(x), round(y)) for x, y in quad]
            points.append(points[0])
            line_width = max(2, min(image.size) // 300)
            ImageDraw.Draw(debug).line(
                points, fill=(255, 40, 40), width=line_width, joint='curve'
            )
            debug.save(debug_path, quality=95)
        return {
            'input': str(source),
            'roi': str(output_path),
            'debug_frame': str(debug_path) if debug_path else None,
            'roi_size': list(roi.size),
            'roi_width_scale': args.roi_width_scale,
            'roi_height_scale': args.roi_height_scale,
        }
    finally:
        cropper.close()


def main():
    configure_console()
    args = build_parser().parse_args()
    if args.command == 'crop-roi':
        print_json(crop_roi_command(args))
        return

    from inference import PalmEmbedder

    embedder = PalmEmbedder(
        args.weights,
        args.device,
        input_format=getattr(args, 'input_format', 'roi'),
        hand_model_path=getattr(args, 'hand_model', DEFAULT_HAND_LANDMARKER),
        roi_width_scale=getattr(args, 'roi_width_scale', 1.30),
        roi_height_scale=getattr(args, 'roi_height_scale', 1.50),
    )
    try:
        if args.command == 'inspect':
            output = {
                'checkpoint': str(Path(args.weights).resolve()),
                'backbone_type': embedder.backbone_type,
                'embedding_dim': 512,
                'image_size': [128, 128],
                'device': str(embedder.device),
            }
        elif args.command == 'embedding':
            vector = embedder.extract_embedding(args.image)
            output = {
                'image': args.image,
                'embedding_dim': int(vector.numel()),
                'l2_norm': float(vector.norm()),
                'first_12_values': vector[:12].tolist(),
            }
        elif args.command == 'enroll':
            from enroll import enroll_user
            count = enroll_user(embedder, args.user, args.images, args.db)
            output = {
                'enrolled_user': args.user,
                'image_count': count,
                'database': args.db,
                'warning': 'Giữ ảnh test độc lập, không dùng lại ảnh enroll.',
            }
        elif args.command == 'verify':
            from verify import verify_image
            output = verify_image(
                embedder, args.image, args.db, args.threshold
            )
        else:
            from evaluate import evaluate
            output = evaluate(
                embedder, args.tests, args.db, args.threshold
            )
        print_json(output)
    finally:
        embedder.close()


if __name__ == '__main__':
    main()
