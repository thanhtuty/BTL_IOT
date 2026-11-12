"""Fine-tune PalmBridge/CompNet on ESP32-CAM palm ROI images.

Workflow: raw/person_01/session_01/* -> prepared/{train,val,test}/person_01/*
Then update encoder layers using ArcFace, keep checkpoint compatible with
models.checkpoint.load_encoder and the existing main.py CLI.
"""
import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import DEFAULT_HAND_LANDMARKER, DEFAULT_WEIGHTS, EMBED_DIM, IMAGE_SIZE, SUPPORTED_EXTENSIONS
from models.checkpoint import load_encoder
from preprocessing import build_transform


class ArcMarginHead(nn.Module):
    """4-way ArcFace training head, not used in production verification."""
    def __init__(self, classes, scale=16.0, margin=0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, EMBED_DIM))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings, labels=None):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale
        sine = torch.sqrt(torch.clamp(1 - cosine.square(), 0, 1))
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        # Stable ArcFace decision boundary.
        threshold = math.cos(math.pi - self.margin)
        phi = torch.where(cosine > threshold, phi,
                          cosine - math.sin(math.pi - self.margin) * self.margin)
        logits = cosine.clone()
        logits.scatter_(1, labels.view(-1, 1),
                        phi.gather(1, labels.view(-1, 1)))
        return logits * self.scale


def images_in(folder):
    return sorted(p for p in folder.iterdir() if p.is_file()
                  and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def prepare_data(raw_dir, output_dir, *, input_format='full-hand',
                 hand_model=DEFAULT_HAND_LANDMARKER, allow_random_split=False,
                 seed=42, progress=None):
    """Split by independent capture sessions where available, crop to palm ROI.

    If a person has <3 sessions, random-image splitting is *opt-in* and is
    explicitly labelled demo-only, because adjacent frames leak across splits.
    """
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    if not raw_dir.is_dir():
        raise ValueError(f'Không tìm thấy ảnh: {raw_dir}')
    if (raw_dir.resolve() == output_dir.resolve() or
            raw_dir.resolve() in output_dir.resolve().parents or
            output_dir.resolve() in raw_dir.resolve().parents):
        raise ValueError('Thư mục output phải tách biệt và không bao trùm dữ liệu gốc.')
    people = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    if len(people) != 4:
        raise ValueError(f'Cần đúng 4 thư mục người; hiện có {len(people)}: {[p.name for p in people]}')
    if input_format not in ('roi', 'full-hand'):
        raise ValueError('input_format phải là roi hoặc full-hand')
    rng = random.Random(seed)
    plans = {}
    demo_only = False
    for person in people:
        sessions = []
        for folder in sorted(person.iterdir()):
            if folder.is_dir():
                pics = images_in(folder)
                if pics:
                    sessions.append((folder.name, pics))
        loose = images_in(person)
        if loose:
            sessions.append(('loose_images', loose))
        total = sum(len(pics) for _, pics in sessions)
        if total < 6:
            raise ValueError(f'{person.name}: cần tối thiểu 6 ảnh, hiện có {total}.')
        if len(sessions) >= 3:
            rng.shuffle(sessions)
            partitions = {
                'test': [sessions[0]], 'val': [sessions[1]],
                'train': sessions[2:]
            }
        elif allow_random_split:
            demo_only = True
            pics = [p for _, group in sessions for p in group]
            rng.shuffle(pics)
            val_n = max(1, round(len(pics) * .15))
            test_n = max(1, round(len(pics) * .15))
            partitions = {
                'test': [('random_test', pics[:test_n])],
                'val': [('random_val', pics[test_n:test_n + val_n])],
                'train': [('random_train', pics[test_n + val_n:])]
            }
        else:
            raise ValueError(
                f'{person.name}: cần ít nhất 3 lượt chụp độc lập (3 ZIP hoặc 3 session). '
                'Nếu chỉ muốn chạy DEMO, dùng --allow-random-split.'
            )
        plans[person.name] = partitions

    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ('train', 'val', 'test'):
        for person in people:
            (output_dir / split / person.name).mkdir(parents=True, exist_ok=True)

    cropper = None
    if input_format == 'full-hand':
        from roi import HandPalmROICropper
        cropper = HandPalmROICropper(hand_model)
    counts = Counter()
    skipped = []
    try:
        for person_name, partitions in plans.items():
            for split, groups in partitions.items():
                for session, photos in groups:
                    for i, path in enumerate(photos):
                        try:
                            with Image.open(path) as opened:
                                img = ImageOps.exif_transpose(opened).convert('RGB')
                            if cropper is not None:
                                img, _ = cropper.crop(img)
                            # Keep a high-resolution ROI; transform -> 128x128 during training.
                            img = img.convert('L')
                            target = output_dir / split / person_name / f'{session}_{i:04d}.png'
                            img.save(target)
                            counts[(split, person_name)] += 1
                            if progress:
                                progress(f'{person_name}: {split} / {path.name}')
                        except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as exc:
                            skipped.append(f'{path}: {exc}')
    finally:
        if cropper is not None:
            cropper.close()

    for person in people:
        for split in ('train', 'val', 'test'):
            if counts[(split, person.name)] == 0:
                raise ValueError(f'{person.name}: không có ROI hợp lệ trong tập {split}. '
                                 'Kiểm tra góc chụp và MediaPipe.')
    return {
        'prepared': str(output_dir), 'input_format': input_format,
        'demo_only': demo_only, 'counts': {
            split: {p.name: counts[(split, p.name)] for p in people}
            for split in ('train', 'val', 'test')
        }, 'skipped': skipped,
    }


def grayscale_loader(path):
    with Image.open(path) as img:
        return img.convert('L')


def epoch_pass(model, head, loader, device, optimizer=None):
    training = optimizer is not None
    # Freeze BN running statistics even when final conv layers receive gradients.
    model.eval()
    head.train(training)
    count, correct, loss_sum = 0, 0, 0.
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            embedding = model.get_embedding(images)
            margin_logits = head(embedding, labels)
            loss = F.cross_entropy(margin_logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g['params']], 1.)
            optimizer.step()
            with torch.no_grad():
                predictions = head(embedding.detach()).argmax(dim=1)
        else:
            with torch.inference_mode():
                embedding = model.get_embedding(images)
                logits = head(embedding)
                loss = F.cross_entropy(logits, labels)
                predictions = logits.argmax(dim=1)
        n = labels.size(0)
        count += n
        loss_sum += loss.item() * n
        correct += (predictions == labels).sum().item()
    return loss_sum / count, correct / count


def finetune(prepared_dir, weights=DEFAULT_WEIGHTS, output='weights/palmbridge_esp32_finetuned.pt',
             *, epochs=12, batch_size=4, device='auto', progress=None):
    prepared_dir = Path(prepared_dir)
    if epochs < 1 or batch_size < 1:
        raise ValueError('Epochs và batch size phải >= 1.')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA chưa sẵn sàng; chọn cpu hoặc cài PyTorch CUDA.')
    device = torch.device(device)
    transform = build_transform()
    datasets = {split: ImageFolder(prepared_dir / split, transform=transform,
                                   loader=grayscale_loader)
                for split in ('train', 'val', 'test')}
    classes = datasets['train'].classes
    if len(classes) != 4 or any(ds.classes != classes for ds in datasets.values()):
        raise ValueError('Mỗi tập train/val/test phải có cùng đúng 4 lớp danh tính.')
    loaders = {split: DataLoader(ds, batch_size=batch_size,
                                 shuffle=(split == 'train'), num_workers=0)
               for split, ds in datasets.items()}
    model, kind = load_encoder(weights, device)
    for p in model.parameters():
        p.requires_grad_(False)
    late = ((model.backbone.block4, model.backbone.embed) if kind == 'custom'
            else (model.backbone.cb3, model.backbone.fc))
    for module in (*late, model.palmbridge):
        for p in module.parameters():
            p.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    head = ArcMarginHead(len(classes)).to(device)
    optimizer = torch.optim.AdamW([
        {'params': trainable, 'lr': 1e-5},
        {'params': head.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_acc, best_loss, stale = -1., float('inf'), 0
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = epoch_pass(model, head, loaders['train'], device, optimizer)
        val_loss, val_acc = epoch_pass(model, head, loaders['val'], device)
        record = {'epoch': ep, 'train_loss': round(tr_loss, 5),
                  'train_acc': round(tr_acc, 5), 'val_loss': round(val_loss, 5),
                  'val_acc': round(val_acc, 5)}
        history.append(record)
        if progress:
            progress(record, ep, epochs)
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc, best_loss, stale = val_acc, val_loss, 0
            torch.save({
                'backbone': {k: v.detach().cpu() for k, v in model.backbone.state_dict().items()},
                'palmbridge': {k: v.detach().cpu() for k, v in model.palmbridge.state_dict().items()},
                'embedding_dim': EMBED_DIM,
                'image_size': IMAGE_SIZE,   # loader expects int 128, not [128,128]
                'class_names': classes,
                'epoch': ep, 'val_accuracy': float(best_acc),
                'source': 'PalmLock ESP32 fine-tune'
            }, output)
        else:
            stale += 1
            if stale >= 5:
                break
    # Evaluate the BEST checkpoint, not the last epoch.
    best_model, _ = load_encoder(output, device)
    # Classification head from final epoch doesn't exactly match selected checkpoint;
    # for reproducible closed-set test we report nearest class centroid instead.
    best_model.eval()
    centers = []
    with torch.inference_mode():
        for label in range(4):
            vectors = []
            for x, y in loaders['train']:
                mask = (y == label)
                if mask.any():
                    vectors.append(best_model.get_embedding(x[mask].to(device)).cpu())
            centers.append(F.normalize(torch.cat(vectors).mean(dim=0), dim=0))
        centers = torch.stack(centers).to(device)
        correct = n = 0
        for x, y in loaders['test']:
            vectors = best_model.get_embedding(x.to(device))
            preds = (vectors @ centers.T).argmax(dim=1).cpu()
            correct += (preds == y).sum().item()
            n += len(y)
    report = {
        'checkpoint': str(output), 'classes': classes, 'device': str(device),
        'backbone': kind, 'best_val_acc': best_acc,
        'test_closed_set_acc': correct / n, 'test_images': n,
        'history': history,
        'warning': ('Closed-set test on 4 people; NOT unknown-person rejection. '
                    'For door control, measure FAR/FRR on independent unknown subjects.')
    }
    out_report = output.parent / 'finetune_report.json'
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output.parent / 'finetune_history.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    return report


def main():
    p = argparse.ArgumentParser(description='PalmLock ESP32: ROI -> fine-tune PalmBridge')
    p.add_argument('--raw', default='dataset/esp32_uploads')
    p.add_argument('--prepared', default='dataset/esp32_prepared')
    p.add_argument('--input-format', choices=['roi', 'full-hand'], default='full-hand')
    p.add_argument('--hand-model', default=str(DEFAULT_HAND_LANDMARKER))
    p.add_argument('--allow-random-split', action='store_true')
    p.add_argument('--weights', default=str(DEFAULT_WEIGHTS))
    p.add_argument('--output', default='weights/palmbridge_esp32_finetuned.pt')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = p.parse_args()
    print('Chuẩn bị ROI và chia dữ liệu...', flush=True)
    prepared = prepare_data(args.raw, args.prepared, input_format=args.input_format,
                            hand_model=args.hand_model,
                            allow_random_split=args.allow_random_split)
    print(json.dumps(prepared, ensure_ascii=False, indent=2), flush=True)
    def callback(info, epoch, epochs):
        print(f"Epoch {epoch}/{epochs} | train={info['train_acc']:.1%} "
              f"val={info['val_acc']:.1%} loss={info['val_loss']:.4f}", flush=True)
    result = finetune(args.prepared, args.weights, args.output,
                      epochs=args.epochs, batch_size=args.batch_size,
                      device=args.device, progress=callback)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()