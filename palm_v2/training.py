"""Train a small 4-class palm encoder on a PC, leaving session_03 untouched."""
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import PalmDataset, require_training_data, samples_for_split
from .model import TinyPalmNet, PERSON_IDS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def supervised_contrastive(embeddings, labels, temperature=0.15):
    """No-op for a batch with no same-class pairs."""
    similarity = embeddings @ embeddings.T / temperature
    n = labels.numel()
    identity = torch.eye(n, device=labels.device, dtype=torch.bool)
    positives = (labels[:, None] == labels[None, :]) & ~identity
    valid = positives.any(1)
    if not valid.any():
        return embeddings.sum() * 0
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    similarity = similarity.masked_fill(identity, -1e9)
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    losses = -(log_probability * positives).sum(dim=1) / positives.sum(dim=1).clamp(min=1)
    return losses[valid].mean()


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        _, logits = model(x)
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum").item())
        correct += int((logits.argmax(1) == y).sum().item())
        total += len(y)
    return correct / max(1, total), loss_sum / max(1, total)


def train(root, checkpoint, epochs=40, batch_size=16, learning_rate=0.002, seed=42, device=None):
    require_training_data(root)
    set_seed(seed)
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_set = PalmDataset(samples_for_split(root, "train"), augment=True)
    val_set = PalmDataset(samples_for_split(root, "val"), augment=False)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    model = TinyPalmNet().to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    best = (-1.0, float("-inf"))
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, _ in loader:
            x, y = x.to(target), y.to(target)
            optimizer.zero_grad(set_to_none=True)
            embedding, logits = model(x)
            loss = F.cross_entropy(logits, y) + 0.2 * supervised_contrastive(embedding, y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(y)
        scheduler.step()
        val_acc, val_loss = accuracy(model, val_loader, target)
        print(f"epoch {epoch:03d}/{epochs} train_loss={train_loss/len(train_set):.4f}"
              f" val_acc={val_acc:.3f} val_loss={val_loss:.4f}", flush=True)
        score = (val_acc, -val_loss)
        if score > best:
            best = score
            torch.save({
                "model_state": model.state_dict(),
                "people": list(PERSON_IDS),
                "input_side": 64,
                "embedding_dim": 128,
                "epoch": epoch,
                "val_accuracy": val_acc,
            }, checkpoint)
    return {"checkpoint": str(checkpoint), "best_val_accuracy": best[0]}


def load_trained(checkpoint, device="cpu"):
    """Only load checkpoint files you created yourself / trust."""
    from pathlib import Path
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload.get("people") != list(PERSON_IDS):
        raise ValueError("Checkpoint identity order differs from person_01..04")
    if payload.get("input_side") != 64 or payload.get("embedding_dim") != 128:
        raise ValueError("Checkpoint is not a PalmLock V2 (64/128) model")
    model = TinyPalmNet().to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model
