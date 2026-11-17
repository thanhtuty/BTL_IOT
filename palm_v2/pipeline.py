"""Enrollment, threshold calibration, verification and held-out evaluation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import PalmDataset, image_files, load_roi, normalize_roi, samples_for_split
from .model import EMBED_DIM, PERSON_IDS
from .training import load_trained


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def embeddings_for(model, samples, device="cpu", batch_size=16):
    if not samples:
        return np.empty((0, EMBED_DIM), np.float32), np.empty((0,), np.int64)
    loader = DataLoader(PalmDataset(samples, augment=False),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    all_z, all_y = [], []
    for x, y, _ in loader:
        z, _ = model(x.to(device))
        all_z.append(z.cpu().numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_z), np.concatenate(all_y)


def enrolled_templates(checkpoint, data_root, path, min_images=5):
    model = load_trained(checkpoint)
    result = []
    for index, person in enumerate(PERSON_IDS):
        samples = [s for s in samples_for_split(data_root, "train") if s[1] == index]
        if len(samples) < min_images:
            raise ValueError(f"{person}: need >= {min_images} images in session_01")
        z, _ = embeddings_for(model, samples)
        center = z.mean(axis=0)
        center /= max(float(np.linalg.norm(center)), 1e-12)
        result.append(center)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, centroids=np.asarray(result, dtype=np.float32),
                        people=np.asarray(PERSON_IDS), model_sha256=np.array(sha256(checkpoint)))
    return {"templates": str(path), "training_people": list(PERSON_IDS)}


def load_templates(checkpoint, path):
    with np.load(path, allow_pickle=False) as data:
        centers = np.array(data["centroids"], dtype=np.float32)
        people = data["people"].tolist()
        checkpoint_sha = str(data["model_sha256"].item())
    if centers.shape != (4, EMBED_DIM) or people != list(PERSON_IDS):
        raise ValueError("Invalid V2 templates shape or people order")
    if checkpoint_sha != sha256(checkpoint):
        raise ValueError("Checkpoint changed. Re-enroll and recalibrate before use.")
    return centers


def score_vectors(z, centers):
    scores = z @ centers.T
    best_index = np.argmax(scores, axis=1)
    best = scores[np.arange(len(scores)), best_index]
    return best_index, best, scores


def margins(scores):
    ordered = np.sort(scores, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def calibration(checkpoint, templates, data_root, output):
    model = load_trained(checkpoint)
    centers = load_templates(checkpoint, templates)
    known = samples_for_split(data_root, "val")
    outsiders = [s for s in samples_for_split(data_root, "val", True) if s[1] < 0]
    if len(known) < 8 or len(outsiders) < 10:
        raise ValueError("Need session_02: at least 2 photos per member and 10 outsider photos")
    kz, ky = embeddings_for(model, known)
    uz, _ = embeddings_for(model, outsiders)
    kid, ks, k_scores = score_vectors(kz, centers)
    _, us, u_scores = score_vectors(uz, centers)
    km, um = margins(k_scores), margins(u_scores)
    # Validation-only grid search for BOTH identity score and second-best gap.
    # FAR is weighted highly because this pipeline may be used for a lock.
    choices = []
    for threshold in np.linspace(-1.0, 1.0, 401):
        for margin_threshold in np.linspace(0.0, 0.35, 36):
            known_accept = (ks >= threshold) & (km >= margin_threshold)
            unknown_accept = (us >= threshold) & (um >= margin_threshold)
            reject_or_wrong = float(np.mean((~known_accept) | (kid != ky)))
            outsider_accept = float(np.mean(unknown_accept))
            wrong_identity_accept = float(np.mean(known_accept & (kid != ky)))
            objective = reject_or_wrong + 5.0 * outsider_accept + 10.0 * wrong_identity_accept
            choices.append((objective, -float(threshold), -float(margin_threshold),
                            reject_or_wrong, outsider_accept))
    score, neg_t, neg_m, false_reject_or_wrong, far = min(choices)
    threshold, margin_threshold = -neg_t, -neg_m
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold": threshold,
        "margin_threshold": margin_threshold,
        "checkpoint_sha256": sha256(checkpoint),
        "templates_sha256": sha256(templates),
        "validation_known": len(known),
        "validation_unknown": len(outsiders),
        "validation_reject_or_wrong_rate": false_reject_or_wrong,
        "validation_unknown_far": far,
        "note": "Validation estimate only. Test on separate session_03 before enabling a real lock."
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_threshold(checkpoint, templates, threshold_path):
    payload = json.loads(Path(threshold_path).read_text(encoding="utf-8"))
    if payload["checkpoint_sha256"] != sha256(checkpoint):
        raise ValueError("Threshold uses a different checkpoint")
    if payload["templates_sha256"] != sha256(templates):
        raise ValueError("Threshold uses different templates")
    value = float(payload["threshold"])
    margin = float(payload["margin_threshold"])
    if not -1.0 <= value <= 1.0 or not 0.0 <= margin <= 2.0:
        raise ValueError("Invalid score/margin threshold")
    return value, margin


@torch.no_grad()
def verify_image(image_path, checkpoint, templates, threshold_path):
    threshold, margin_threshold = load_threshold(checkpoint, templates, threshold_path)
    centers = load_templates(checkpoint, templates)
    model = load_trained(checkpoint)
    array = normalize_roi(load_roi(image_path))
    x = torch.from_numpy(array[None, None])
    z, _ = model(x)
    index, best, scores = score_vectors(z.numpy(), centers)
    gap = float(margins(scores)[0])
    accepted = bool(best[0] >= threshold and gap >= margin_threshold)
    winner = int(index[0])
    return {
        "person": PERSON_IDS[winner] if accepted else "unknown",
        "candidate": PERSON_IDS[winner],
        "accepted": accepted,
        "score": float(best[0]),
        "margin": gap,
        "threshold": float(threshold),
        "margin_threshold": float(margin_threshold),
        "scores": {name: float(score) for name, score in zip(PERSON_IDS, scores[0])}
    }


def evaluate_test(checkpoint, templates, threshold_path, data_root):
    """Uses session_03 only. Do not tune threshold based on these results."""
    threshold, margin_threshold = load_threshold(checkpoint, templates, threshold_path)
    model = load_trained(checkpoint)
    centers = load_templates(checkpoint, templates)
    known = samples_for_split(data_root, "test")
    outsider = [s for s in samples_for_split(data_root, "test", True) if s[1] < 0]
    if not known or not outsider:
        raise ValueError("Test needs session_03 images of members AND outsiders")
    z, y = embeddings_for(model, known)
    index, best, scores = score_vectors(z, centers)
    accepted = (best >= threshold) & (margins(scores) >= margin_threshold)
    uz, _ = embeddings_for(model, outsider)
    _, unknown_best, unknown_scores = score_vectors(uz, centers)
    unknown_accepted = (unknown_best >= threshold) & (margins(unknown_scores) >= margin_threshold)
    report = {
        "known_test_images": int(len(y)),
        "unknown_test_images": int(len(unknown_best)),
        "threshold": threshold,
        "margin_threshold": margin_threshold,
        "correct_id_and_accepted_rate": float(np.mean(accepted & (index == y))),
        "false_reject_rate_FRR": float(np.mean(~accepted)),
        "wrong_identity_accepted_rate": float(np.mean(accepted & (index != y))),
        "outsider_false_accept_rate_FAR": float(np.mean(unknown_accepted)),
        "note": "Small test sets give uncertain FAR/FRR estimates."
    }
    return report
