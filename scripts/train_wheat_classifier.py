import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 42
PILOT_DIR = "eval_photos"
MODELS_DIR = "models"
REPORT_DIR = "data/processed"
PLOTS_DIR = "data/processed/plots"
IMAGE_TYPES = (".jpg", ".jpeg", ".png")
PREFIXES = {
    "yellowrust": "yellow-rust",
    "brownrust": "brown-rust",
    "mildew": "powdery-mildew",
    "healthy": "healthy",
    "septoria": "septoria",
}
DISEASES = ["yellow-rust", "brown-rust", "powdery-mildew"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_dataset(data_dir):
    items = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            if not name.lower().endswith(IMAGE_TYPES):
                continue
            m = re.match(r"[a-z]+", name.lower())
            label = PREFIXES.get(m.group(0)) if m else None
            if label:
                items.append((os.path.join(root, name), label))
    return sorted(items)


def collect_pilot():
    items = []
    if not os.path.isdir(PILOT_DIR):
        return items
    for label in sorted(os.listdir(PILOT_DIR)):
        folder = os.path.join(PILOT_DIR, label)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(IMAGE_TYPES):
                items.append((os.path.join(folder, name), label))
    return items


class LeafDataset(Dataset):
    def __init__(self, items, classes, transform):
        self.transform = transform
        self.samples = []
        for path, label in items:
            img = Image.open(path).convert("RGB")
            img.thumbnail((384, 384))
            self.samples.append((img, classes.index(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img, y = self.samples[i]
        return self.transform(img), y


def build_model(arch, n_classes):
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        head = model.fc
    elif arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_classes)
        head = model.classifier[3]
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model, head


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, ys = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            ys.extend(y.cpu().tolist())
    return total_loss / len(ys), accuracy_score(ys, preds), f1_score(ys, preds, average="macro", zero_division=0)


@torch.no_grad()
def get_logits(model, loader, device):
    model.eval()
    outputs, ys = [], []
    for x, y in loader:
        outputs.append(model(x.to(device)).cpu())
        ys.extend(y.tolist())
    return torch.cat(outputs), np.array(ys)


def fit_temperature(logits, ys):
    log_t = torch.zeros(1, requires_grad=True)
    target = torch.tensor(ys)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), target)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def expected_calibration_error(probs, ys, bins=10):
    conf, pred = probs.max(1)
    conf, pred = conf.numpy(), pred.numpy()
    correct = pred == ys
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def coverage_table(probs, ys, thresholds=(0.0, 0.5, 0.7, 0.8, 0.9)):
    conf, pred = probs.max(1)
    conf, pred = conf.numpy(), pred.numpy()
    rows = []
    for t in thresholds:
        keep = conf >= t
        coverage = float(keep.mean())
        accuracy = float((pred[keep] == ys[keep]).mean()) if keep.any() else float("nan")
        rows.append({"threshold": t, "coverage": round(coverage, 3), "accuracy": round(accuracy, 3)})
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fmt(k, n):
    low, high = wilson(k, n)
    return f"{k}/{n} (95% CI {100 * low:.0f}-{100 * high:.0f}%)"


def save_confusion(cm, classes, path, title):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--arch", default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Architecture: {args.arch}")

    items = collect_dataset(args.data)
    pilot = collect_pilot()
    print(f"Dataset images found: {len(items)}  Pilot photos: {len(pilot)}")
    if not items:
        print("No images found. Check --data.")
        return

    pilot_hashes = {md5(p) for p, _ in pilot}
    remaining = [(p, l) for p, l in items if md5(p) not in pilot_hashes]
    removed = len(items) - len(remaining)
    print(f"Removed {removed} images that are also in the pilot set (expected {len(pilot)})")
    if pilot and removed != len(pilot):
        print("Warning: the number removed differs from the pilot size, so some pilot photos may still be in training.")

    classes = sorted({label for _, label in remaining})
    print("Classes:", classes)

    labels = [l for _, l in remaining]
    train_items, temp_items = train_test_split(remaining, test_size=0.30, stratify=labels, random_state=SEED)
    temp_labels = [l for _, l in temp_items]
    val_items, test_items = train_test_split(temp_items, test_size=0.50, stratify=temp_labels, random_state=SEED)
    print(f"Split: train {len(train_items)}, val {len(val_items)}, test {len(test_items)}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "classifier_split.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "split"])
        for name, group in (("train", train_items), ("val", val_items), ("test", test_items)):
            for path, label in group:
                writer.writerow([os.path.basename(path), label, name])

    print("Loading images into memory...")
    started = time.time()
    train_ds = LeafDataset(train_items, classes, TRAIN_TF)
    val_ds = LeafDataset(val_items, classes, EVAL_TF)
    test_ds = LeafDataset(test_items, classes, EVAL_TF)
    pilot_ds = LeafDataset(pilot, classes, EVAL_TF) if pilot else None
    print(f"Loaded in {time.time() - started:.0f}s")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch)

    counts = np.bincount([y for _, y in train_ds.samples], minlength=len(classes))
    weights = torch.tensor(len(train_ds) / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    model, head = build_model(args.arch, len(classes))
    model.to(device)
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": 1e-4}, {"params": head.parameters(), "lr": 1e-3}],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, best_state, stale = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, _ = run_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, device, criterion)
        scheduler.step()
        print(f"Epoch {epoch:2d}  train loss {tr_loss:.3f} acc {tr_acc:.3f}  val loss {va_loss:.3f} acc {va_acc:.3f} macro-F1 {va_f1:.3f}")
        if va_f1 > best_f1:
            best_f1, best_state, stale = va_f1, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)

    val_logits, val_y = get_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_y)
    print(f"\nTemperature fitted on validation set: {temperature:.3f}")

    test_logits, test_y = get_logits(model, test_loader, device)
    raw_probs = F.softmax(test_logits, dim=1)
    cal_probs = F.softmax(test_logits / temperature, dim=1)
    test_pred = raw_probs.argmax(1).numpy()

    print("\n=== HELD-OUT TEST SET ===")
    print(classification_report(test_y, test_pred, labels=list(range(len(classes))), target_names=classes, digits=3, zero_division=0))
    test_acc = accuracy_score(test_y, test_pred)
    test_f1 = f1_score(test_y, test_pred, average="macro", zero_division=0)
    ece_raw = expected_calibration_error(raw_probs, test_y)
    ece_cal = expected_calibration_error(cal_probs, test_y)
    print(f"Accuracy {fmt(int((test_pred == test_y).sum()), len(test_y))}  macro-F1 {test_f1:.3f}")
    print(f"Expected calibration error: raw {ece_raw:.3f}, after temperature scaling {ece_cal:.3f}")
    print("Accuracy versus coverage when abstaining below a confidence threshold (calibrated):")
    table = coverage_table(cal_probs, test_y)
    for row in table:
        print(f"  confidence >= {row['threshold']}: coverage {row['coverage']}, accuracy {row['accuracy']}")

    cm = confusion_matrix(test_y, test_pred, labels=list(range(len(classes))))
    save_confusion(cm, classes, os.path.join(PLOTS_DIR, f"classifier_confusion_{args.arch}.png"), f"{args.arch}: held-out test set")

    report = {
        "arch": args.arch,
        "classes": classes,
        "split_sizes": {"train": len(train_items), "val": len(val_items), "test": len(test_items)},
        "test_accuracy": round(float(test_acc), 4),
        "test_macro_f1": round(float(test_f1), 4),
        "ece_raw": round(ece_raw, 4),
        "ece_calibrated": round(ece_cal, 4),
        "temperature": round(temperature, 4),
        "coverage_table": table,
    }

    if pilot_ds is not None:
        pilot_loader = DataLoader(pilot_ds, batch_size=args.batch)
        p_logits, p_y = get_logits(model, pilot_loader, device)
        p_probs = F.softmax(p_logits / temperature, dim=1)
        p_conf, p_pred = p_probs.max(1)
        p_top3 = p_probs.topk(min(3, len(classes)), dim=1).indices.numpy()
        p_pred, p_conf = p_pred.numpy(), p_conf.numpy()

        rows = []
        for (path, label), pred, conf, top3 in zip(pilot, p_pred, p_conf, p_top3):
            rows.append({
                "label": label,
                "file": os.path.basename(path),
                "predicted": classes[pred],
                "confidence": round(float(conf), 3),
                "top3": "|".join(classes[i] for i in top3),
            })
        out_csv = os.path.join(REPORT_DIR, f"classifier_pilot_{args.arch}.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["label", "file", "predicted", "confidence", "top3"])
            writer.writeheader()
            writer.writerows(rows)

        diseased = [r for r in rows if r["label"] in DISEASES]
        healthy = [r for r in rows if r["label"] == "healthy"]
        top1 = sum(1 for r in diseased if r["predicted"] == r["label"])
        top3_hits = sum(1 for r in diseased if r["label"] in r["top3"].split("|"))
        healthy_ok = sum(1 for r in healthy if r["predicted"] == "healthy")
        print("\n=== PILOT PHOTOS (the same 40 used for the vision-model evaluation) ===")
        print(f"Diseased photos: top-1 {fmt(top1, len(diseased))}, top-3 {fmt(top3_hits, len(diseased))}")
        print(f"Healthy photos called healthy: {fmt(healthy_ok, len(healthy))}")
        report["pilot"] = {
            "diseased_top1": top1,
            "diseased_top3": top3_hits,
            "diseased_n": len(diseased),
            "healthy_correct": healthy_ok,
            "healthy_n": len(healthy),
        }

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(
        {"state_dict": best_state, "classes": classes, "arch": args.arch, "temperature": temperature},
        os.path.join(MODELS_DIR, f"wheat_classifier_{args.arch}.pt"),
    )
    with open(os.path.join(REPORT_DIR, f"classifier_report_{args.arch}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved model to {MODELS_DIR}/wheat_classifier_{args.arch}.pt and report to {REPORT_DIR}/classifier_report_{args.arch}.json")


if __name__ == "__main__":
    main()