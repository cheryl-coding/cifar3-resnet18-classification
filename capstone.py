"""Reproducible CIFAR-10 cat/dog/horse transfer-learning capstone.

Usage: python capstone.py prepare|train|evaluate [options]
No AWS service is required; execute inside a single SageMaker notebook or locally.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

CLASSES = ("cat", "dog", "horse")
CIFAR_IDS = {"cat": 3, "dog": 5, "horse": 7}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Avoid nondeterminism when possible; backend/hardware can still affect results.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stratified_indices(train_labels, test_labels, seed=42, ntrain=600, nval=150, ntest=150):
    """Return disjoint records; official CIFAR train/test boundary is never crossed."""
    rng = np.random.default_rng(seed)
    records = []
    for class_index, name in enumerate(CLASSES):
        original_id = CIFAR_IDS[name]
        tr = np.flatnonzero(np.asarray(train_labels) == original_id)
        te = np.flatnonzero(np.asarray(test_labels) == original_id)
        if len(tr) < ntrain + nval or len(te) < ntest:
            raise ValueError(f"Not enough images for {name}")
        tr = rng.permutation(tr)
        te = rng.permutation(te)
        for split, indices, origin in (
            ("train", tr[:ntrain], "train"),
            ("val", tr[ntrain:ntrain + nval], "train"),
            ("test", te[:ntest], "test"),
        ):
            for idx in indices:
                records.append({"split": split, "source": origin, "index": int(idx),
                                "target": class_index, "class_name": name})
    return records


def load_cifar(root, download=False):
    try:
        train = datasets.CIFAR10(root=str(root), train=True, download=download)
        test = datasets.CIFAR10(root=str(root), train=False, download=download)
    except Exception as exc:
        raise RuntimeError("CIFAR-10 unavailable. Run 'python capstone.py prepare --download' "
                           "where internet access is available, or place cifar-10-batches-py "
                           f"inside {root} before starting training. Original error: {exc}") from exc
    return train, test


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_split(path):
    with Path(path).open(newline="") as f:
        records = list(csv.DictReader(f))
    for r in records:
        r["index"] = int(r["index"])
        r["target"] = int(r["target"])
    return records


def plot_eda(records, base_train, output):
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter((r["split"], r["class_name"]) for r in records)
    fig, ax = plt.subplots(figsize=(8, 4.3))
    x = np.arange(3)
    for i, split in enumerate(("train", "val", "test")):
        vals = [counts[(split, cls)] for cls in CLASSES]
        ax.bar(x + (i - 1) * 0.24, vals, width=0.24, label=split)
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("Number of images")
    ax.set_title("Stratified, balanced dataset split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(9, 7))
    for row, cls in enumerate(CLASSES):
        examples = [r for r in records if r["split"] == "train" and r["class_name"] == cls][:4]
        for col, r in enumerate(examples):
            img, _ = base_train[r["index"]]
            ax = axes[row, col]
            ax.imshow(img)
            ax.set_title(cls)
            ax.axis("off")
    fig.suptitle("Representative 32 x 32 CIFAR-10 samples (displayed enlarged)")
    fig.tight_layout()
    fig.savefig(output / "sample_images.png", dpi=180)
    plt.close(fig)


def prepare(args):
    set_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base_train, base_test = load_cifar(args.data_root, download=args.download)
    records = stratified_indices(base_train.targets, base_test.targets, seed=args.seed)
    write_csv(output / "splits.csv", records, ["split", "source", "index", "target", "class_name"])
    plot_eda(records, base_train, output / "figures")
    (output / "dataset_info.json").write_text(json.dumps({
        "source": "CIFAR-10 official training/test subsets", "seed": args.seed,
        "classes": list(CLASSES), "original_pixel_size": [32, 32],
        "preprocessing": "resize 224x224, ImageNet normalization; training-only flip",
        "counts": {s: dict(Counter(r["class_name"] for r in records if r["split"] == s))
                   for s in ("train", "val", "test")},
    }, indent=2))
    print(f"Prepared {len(records)} labeled examples in {output / 'splits.csv'}")


class CifarThreeDataset(Dataset):
    def __init__(self, train_base, test_base, records, split):
        self.rows = [r for r in records if r["split"] == split]
        self.train_base = train_base
        self.test_base = test_base
        aug = [transforms.Resize((224, 224))]
        if split == "train":
            aug.append(transforms.RandomHorizontalFlip(p=0.5))
        aug.extend((transforms.ToTensor(), transforms.Normalize(MEAN, STD)))
        self.transform = transforms.Compose(aug)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        base = self.train_base if r["source"] == "train" else self.test_base
        img, _ = base[r["index"]]
        return self.transform(img), r["target"]


def get_loaders(args, splits=("train", "val", "test")):
    records = read_split(Path(args.output) / "splits.csv")
    base_train, base_test = load_cifar(args.data_root, download=False)
    return {split: DataLoader(CifarThreeDataset(base_train, base_test, records, split),
                              batch_size=args.batch_size, shuffle=(split == "train"),
                              num_workers=args.num_workers)
            for split in splits}


def new_model(mode, weights_path=None):
    if mode not in ("frozen", "fine_tune"):
        raise ValueError(f"Unknown mode: {mode}")
    if weights_path:
        model = models.resnet18(weights=None)
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        # Torchvision's ImageNet checkpoint has a 1000-class classifier.
        model.load_state_dict(state)
    else:
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as exc:
            raise RuntimeError("Pretrained weights could not be downloaded. If your AWS lab has "
                               "restricted internet, supply --weights-file PATH to the "
                               "cached resnet18-f37072fd.pth file.") from exc
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    if mode == "fine_tune":
        for p in model.layer4.parameters():
            p.requires_grad = True
    return model


def epoch_pass(model, loader, device, optimizer=None, mode="frozen"):
    training = optimizer is not None
    if training:
        # Fixed layers' batch-norm statistics should not drift during training.
        model.eval()
        model.fc.train()
        if mode == "fine_tune":
            model.layer4.train()
    else:
        model.eval()
    ys, preds = [], []
    total_loss, count = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        size = labels.size(0)
        total_loss += float(loss.item()) * size
        count += size
        ys.extend(labels.detach().cpu().tolist())
        preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return {"loss": total_loss / max(count, 1),
            "accuracy": accuracy_score(ys, preds),
            "macro_f1": f1_score(ys, preds, average="macro", zero_division=0)}


def train(args):
    set_seed(args.seed)
    loaders = get_loaders(args, ("train", "val"))
    device = choose_device(args.device)
    model = new_model(args.mode, args.weights_file).to(device)
    lr = args.lr if args.lr is not None else (1e-3 if args.mode == "frozen" else 1e-4)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=lr, weight_decay=1e-4)
    out = Path(args.output) / args.mode
    out.mkdir(parents=True, exist_ok=True)
    history = []
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        tr = epoch_pass(model, loaders["train"], device, optimizer, args.mode)
        va = epoch_pass(model, loaders["val"], device)
        record = {"epoch": epoch, "train_loss": tr["loss"],
                  "train_accuracy": tr["accuracy"], "train_macro_f1": tr["macro_f1"],
                  "val_loss": va["loss"], "val_accuracy": va["accuracy"],
                  "val_macro_f1": va["macro_f1"], "seconds": round(time.perf_counter() - start, 2)}
        history.append(record)
        print(json.dumps(record), flush=True)
        if va["macro_f1"] > best_f1:
            best_f1 = va["macro_f1"]
            torch.save({key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                       out / "best_model.pth")
            (out / "training_meta.json").write_text(json.dumps({
                "mode": args.mode, "best_epoch": epoch, "best_val_macro_f1": best_f1,
                "seed": args.seed, "lr": lr, "epochs_requested": args.epochs,
                "batch_size": args.batch_size, "device": str(device),
                "pretraining": "ImageNet1K V1", "pretrained_weights_file": args.weights_file,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
            }, indent=2))
    write_csv(out / "history.csv", history, list(history[0].keys()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for key, ax, label in (("accuracy", axes[0], "Accuracy"), ("loss", axes[1], "Cross-entropy loss")):
        for split in ("train", "val"):
            ax.plot([r["epoch"] for r in history], [r[f"{split}_{key}"] for r in history],
                    marker="o", label=split)
        ax.set_xlabel("Epoch"); ax.set_ylabel(label); ax.set_title(f"{args.mode}: {label}"); ax.legend()
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=180)
    plt.close(fig)
    print(f"Training done. Best validation macro-F1: {best_f1:.4f}. Checkpoint: {out / 'best_model.pth'}")


def bootstrap_accuracy_ci(y_true, y_pred, seed=42, n_boot=2000):
    truth, pred = np.asarray(y_true), np.asarray(y_pred)
    if len(truth) == 0:
        raise ValueError("Cannot bootstrap empty predictions")
    rng = np.random.default_rng(seed)
    correct = (truth == pred).astype(float)
    scores = [float(correct[rng.integers(len(correct), size=len(correct))].mean())
              for _ in range(n_boot)]
    return [float(v) for v in np.quantile(scores, [0.025, 0.975])]


def evaluate(args):
    set_seed(args.seed)
    device = choose_device(args.device)
    loader = get_loaders(args, ("test",))["test"]
    root = Path(args.output) / args.mode
    checkpoint = root / "best_model.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Train {args.mode} first: {checkpoint} not found")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    all_truth, all_pred, all_prob = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            probabilities = model(images.to(device)).softmax(dim=1).cpu().numpy()
            all_truth.extend(labels.tolist())
            all_pred.extend(probabilities.argmax(axis=1).tolist())
            all_prob.extend(probabilities.max(axis=1).tolist())
    cm = confusion_matrix(all_truth, all_pred, labels=[0, 1, 2])
    per_class = classification_report(all_truth, all_pred, target_names=CLASSES,
                                      output_dict=True, zero_division=0)
    info = {"model": args.mode, "dataset": "official CIFAR-10 held-out test subset",
            "test_count": len(all_truth), "accuracy": float(accuracy_score(all_truth, all_pred)),
            "macro_f1": float(f1_score(all_truth, all_pred, average="macro", zero_division=0)),
            "accuracy_ci_95_bootstrap": bootstrap_accuracy_ci(all_truth, all_pred, args.seed),
            "class_report": per_class, "confusion_matrix": cm.tolist()}
    (root / "test_metrics.json").write_text(json.dumps(info, indent=2))
    write_csv(root / "test_predictions.csv", (
        {"test_position": i, "true": CLASSES[t], "pred": CLASSES[p], "confidence": f"{pr:.6f}"}
        for i, (t, p, pr) in enumerate(zip(all_truth, all_pred, all_prob))),
        ["test_position", "true", "pred", "confidence"])
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center", fontsize=13,
                color="white" if val > cm.max() / 2 else "black")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("Actual label")
    ax.set_title(f"{args.mode}: confusion matrix (n={len(all_truth)})")
    fig.tight_layout(); fig.savefig(root / "confusion_matrix.png", dpi=180); plt.close(fig)
    print(json.dumps({k: info[k] for k in ("model", "test_count", "accuracy", "macro_f1", "accuracy_ci_95_bootstrap")}, indent=2))
    print(f"Saved {root / 'test_metrics.json'}")


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for cmd in ("prepare", "train", "evaluate"):
        q = sub.add_parser(cmd)
        q.add_argument("--output", default="runs")
        q.add_argument("--data-root", default="data")
        q.add_argument("--seed", type=int, default=42)
        if cmd == "prepare":
            q.add_argument("--download", action="store_true", help="Download CIFAR-10 if absent")
        else:
            q.add_argument("--mode", choices=("frozen", "fine_tune"), required=True)
            q.add_argument("--batch-size", type=int, default=32)
            q.add_argument("--num-workers", type=int, default=0)
            q.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
            if cmd == "train":
                q.add_argument("--epochs", type=int, default=3)
                q.add_argument("--lr", type=float, default=None)
                q.add_argument("--weights-file", default=None,
                               help="Offline original ImageNet ResNet18 state_dict file")
    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    {"prepare": prepare, "train": train, "evaluate": evaluate}[args.command](args)
