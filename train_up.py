"""
ResNet50 + ArcFace for Fingerprint Verification (SD302a)
=========================================================

Train a feature extractor with ArcFace loss, evaluate with
verification metrics (TAR@FAR, EER, Rank-1).

Supports two modes of comparison:
  A) Train on SD302a train split (1600 ids) → baseline
  B) Train on synthetic data → measure synthetic quality

Both evaluated on SD302a val/test (200 ids, unseen during training).

Usage:
    # Baseline: train on SD302a
    python train_resnet50.py --train_dir data/sd302a_811/train --test_dir data/sd302a_811/test

    # Synthetic: train on your generated data
    python train_resnet50.py --train_dir path/to/synthetic --test_dir data/sd302a_811/test

    # Resume
    python train_resnet50.py --resume checkpoints/best_model.pth
"""

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, models
from PIL import Image


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ═════════════════════════════════════════════════════════════════════
#  ArcFace Head
# ═════════════════════════════════════════════════════════════════════

class ArcFaceHead(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) — numerically stable version.
    Uses cos(θ+m) = cosθ·cosm - sinθ·sinm to avoid torch.acos.
    Includes monotonicity guard for θ+m > π.
    """

    def __init__(self, embedding_dim, num_classes, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # Threshold for monotonicity: cos(π - m)
        self.threshold = math.cos(math.pi - m)
        # Fallback penalty: sin(m) * m
        self.mm = self.sin_m * m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.W)

    def set_margin(self, m):
        """Update margin (for margin warmup)."""
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = self.sin_m * m

    def forward(self, embeddings, labels):
        W = F.normalize(self.W, dim=1)
        x = F.normalize(embeddings, dim=1)
        cosine = F.linear(x, W).clamp(-1.0 + 1e-7, 1.0 - 1e-7)  # (B, C)

        # cos(θ+m) via identity, no acos needed
        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(θ+m)

        # Safety: when cosθ < cos(π-m), i.e. θ+m > π, use linear fallback
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.s
        return logits


# ═════════════════════════════════════════════════════════════════════
#  Model: ResNet50 Backbone + Embedding
# ═════════════════════════════════════════════════════════════════════

class FingerprintNet(nn.Module):
    """
    ResNet50 backbone → 512-dim normalized embedding.
    During training, used with ArcFaceHead.
    During inference, outputs L2-normalized embeddings for matching.
    """

    def __init__(self, embedding_dim=512, pretrained=True):
        super().__init__()
        if pretrained:
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            backbone = models.resnet50(weights=None)

        # Remove original FC
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 2048, 1, 1)
        self.embedding = nn.Linear(2048, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)                      # (B, 2048)
        x = self.embedding(x)                 # (B, 512)
        x = self.bn(x)
        return x  # NOT normalized here; ArcFace normalizes internally

    def extract(self, x):
        """Extract L2-normalized embedding for verification."""
        x = self.forward(x)
        return F.normalize(x, dim=1)


# ═════════════════════════════════════════════════════════════════════
#  Dataset
# ═════════════════════════════════════════════════════════════════════

def scan_imagefolder(root: Path):
    """
    Scan ImageFolder-style directory:  root/identity_name/image.ext
    Returns: identity_to_images dict, class_to_idx dict
    """
    identity_to_images = defaultdict(list)
    for identity_dir in sorted(root.iterdir()):
        if not identity_dir.is_dir():
            continue
        for img_path in sorted(identity_dir.iterdir()):
            if img_path.suffix.lower() in VALID_EXTS:
                identity_to_images[identity_dir.name].append(str(img_path))

    class_to_idx = {name: idx for idx, name in enumerate(sorted(identity_to_images.keys()))}
    return identity_to_images, class_to_idx


class FingerprintDataset(Dataset):
    def __init__(self, items, transform=None):
        """items: list of (path, class_idx)"""
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ═════════════════════════════════════════════════════════════════════
#  Transforms
# ═════════════════════════════════════════════════════════════════════

def get_transforms(image_size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ═════════════════════════════════════════════════════════════════════
#  Verification Evaluation
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_all_embeddings(model, loader, device):
    """Extract embeddings and labels for all images in loader."""
    model.eval()
    all_embeddings = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        emb = model.extract(images)
        all_embeddings.append(emb.cpu())
        all_labels.append(labels)

    return torch.cat(all_embeddings), torch.cat(all_labels)


def compute_verification_metrics(embeddings, labels, num_pairs=50000, seed=42):
    """
    Generate genuine/impostor pairs, compute TAR@FAR and EER.
    """
    rng = random.Random(seed)
    labels_np = labels.numpy()

    # Group indices by label
    label_to_indices = defaultdict(list)
    for i, lab in enumerate(labels_np):
        label_to_indices[lab].append(i)

    # Generate genuine pairs (same identity)
    genuine_scores = []
    genuine_identities = [lab for lab, inds in label_to_indices.items() if len(inds) >= 2]

    for _ in range(num_pairs // 2):
        lab = rng.choice(genuine_identities)
        i, j = rng.sample(label_to_indices[lab], 2)
        sim = F.cosine_similarity(embeddings[i:i+1], embeddings[j:j+1]).item()
        genuine_scores.append(sim)

    # Generate impostor pairs (different identity)
    impostor_scores = []
    all_labels_list = list(label_to_indices.keys())

    for _ in range(num_pairs // 2):
        lab1, lab2 = rng.sample(all_labels_list, 2)
        i = rng.choice(label_to_indices[lab1])
        j = rng.choice(label_to_indices[lab2])
        sim = F.cosine_similarity(embeddings[i:i+1], embeddings[j:j+1]).item()
        impostor_scores.append(sim)

    genuine = np.array(genuine_scores)
    impostor = np.array(impostor_scores)

    # Sweep thresholds to compute FAR/FRR
    thresholds = np.linspace(-1.0, 1.0, 2000)
    fars = []
    frrs = []
    for t in thresholds:
        far = np.mean(impostor >= t)   # False Accept Rate
        frr = np.mean(genuine < t)     # False Reject Rate
        fars.append(far)
        frrs.append(frr)
    fars = np.array(fars)
    frrs = np.array(frrs)
    tars = 1.0 - frrs

    # EER: where FAR ≈ FRR
    eer_idx = np.argmin(np.abs(fars - frrs))
    eer = (fars[eer_idx] + frrs[eer_idx]) / 2.0

    # TAR @ specific FARs
    def tar_at_far(target_far):
        idx = np.searchsorted(fars[::-1], target_far)
        idx = len(fars) - 1 - idx
        idx = max(0, min(idx, len(tars) - 1))
        return tars[idx]

    metrics = {
        "EER": float(eer) * 100,
        "TAR@FAR=1%": float(tar_at_far(0.01)) * 100,
        "TAR@FAR=0.1%": float(tar_at_far(0.001)) * 100,
        "TAR@FAR=0.01%": float(tar_at_far(0.0001)) * 100,
    }
    return metrics


def compute_rank1(embeddings, labels):
    """
    Rank-1 identification: for each probe, find closest gallery match.
    Use leave-one-out protocol.
    """
    labels_np = labels.numpy()
    # Cosine similarity matrix
    sim_matrix = torch.mm(embeddings, embeddings.t())
    # Mask diagonal (don't match with self)
    sim_matrix.fill_diagonal_(-2.0)

    # For each query, find the index of max similarity
    _, top_idx = sim_matrix.max(dim=1)
    predicted_labels = labels_np[top_idx.numpy()]

    rank1 = np.mean(predicted_labels == labels_np) * 100
    return rank1


# ═════════════════════════════════════════════════════════════════════
#  Training
# ═════════════════════════════════════════════════════════════════════

def train_one_epoch(model, arc_head, loader, criterion, optimizer,
                    scaler, device, use_amp, epoch, total_epochs):
    model.train()
    arc_head.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(loader)
    log_interval = max(1, num_batches // 5)

    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            embeddings = model(images)
            logits = arc_head(embeddings, labels)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(arc_head.parameters()),
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(arc_head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, pred = logits.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

        if (i + 1) % log_interval == 0:
            print(f"  [{epoch+1}/{total_epochs}] [{i+1}/{num_batches}] "
                  f"Loss: {running_loss/total:.4f}  Acc: {100.*correct/total:.2f}%")

    return running_loss / total, 100. * correct / total


# ═════════════════════════════════════════════════════════════════════
#  Args & Main
# ═════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="ResNet50 + ArcFace Fingerprint Verification")

    # Data
    p.add_argument("--train_dir", type=str, default="data/sd302a_811/train",
                    help="Training data (ImageFolder: identity/image.ext)")
    p.add_argument("--val_dir", type=str, default="data/sd302a_811/val",
                    help="Validation data for verification eval")
    p.add_argument("--test_dir", type=str, default="data/sd302a_811/test",
                    help="Test data for final verification eval")

    # Model
    p.add_argument("--embedding_dim", type=int, default=512)
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--arcface_s", type=float, default=30.0, help="ArcFace scale")
    p.add_argument("--arcface_m", type=float, default=0.5, help="ArcFace margin")

    # Training
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--margin_warmup_epochs", type=int, default=10,
                    help="Epochs to linearly ramp ArcFace margin from 0 to target")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true")

    # Checkpointing
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save_every", type=int, default=5)

    # Wandb
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="fingerprint-arcface")
    p.add_argument("--wandb_name", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  ResNet50 + ArcFace Fingerprint Verification")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Load train data ──────────────────────────────────────────
    train_dir = Path(args.train_dir)
    identity_to_images, class_to_idx = scan_imagefolder(train_dir)
    num_train_classes = len(class_to_idx)

    train_items = []
    for identity, paths in identity_to_images.items():
        idx = class_to_idx[identity]
        for p in paths:
            train_items.append((p, idx))

    print(f"\n  Train dir: {train_dir}")
    print(f"  Train classes: {num_train_classes}")
    print(f"  Train images: {len(train_items)}")

    # ── Load val/test data ───────────────────────────────────────
    val_dir = Path(args.val_dir) if args.val_dir else None
    test_dir = Path(args.test_dir) if args.test_dir else None

    train_tf, val_tf = get_transforms(args.image_size)
    train_ds = FingerprintDataset(train_items, train_tf)

    pw = args.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=pw,
    )

    def make_eval_loader(data_dir):
        id2img, c2i = scan_imagefolder(data_dir)
        items = [(p, c2i[name]) for name, paths in id2img.items() for p in paths]
        ds = FingerprintDataset(items, val_tf)
        return DataLoader(ds, batch_size=args.batch_size * 2, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True), len(c2i)

    val_loader, val_classes = (None, 0)
    if val_dir and val_dir.exists():
        val_loader, val_classes = make_eval_loader(val_dir)
        print(f"  Val dir: {val_dir} ({val_classes} ids, {len(val_loader.dataset)} imgs)")

    test_loader, test_classes = (None, 0)
    if test_dir and test_dir.exists():
        test_loader, test_classes = make_eval_loader(test_dir)
        print(f"  Test dir: {test_dir} ({test_classes} ids, {len(test_loader.dataset)} imgs)")

    # ── Model + ArcFace ──────────────────────────────────────────
    print(f"\n  Building model (emb_dim={args.embedding_dim})...")
    model = FingerprintNet(
        embedding_dim=args.embedding_dim,
        pretrained=not args.no_pretrained,
    ).to(device)

    arc_head = ArcFaceHead(
        embedding_dim=args.embedding_dim,
        num_classes=num_train_classes,
        s=args.arcface_s,
        m=args.arcface_m,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Backbone params: {n_params:,}")
    print(f"  ArcFace: s={args.arcface_s}, m={args.arcface_m}")

    # ── Optimizer / Scheduler ────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        list(model.parameters()) + list(arc_head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6,
    )

    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    print(f"  LR: {args.lr} | Warmup: {args.warmup_epochs}ep | AMP: {'ON' if use_amp else 'OFF'}")

    # ── Resume ───────────────────────────────────────────────────
    start_epoch = 0
    best_eer = 100.0
    history = []

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "arc_head_state_dict" in ckpt:
            arc_head.load_state_dict(ckpt["arc_head_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_eer = ckpt.get("best_eer", 100.0)
        history = ckpt.get("history", [])
        print(f"  Resumed from epoch {start_epoch}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Wandb init ───────────────────────────────────────────────
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb chưa cài. Chạy: pip install wandb")
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
        )

    # ── Training loop ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training for {args.epochs} epochs")
    print(f"{'='*60}\n")

    target_margin = args.arcface_m

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Linear LR warmup
        if epoch < args.warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * (epoch + 1) / args.warmup_epochs

        # Margin warmup: ramp from 0 → target_margin
        if epoch < args.margin_warmup_epochs:
            current_m = target_margin * (epoch + 1) / args.margin_warmup_epochs
            arc_head.set_margin(current_m)
            if epoch == 0 or (epoch + 1) == args.margin_warmup_epochs:
                print(f"  ArcFace margin: {current_m:.3f}")

        train_loss, train_acc = train_one_epoch(
            model, arc_head, train_loader, criterion, optimizer,
            scaler, device, use_amp, epoch, args.epochs,
        )

        if epoch >= args.warmup_epochs:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        # Verification eval on val
        val_metrics = {}
        val_rank1 = 0.0
        if val_loader is not None:
            emb, lab = extract_all_embeddings(model, val_loader, device)
            val_metrics = compute_verification_metrics(emb, lab)
            val_rank1 = compute_rank1(emb, lab)

        epoch_log = {
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "lr": lr, **{f"val_{k}": v for k, v in val_metrics.items()},
            "val_rank1": val_rank1,
        }
        history.append(epoch_log)

        eer = val_metrics.get("EER", 100.0)
        tar1 = val_metrics.get("TAR@FAR=1%", 0.0)

        print(f"\n  Epoch {epoch+1}/{args.epochs} ({dt:.0f}s)")
        print(f"  Train: Loss={train_loss:.4f} Acc={train_acc:.2f}%")
        if val_metrics:
            print(f"  Val:   EER={eer:.2f}% | TAR@FAR=1%={tar1:.2f}% | Rank-1={val_rank1:.2f}%")
        print(f"  LR: {lr:.2e}")

        # Save best (lowest EER)
        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "arc_head_state_dict": arc_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_eer": min(best_eer, eer),
            "embedding_dim": args.embedding_dim,
            "num_train_classes": num_train_classes,
            "history": history,
            "args": vars(args),
        }

        if eer < best_eer:
            best_eer = eer
            save_dict["best_eer"] = best_eer
            torch.save(save_dict, ckpt_dir / "best_model.pth")
            print(f"  >>> Best model! EER={best_eer:.2f}%")

        if (epoch + 1) % args.save_every == 0:
            torch.save(save_dict, ckpt_dir / f"epoch_{epoch+1:03d}.pth")

        if args.use_wandb and wandb is not None:
            wandb.log({
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/EER": eer,
                "val/TAR@FAR=1%": tar1,
                "val/Rank-1": val_rank1,
                "lr": lr,
            }, step=epoch + 1)

        print()

    # ── Final test evaluation ────────────────────────────────────
    if test_loader is not None:
        print(f"{'='*60}")
        print(f"  Final Test Evaluation")
        print(f"{'='*60}")

        best_path = ckpt_dir / "best_model.pth"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded best model (epoch {ckpt['epoch']+1})")

        emb, lab = extract_all_embeddings(model, test_loader, device)
        test_metrics = compute_verification_metrics(emb, lab)
        test_rank1 = compute_rank1(emb, lab)

        print(f"\n  EER:          {test_metrics['EER']:.2f}%")
        print(f"  TAR@FAR=1%:   {test_metrics['TAR@FAR=1%']:.2f}%")
        print(f"  TAR@FAR=0.1%: {test_metrics['TAR@FAR=0.1%']:.2f}%")
        print(f"  Rank-1:       {test_rank1:.2f}%")

        results = {
            "test": {**test_metrics, "rank1": test_rank1},
            "best_val_eer": best_eer,
            "num_train_classes": num_train_classes,
            "args": vars(args),
        }
        with open(ckpt_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n  Results saved to {ckpt_dir}/results.json")

        if args.use_wandb and wandb is not None:
            wandb.log({
                "test/EER": test_metrics["EER"],
                "test/TAR@FAR=1%": test_metrics["TAR@FAR=1%"],
                "test/TAR@FAR=0.1%": test_metrics["TAR@FAR=0.1%"],
                "test/Rank-1": test_rank1,
            })

    if args.use_wandb and wandb is not None:
        wandb.finish()

    print(f"\n  Done!\n")


if __name__ == "__main__":
    main()
