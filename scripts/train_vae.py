"""
Train the Fingerprint VAE (Phase 1).

Must be run BEFORE training MM-DiT, since MM-DiT operates in VAE latent space.

Usage:
    python scripts/train_vae.py --config configs/default_config.yaml --epochs 20
    python scripts/train_vae.py --config configs/default_config.yaml --epochs 5 --synthetic

Output:
    weights/vae_fingerprint.pt
"""

import sys
import os
import argparse
import time
import csv
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.vae import FingerprintVAE
from src.data.dataset import CrossSensorFingerprintDataset, load_canonical_image
from src.losses.vae_perceptual_loss import VAEPerceptualLoss
from src.losses.orientation_loss import OrientationCoherenceLoss


class SingleImageDataset(torch.utils.data.Dataset):
    """
    De-duplicated view over the pair dataset — one entry per distinct image.

    Also loads each image's cached Stage-1 `S_aligned` map (orientation channels)
    when available, for the optional `L_Orient` term — items whose source image
    has no Stage-1 cache are dropped rather than fed a random/zero map, same
    safeguard `CrossSensorFingerprintDataset(require_cache=True)` uses.
    """

    def __init__(self, config, split="train"):
        base = CrossSensorFingerprintDataset(config, split=split)
        self.image_size = base.image_size
        self.cached_dir = config.get("dataset", {}).get("cached_prep_dir", "./data/cached_stage1")
        items = base.unique_source_images()
        if not items:   # synthetic fallback (tests / --synthetic)
            self.items = [{"cache_key": f"syn_{i}", "path": "", "has_cache": False} for i in range(100)]
            return

        self.items = []
        for it in items:
            cache_path = os.path.join(self.cached_dir, f"{it['cache_key']}_stage1.pt")
            it = dict(it, has_cache=os.path.exists(cache_path))
            self.items.append(it)
        n_missing = sum(1 for it in self.items if not it["has_cache"])
        if n_missing:
            print(f"[SingleImageDataset] {n_missing}/{len(self.items)} image(s) missing Stage-1 "
                  f"cache — L_Orient skipped for those (reconstruction/perceptual loss unaffected).")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        path = item["path"]
        if path:
            img = load_canonical_image(path, self.image_size)
        else:
            img = np.random.randint(50, 200, (self.image_size,) * 2, dtype=np.uint8)
        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)

        if item["has_cache"]:
            cache_path = os.path.join(self.cached_dir, f"{item['cache_key']}_stage1.pt")
            cached = torch.load(cache_path, map_location="cpu")
            S = cached["S_aligned"]
            if S.dim() == 4:
                S = S.squeeze(0)
            if S.shape[-1] != self.image_size:
                S = F.interpolate(S.unsqueeze(0), size=(self.image_size, self.image_size),
                                   mode="bilinear", align_corners=False).squeeze(0)
        else:
            S = torch.zeros(6, self.image_size, self.image_size)

        return {"img_A": img_t, "S_aligned": S, "has_cache": torch.tensor(item["has_cache"])}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Fingerprint VAE")
    parser.add_argument("--config", type=str, default="./configs/default_config.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl_weight", type=float, default=1e-4,
                        help="Beta for KL divergence (keep small for sharp reconstruction)")
    parser.add_argument("--perceptual_weight", type=float, default=0.06,
                        help="Weight for DMD dense-feature perceptual loss (0 = disabled). "
                             "Was 0.1; lowered after backprop-through-DMD's strided convs turned "
                             "out to be a second source of grid/checkerboard-style artifacts on "
                             "top of the (now-fixed) decoder ConvTranspose2d one — see "
                             "VAEPerceptualLoss's built-in anti-alias prefilter, which addresses "
                             "the same issue from the other side.")
    parser.add_argument("--orient_weight", type=float, default=0.05,
                        help="Weight for orientation-coherence loss against cached Stage-1 "
                             "S_aligned maps (0 = disabled). Note: this term's floor is ~0.9-1.0 "
                             "even for the real image vs. its own S_aligned (Sobel-based, noisy), "
                             "not near 0 — judge it by relative change, not absolute value.")
    parser.add_argument("--dmd_checkpoint", type=str, default="./weights/dmd.pt")
    parser.add_argument("--output", type=str, default="./weights/vae_fingerprint.pt")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force synthetic data even if real data exists")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max_steps_per_epoch", type=int, default=0,
                        help="Cap steps per epoch (0 = full epoch); use for smoke tests")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("PHASE 1: TRAINING FINGERPRINT VAE")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"KL weight (beta): {args.kl_weight}")
    print(f"Output: {args.output}")

    # Model
    vae = FingerprintVAE(
        in_channels=1,
        latent_channels=config.get("vae", {}).get("latent_channels", 4),
        base_channels=64
    ).to(device)

    total_params = sum(p.numel() for p in vae.parameters())
    print(f"VAE params: {total_params:,}")

    perceptual_loss_fn = None
    if args.perceptual_weight > 0:
        perceptual_loss_fn = VAEPerceptualLoss(checkpoint_path=args.dmd_checkpoint).to(device)
    orient_loss_fn = OrientationCoherenceLoss().to(device) if args.orient_weight > 0 else None
    print(f"Perceptual weight: {args.perceptual_weight} | Orient weight: {args.orient_weight}")

    optimizer = optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    resumed_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        # Plain `load_state_dict(strict=False)` only tolerates NAME mismatches
        # (missing/unexpected keys) — a key present in both checkpoint and model
        # under the *same name* but a different shape (e.g. `decoder.up1`, which
        # kept its name across the 8x->4x latent change but now maps to a
        # different channel count) still raises, strict or not: PyTorch checks
        # shape before copying and always raises if any shape mismatches, only
        # `strict` gates the missing/unexpected-key checks. Filter to
        # name-*and*-shape matches ourselves so an architecture change can't
        # crash the resume; anything filtered out reinitializes from scratch.
        model_sd = vae.state_dict()
        ckpt_sd = ckpt["model_state_dict"]
        compatible = {k: v for k, v in ckpt_sd.items()
                      if k in model_sd and v.shape == model_sd[k].shape}
        skipped_shape = [k for k in ckpt_sd if k in model_sd and k not in compatible]
        missing, unexpected = vae.load_state_dict(compatible, strict=False)
        if missing or unexpected or skipped_shape:
            print(f"[resume] Non-strict load: {len(missing)} missing / {len(unexpected)} "
                  f"unexpected / {len(skipped_shape)} shape-mismatched keys (expected after "
                  f"an architecture change — those layers reinit from scratch, everything "
                  f"else carries over).")
        if missing or unexpected or skipped_shape:
            # A shape/name mismatch here means the saved per-param Adam momentum
            # buffers (exp_avg/exp_avg_sq) won't match the new params' shapes either
            # — load_state_dict itself won't raise (it only matches by position), the
            # mismatch instead surfaces later, deep inside optimizer.step(). Skip it
            # outright rather than risk a crash mid-epoch; a fresh AdamW state for
            # the reinitialized layers is what we want anyway.
            print("[resume] Skipping optimizer state (architecture changed) — "
                  "starting with a fresh optimizer.")
        else:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # CosineAnnealingLR.get_lr() is recursive (each step multiplies the *current*
        # group['lr'] by a ratio, it does not jump back to base_lr except in a couple
        # of special-cased steps) — the line above just overwrote 'lr' with whatever
        # the old run's schedule had decayed to (near its own eta_min), and the
        # scheduler would silently keep multiplying that near-zero value by ~1 for
        # most of the new T_max range, i.e. --lr is ignored for the entire resumed
        # run. Force both the live lr and the scheduler's cached base_lrs back to
        # the value this run was actually asked for.
        for g in optimizer.param_groups:
            g["lr"] = args.lr
            g["initial_lr"] = args.lr
        scheduler.base_lrs = [args.lr for _ in optimizer.param_groups]
        start_epoch = ckpt.get("epoch", 0) + 1
        # Same root cause as the base_lrs fix above: T_max was set to args.epochs
        # (the absolute target), but the training loop below only runs
        # `args.epochs - start_epoch` iterations post-resume, each advancing the
        # scheduler's own step counter by 1 starting from 0 — so the decay only
        # ever covers that fraction of the intended T_max span and ends well above
        # eta_min. Rescale T_max to the epoch count this run will actually execute.
        scheduler.T_max = max(1, args.epochs - start_epoch)
        # `--output` is often the same path as `--resume` (iteratively continuing the
        # same experimental checkpoint) — without seeding best_loss from it, the
        # *first* post-resume epoch would unconditionally overwrite a good checkpoint
        # with a possibly-worse one (best_loss otherwise starts at +inf every run).
        resumed_loss = ckpt.get("loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, LR reset to {args.lr:.2e} "
              f"(fresh {args.epochs - start_epoch}-epoch cosine decay to {scheduler.eta_min:.2e}), "
              f"best_loss seeded at {resumed_loss:.5f}")

    # Dataset — the VAE only needs individual images, so train on the de-duplicated
    # image list (~12k) rather than the ~72k cross-sensor pairs, which would show the
    # same image dozens of times per epoch.
    dataset = SingleImageDataset(config, split="train")
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=config["dataset"].get("num_workers", 0),
        pin_memory=(device == "cuda"), drop_last=True,
        persistent_workers=config["dataset"].get("num_workers", 0) > 0
    )
    print(f"Dataset: {len(dataset)} unique images, {len(dataloader)} batches/epoch")

    # Logging
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "vae_train_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "step", "loss_total", "loss_recon", "loss_kl",
                          "loss_perceptual", "loss_orient", "lr"])

    # Training loop
    best_loss = resumed_loss
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        vae.train()
        epoch_losses = {"total": 0, "recon": 0, "kl": 0, "perceptual": 0, "orient": 0}
        num_batches = 0
        t_epoch = time.time()

        for step, batch in enumerate(dataloader):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            # Use img_A for VAE training (any single fingerprint image)
            img = batch["img_A"].to(device)

            # Ensure proper size
            if img.shape[-1] != 256 or img.shape[-2] != 256:
                img = F.interpolate(img, size=(256, 256), mode="bilinear", align_corners=False)

            # Forward
            recon, mu, logvar = vae(img)
            loss_dict = FingerprintVAE.loss(recon, img, mu, logvar, kl_weight=args.kl_weight)
            loss_total = loss_dict["loss_total"]

            loss_perceptual = torch.tensor(0.0, device=device)
            if perceptual_loss_fn is not None:
                loss_perceptual = perceptual_loss_fn(recon, img)
                loss_total = loss_total + args.perceptual_weight * loss_perceptual

            loss_orient = torch.tensor(0.0, device=device)
            if orient_loss_fn is not None:
                has_cache = batch["has_cache"].to(device)
                if has_cache.any():
                    S_aligned = batch["S_aligned"].to(device)
                    # Compute per-sample then average only over samples with a real
                    # cached map — a synthetic/zero S_aligned would otherwise train
                    # against garbage supervision for the uncached remainder.
                    per_sample = torch.stack([
                        orient_loss_fn(recon[i:i+1], S_aligned[i:i+1])
                        for i in range(recon.shape[0]) if has_cache[i]
                    ])
                    loss_orient = per_sample.mean()
                    loss_total = loss_total + args.orient_weight * loss_orient

            # Backward
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate
            epoch_losses["total"] += loss_total.item()
            epoch_losses["recon"] += loss_dict["loss_recon"].item()
            epoch_losses["kl"] += loss_dict["loss_kl"].item()
            epoch_losses["perceptual"] += loss_perceptual.item()
            epoch_losses["orient"] += loss_orient.item()
            num_batches += 1
            global_step += 1

            # Log every 10 steps
            if (step + 1) % 10 == 0 or step == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} Step {step+1}/{len(dataloader)} | "
                      f"Loss: {loss_total.item():.5f} "
                      f"(Recon: {loss_dict['loss_recon'].item():.5f}, "
                      f"KL: {loss_dict['loss_kl'].item():.3f}, "
                      f"Perc: {loss_perceptual.item():.4f}, "
                      f"Orient: {loss_orient.item():.4f}) | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            log_writer.writerow([
                epoch, global_step,
                f"{loss_total.item():.6f}",
                f"{loss_dict['loss_recon'].item():.6f}",
                f"{loss_dict['loss_kl'].item():.6f}",
                f"{loss_perceptual.item():.6f}",
                f"{loss_orient.item():.6f}",
                f"{optimizer.param_groups[0]['lr']:.2e}"
            ])

        scheduler.step()

        # Epoch summary
        avg_total = epoch_losses["total"] / max(num_batches, 1)
        avg_recon = epoch_losses["recon"] / max(num_batches, 1)
        avg_kl = epoch_losses["kl"] / max(num_batches, 1)
        avg_perceptual = epoch_losses["perceptual"] / max(num_batches, 1)
        avg_orient = epoch_losses["orient"] / max(num_batches, 1)
        elapsed = time.time() - t_epoch

        print(f"\n  Epoch {epoch+1} Summary: "
              f"Avg Loss={avg_total:.5f} (Recon={avg_recon:.5f}, KL={avg_kl:.3f}, "
              f"Perc={avg_perceptual:.4f}, Orient={avg_orient:.4f}) "
              f"| Time: {elapsed:.1f}s")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": vae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_total,
        }

        # Save latest
        latest_path = os.path.join(output_dir, "vae_latest.pt")
        torch.save(checkpoint, latest_path)

        # Save best
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(checkpoint, args.output)
            print(f"  >> Saved best model (loss={best_loss:.5f}) to {args.output}")

    log_file.close()
    print(f"\nVAE training complete. Best loss: {best_loss:.5f}")
    print(f"Weights saved to: {args.output}")
    print(f"Training log saved to: {log_path}")

    # Final reconstruction test
    vae.eval()
    with torch.no_grad():
        test_img = next(iter(dataloader))["img_A"][:1].to(device)
        if test_img.shape[-1] != 256:
            test_img = F.interpolate(test_img, size=(256, 256), mode="bilinear", align_corners=False)
        recon, _, _ = vae(test_img)
        recon_mse = F.mse_loss(recon, test_img).item()
        print(f"\nFinal reconstruction MSE: {recon_mse:.6f}")
        print(f"Pixel range: input [{test_img.min():.3f}, {test_img.max():.3f}] "
              f"-> recon [{recon.min():.3f}, {recon.max():.3f}]")


if __name__ == "__main__":
    main()
