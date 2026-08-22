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


class SingleImageDataset(torch.utils.data.Dataset):
    """De-duplicated view over the pair dataset — one entry per distinct image."""

    def __init__(self, config, split="train"):
        base = CrossSensorFingerprintDataset(config, split=split)
        self.image_size = base.image_size
        self.items = base.unique_source_images()
        if not self.items:   # synthetic fallback (tests / --synthetic)
            self.items = [{"cache_key": f"syn_{i}", "path": ""} for i in range(100)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path = self.items[idx]["path"]
        if path:
            img = load_canonical_image(path, self.image_size)
        else:
            img = np.random.randint(50, 200, (self.image_size,) * 2, dtype=np.uint8)
        return {"img_A": torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Fingerprint VAE")
    parser.add_argument("--config", type=str, default="./configs/default_config.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl_weight", type=float, default=1e-4,
                        help="Beta for KL divergence (keep small for sharp reconstruction)")
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

    optimizer = optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        vae.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch}")

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
    log_writer.writerow(["epoch", "step", "loss_total", "loss_recon", "loss_kl", "lr"])

    # Training loop
    best_loss = float("inf")
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        vae.train()
        epoch_losses = {"total": 0, "recon": 0, "kl": 0}
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

            # Backward
            optimizer.zero_grad()
            loss_dict["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate
            epoch_losses["total"] += loss_dict["loss_total"].item()
            epoch_losses["recon"] += loss_dict["loss_recon"].item()
            epoch_losses["kl"] += loss_dict["loss_kl"].item()
            num_batches += 1
            global_step += 1

            # Log every 10 steps
            if (step + 1) % 10 == 0 or step == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} Step {step+1}/{len(dataloader)} | "
                      f"Loss: {loss_dict['loss_total'].item():.5f} "
                      f"(Recon: {loss_dict['loss_recon'].item():.5f}, "
                      f"KL: {loss_dict['loss_kl'].item():.3f}) | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            log_writer.writerow([
                epoch, global_step,
                f"{loss_dict['loss_total'].item():.6f}",
                f"{loss_dict['loss_recon'].item():.6f}",
                f"{loss_dict['loss_kl'].item():.6f}",
                f"{optimizer.param_groups[0]['lr']:.2e}"
            ])

        scheduler.step()

        # Epoch summary
        avg_total = epoch_losses["total"] / max(num_batches, 1)
        avg_recon = epoch_losses["recon"] / max(num_batches, 1)
        avg_kl = epoch_losses["kl"] / max(num_batches, 1)
        elapsed = time.time() - t_epoch

        print(f"\n  Epoch {epoch+1} Summary: "
              f"Avg Loss={avg_total:.5f} (Recon={avg_recon:.5f}, KL={avg_kl:.3f}) "
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
