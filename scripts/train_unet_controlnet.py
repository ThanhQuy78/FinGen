"""
Training script for the UNet + ControlNet + Flow-Matching backbone
(src/models/unet_controlnet.py) — an alternative to train_mmdit.py's
DualStreamMMDiT. See the module docstring in unet_controlnet.py for why the
identity/structure signal moves from MM-DiT's dual-stream cross-attention to
a ControlNet-style additive zero-conv side-branch on a UNet.

Everything except the model class, its forward-call signature, and the
output directory is unchanged from train_mmdit.py: same VAE, same
RectifiedFlowTrajectoryManager (already backbone-agnostic), same
Stage4CompositeLossBuilder (image-space losses, architecture-blind), same
dataset/dataloaders, same EMA/scheduler/checkpoint machinery.

Phase 2: Requires a trained VAE (run train_vae.py first).

Usage:
    python scripts/train_unet_controlnet.py --config configs/unet_controlnet_config.yaml
    python scripts/train_unet_controlnet.py --config configs/unet_controlnet_config.yaml --small
    python scripts/train_unet_controlnet.py --config configs/unet_controlnet_config.yaml --resume outputs/training_unet_controlnet/latest.pt
"""

import sys
import os
import argparse
import time
import csv
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.unet_controlnet import UNetControlNetDenoiser
from src.models.vae import FingerprintVAE
from src.models.flow_matching import RectifiedFlowTrajectoryManager
from src.losses.loss_builder import Stage4CompositeLossBuilder
from src.data.dataset import CrossSensorFingerprintDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet+ControlNet for fingerprint generation")
    parser.add_argument("--config", type=str, default="./configs/unet_controlnet_config.yaml")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume from")
    parser.add_argument("--small", action="store_true",
                        help="Use lightweight model (32ch/2 levels/1 res block) for CPU/small GPU")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=0,
                        help="Override training.epochs (0 = use config)")
    parser.add_argument("--max_steps_per_epoch", type=int, default=0,
                        help="Cap train/val steps per epoch (0 = full); use for smoke tests")
    parser.add_argument("--min_lr_ratio", type=float, default=0.05,
                        help="Cosine decay floor as a fraction of the peak LR (post-warmup)")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Override training.batch_size (0 = use config)")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Accumulate gradients over N steps before optimizer.step()")
    return parser.parse_args()


class EMAModel:
    """Exponential Moving Average of model parameters for stable inference."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters()}

    def update(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model: torch.nn.Module):
        self.backup = {name: p.data.clone() for name, p in model.named_parameters()}
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir or config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    use_amp = config["training"].get("amp", True) and device == "cuda"

    print("=" * 60)
    print("UNet + ControlNet + Flow-Matching TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Output: {output_dir}")

    # ─── Load VAE (frozen) ───
    vae_path = config.get("vae", {}).get("weights", "./weights/vae_fingerprint.pt")
    vae = FingerprintVAE(
        latent_channels=config.get("vae", {}).get("latent_channels", 4),
        base_channels=config.get("vae", {}).get("base_channels", 64)
    ).to(device)

    if os.path.exists(vae_path):
        vae.load_pretrained(vae_path)
        print(f"Loaded pretrained VAE from {vae_path}")
    else:
        print(f"WARNING: VAE weights not found at {vae_path}")
        print("  Using random VAE. Run 'python scripts/train_vae.py' first for proper results.")

    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # ─── Initialize UNet+ControlNet ───
    um = config["unet_model"]
    if args.small:
        base_channels, channel_mult, num_res_blocks, attn_resolutions, num_heads, control_levels = \
            32, (1, 2), 1, (), 4, 1
        print("Using SMALL model config (32ch, 2 levels, 1 res block/level, no attention)")
    else:
        base_channels = um["base_channels"]
        channel_mult = tuple(um["channel_mult"])
        num_res_blocks = um["num_res_blocks"]
        attn_resolutions = tuple(um["attn_resolutions"])
        num_heads = um["num_heads"]
        control_levels = um["control_levels"]

    # Recorded in every checkpoint so an eval script can rebuild the exact
    # architecture instead of guessing from config (same convention as
    # train_mmdit.py's model_kwargs — breaks after a --small run otherwise).
    model_kwargs = dict(
        in_channels=um["in_channels"],
        out_channels=um["out_channels"],
        struct_channels=um["struct_channels"],
        base_channels=base_channels,
        channel_mult=channel_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        num_heads=num_heads,
        num_sensors=um["num_sensor_classes"],
        latent_size=um["latent_size"],
        control_levels=control_levels,
    )
    model = UNetControlNetDenoiser(**model_kwargs).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"UNet+ControlNet params: {total_params:,}")

    # ─── Dataset ───
    train_dataset = CrossSensorFingerprintDataset(config, split="train")
    val_dataset = CrossSensorFingerprintDataset(config, split="val")

    train_subjects = {s["subject_id"] for s in train_dataset.samples}
    val_subjects = {s["subject_id"] for s in val_dataset.samples}
    overlap = train_subjects & val_subjects
    assert not overlap, f"train/val subject leakage: {sorted(overlap)[:5]}"

    batch_size = args.batch_size or config["training"]["batch_size"]
    num_workers = config["dataset"].get("num_workers", 0)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True,
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
        persistent_workers=num_workers > 0
    )

    print(f"Dataset: {len(train_dataset)} train pairs ({len(train_subjects)} subjects), "
          f"{len(val_dataset)} val pairs ({len(val_subjects)} subjects), "
          f"{len(train_loader)} batches/epoch")

    # ─── Training components ───
    traj_manager = RectifiedFlowTrajectoryManager()
    loss_builder = Stage4CompositeLossBuilder(
        l_diff_weight=config["losses"]["l_diff_weight"],
        l_identity_weight=config["losses"]["l_identity_weight"],
        l_orient_weight=config["losses"]["l_orient_weight"],
        warmup_epochs=config["losses"]["warmup_epochs"],
        timestep_decay=config["losses"].get("timestep_decay", True),
        identity_embedder=config["losses"].get("identity_embedder", "dmd"),
        identity_checkpoint=config["losses"].get("identity_checkpoint", "./weights/dmd.pt"),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=1e-5)

    grad_accum = max(1, args.grad_accum_steps)
    warmup_steps = config["training"].get("warmup_steps", 1000)
    total_epochs = args.epochs or config["training"]["epochs"]
    total_steps = (len(train_loader) // grad_accum) * total_epochs
    min_lr_ratio = args.min_lr_ratio

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(1.0, progress)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMAModel(model, decay=config["training"].get("ema_decay", 0.9999))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ─── Resume from checkpoint ───
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    # ─── Logging ───
    log_path = os.path.join(output_dir, "train_log.csv")
    log_mode = "a" if args.resume else "w"
    log_file = open(log_path, log_mode, newline="")
    log_writer = csv.writer(log_file)
    if not args.resume:
        log_writer.writerow(["epoch", "step", "loss_total", "loss_diff",
                             "loss_identity", "loss_orient", "lr"])

    log_every = config["training"].get("log_every_steps", 50)
    save_every = config["training"].get("save_every_epochs", 5)
    val_every = config["training"].get("val_every_epochs", 1)
    grad_clip = config["training"].get("grad_clip_max_norm", 1.0)
    use_bridge = config["training"].get("use_direct_bridge", False)
    latent_size = um.get("latent_size", 64)

    # ─── Training Loop ───
    print(f"\nStarting training from epoch {start_epoch} to {total_epochs}")
    print("-" * 60)

    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_losses = {"total": 0, "diff": 0, "identity": 0, "orient": 0}
        num_steps = 0
        t_epoch = time.time()

        for step, batch in enumerate(train_loader):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            img_a = batch["img_A"].to(device)
            img_b = batch["img_B"].to(device)
            S_aligned = batch["S_aligned"].to(device)
            sensor_b = batch["sensor_B"].to(device)
            is_aligned = bool(batch["is_aligned"][0].item())

            if img_a.shape[-1] != 256:
                img_a = F.interpolate(img_a, size=(256, 256), mode="bilinear", align_corners=False)
                img_b = F.interpolate(img_b, size=(256, 256), mode="bilinear", align_corners=False)

            # Resize S_aligned to match latent spatial dimensions for the ControlNet branch
            S_for_model = F.interpolate(
                S_aligned, size=(latent_size, latent_size),
                mode="bilinear", align_corners=False
            )

            with torch.no_grad():
                z_a = vae.encode(img_a)
                z_b = vae.encode(img_b)

            source = z_a if (use_bridge and is_aligned) else None
            x_t, t, v_target, x_0 = traj_manager.sample_trajectory(
                z_b, source_latent=source, use_direct_bridge=(use_bridge and is_aligned)
            )

            with torch.amp.autocast(device_type=device, enabled=use_amp):
                v_pred = model(x_t, t, sensor_b, S_for_model)

                z1_est = traj_manager.compute_x0_estimate(x_t, v_pred, t)

                with torch.no_grad():
                    img_est = vae.decode(z1_est.float())

                loss_dict = loss_builder(
                    v_pred, v_target, img_est, img_a, S_aligned, t, epoch=epoch
                )

            scaler.scale(loss_dict["loss_total"] / grad_accum).backward()

            is_accum_boundary = (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader)
            if is_accum_boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                ema.update(model)

            epoch_losses["total"] += loss_dict["loss_total"].item()
            epoch_losses["diff"] += loss_dict["loss_diff"].item()
            epoch_losses["identity"] += loss_dict["loss_identity"].item()
            epoch_losses["orient"] += loss_dict["loss_orient"].item()
            num_steps += 1
            global_step += 1

            if global_step % log_every == 0 or step == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  E{epoch+1} S{step+1}/{len(train_loader)} (G{global_step}) | "
                      f"L={loss_dict['loss_total'].item():.4f} "
                      f"Diff={loss_dict['loss_diff'].item():.4f} "
                      f"ID={loss_dict['loss_identity'].item():.4f} "
                      f"Ori={loss_dict['loss_orient'].item():.4f} | "
                      f"LR={lr:.2e}")

            log_writer.writerow([
                epoch, global_step,
                f"{loss_dict['loss_total'].item():.6f}",
                f"{loss_dict['loss_diff'].item():.6f}",
                f"{loss_dict['loss_identity'].item():.6f}",
                f"{loss_dict['loss_orient'].item():.6f}",
                f"{optimizer.param_groups[0]['lr']:.2e}"
            ])
            log_file.flush()

        elapsed = time.time() - t_epoch
        avg = {k: v / max(num_steps, 1) for k, v in epoch_losses.items()}
        print(f"\n  Epoch {epoch+1}/{total_epochs} Summary: "
              f"L={avg['total']:.4f} Diff={avg['diff']:.4f} "
              f"ID={avg['identity']:.4f} Ori={avg['orient']:.4f} | {elapsed:.1f}s")

        if (epoch + 1) % val_every == 0:
            model.eval()
            val_loss_sum = 0
            val_steps = 0

            with torch.no_grad():
                for v_step, batch in enumerate(val_loader):
                    if args.max_steps_per_epoch and v_step >= args.max_steps_per_epoch:
                        break
                    img_a = batch["img_A"].to(device)
                    img_b = batch["img_B"].to(device)
                    S_aligned = batch["S_aligned"].to(device)
                    sensor_b = batch["sensor_B"].to(device)

                    if img_a.shape[-1] != 256:
                        img_a = F.interpolate(img_a, size=(256, 256), mode="bilinear", align_corners=False)
                        img_b = F.interpolate(img_b, size=(256, 256), mode="bilinear", align_corners=False)

                    S_for_model = F.interpolate(S_aligned, size=(latent_size, latent_size),
                                                mode="bilinear", align_corners=False)
                    z_b = vae.encode(img_b)
                    x_t, t, v_target, _ = traj_manager.sample_trajectory(z_b)
                    v_pred = model(x_t, t, sensor_b, S_for_model)
                    val_loss_sum += F.mse_loss(v_pred, v_target).item()
                    val_steps += 1

            val_loss = val_loss_sum / max(val_steps, 1)
            print(f"  Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ema.apply(model)
                save_checkpoint(model, optimizer, scheduler, ema, epoch, global_step,
                                best_val_loss, os.path.join(output_dir, "best.pt"),
                                model_kwargs, include_optimizer=False)
                ema.restore(model)
                print(f"  >> Saved BEST model (val_loss={val_loss:.4f})")

        save_checkpoint(model, optimizer, scheduler, ema, epoch, global_step,
                        best_val_loss, os.path.join(output_dir, "latest.pt"), model_kwargs)
        if (epoch + 1) % save_every == 0:
            print(f"  >> latest.pt up to date (epoch {epoch+1})")

    log_file.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints in: {output_dir}")


def save_checkpoint(model, optimizer, scheduler, ema, epoch, global_step, best_val_loss, path,
                    model_kwargs=None, include_optimizer=True):
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_kwargs": model_kwargs,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "best_val_loss": best_val_loss,
    }
    if include_optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path)


if __name__ == "__main__":
    main()
