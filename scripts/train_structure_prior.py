"""
Training script for the unconditional structural-map prior
(src/models/structure_prior_unet.py) — learns p(S_aligned) over real
fingerprints via flow matching, so a brand-new structural map can later be
sampled from pure noise (scripts/sample_structure_prior.py) and rendered
into a full image by the existing UNet+ControlNet pipeline.

Much simpler than train_unet_controlnet.py: no VAE (operates directly on the
6-channel structural map, not a latent), no sensor conditioning, no
identity/orientation losses (there's no "source image" to preserve identity
against — this *is* the identity-generation step), just flow-matching MSE.
Reuses RectifiedFlowTrajectoryManager.sample_trajectory (flow_matching.py)
since it only depends on the target tensor, not on what it represents.

Usage:
    python scripts/train_structure_prior.py --config configs/structure_prior_config.yaml
    python scripts/train_structure_prior.py --config configs/structure_prior_config.yaml --small
"""

import sys
import os
import argparse
import time
import csv
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.structure_prior_unet import StructurePriorUNet
from src.models.flow_matching import RectifiedFlowTrajectoryManager
from src.data.structure_map_dataset import StructureMapDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train the unconditional structural-map prior")
    parser.add_argument("--config", type=str, default="./configs/structure_prior_config.yaml")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--small", action="store_true",
                        help="Use lightweight model (24ch/2 levels/1 res block) for CPU/small GPU")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max_steps_per_epoch", type=int, default=0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=0)
    return parser.parse_args()


class EMAModel:
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
    print("STRUCTURAL-MAP PRIOR TRAINING (unconditional flow matching)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Output: {output_dir}")

    pm = config["prior_model"]
    if args.small:
        # 32 (not 24) so every level's channel count stays divisible by
        # GroupNorm's group count (min(32, channels)) — same small config
        # unet_controlnet.py uses, for the same reason.
        base_channels, channel_mult, num_res_blocks, attn_resolutions, num_heads = \
            32, (1, 2), 1, (), 4
        print("Using SMALL model config (32ch, 2 levels, 1 res block/level, no attention)")
    else:
        base_channels = pm["base_channels"]
        channel_mult = tuple(pm["channel_mult"])
        num_res_blocks = pm["num_res_blocks"]
        attn_resolutions = tuple(pm["attn_resolutions"])
        num_heads = pm["num_heads"]

    model_kwargs = dict(
        struct_channels=pm["struct_channels"],
        base_channels=base_channels,
        channel_mult=channel_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        num_heads=num_heads,
        map_size=pm["map_size"],
    )
    model = StructurePriorUNet(**model_kwargs).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"StructurePriorUNet params: {total_params:,}")

    ds_cfg = config["dataset"]
    train_dataset = StructureMapDataset(
        cached_prep_dir=ds_cfg["cached_prep_dir"], map_size=pm["map_size"],
        split="train", val_fraction=ds_cfg["val_fraction"], split_seed=ds_cfg["split_seed"],
    )
    val_dataset = StructureMapDataset(
        cached_prep_dir=ds_cfg["cached_prep_dir"], map_size=pm["map_size"],
        split="val", val_fraction=ds_cfg["val_fraction"], split_seed=ds_cfg["split_seed"],
    )
    overlap = train_dataset.subjects & val_dataset.subjects
    assert not overlap, f"train/val subject leakage: {sorted(overlap)[:5]}"

    batch_size = args.batch_size or config["training"]["batch_size"]
    num_workers = ds_cfg.get("num_workers", 0)
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
    print(f"Dataset: {len(train_dataset)} train maps, {len(val_dataset)} val maps, "
          f"{len(train_loader)} batches/epoch")

    traj_manager = RectifiedFlowTrajectoryManager()
    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=1e-5)

    warmup_steps = config["training"].get("warmup_steps", 500)
    total_epochs = args.epochs or config["training"]["epochs"]
    total_steps = len(train_loader) * total_epochs
    min_lr_ratio = args.min_lr_ratio

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = min(1.0, (step - warmup_steps) / max(total_steps - warmup_steps, 1))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMAModel(model, decay=config["training"].get("ema_decay", 0.9999))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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

    log_path = os.path.join(output_dir, "train_log.csv")
    log_mode = "a" if args.resume else "w"
    log_file = open(log_path, log_mode, newline="")
    log_writer = csv.writer(log_file)
    if not args.resume:
        log_writer.writerow(["epoch", "step", "loss_diff", "lr"])

    log_every = config["training"].get("log_every_steps", 50)
    save_every = config["training"].get("save_every_epochs", 5)
    val_every = config["training"].get("val_every_epochs", 1)
    grad_clip = config["training"].get("grad_clip_max_norm", 1.0)

    print(f"\nStarting training from epoch {start_epoch} to {total_epochs}")
    print("-" * 60)

    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_loss_sum, num_steps = 0.0, 0
        t_epoch = time.time()

        for step, s_map in enumerate(train_loader):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            s_map = s_map.to(device)

            x_t, t, v_target, _ = traj_manager.sample_trajectory(s_map)

            with torch.amp.autocast(device_type=device, enabled=use_amp):
                v_pred = model(x_t, t)
                loss = torch.nn.functional.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            epoch_loss_sum += loss.item()
            num_steps += 1
            global_step += 1

            if global_step % log_every == 0 or step == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  E{epoch+1} S{step+1}/{len(train_loader)} (G{global_step}) | "
                      f"Loss={loss.item():.4f} | LR={lr:.2e}")

            log_writer.writerow([epoch, global_step, f"{loss.item():.6f}",
                                 f"{optimizer.param_groups[0]['lr']:.2e}"])
            log_file.flush()

        elapsed = time.time() - t_epoch
        avg_loss = epoch_loss_sum / max(num_steps, 1)
        print(f"\n  Epoch {epoch+1}/{total_epochs} Summary: Loss={avg_loss:.4f} | {elapsed:.1f}s")

        if (epoch + 1) % val_every == 0:
            model.eval()
            val_loss_sum, val_steps = 0.0, 0
            with torch.no_grad():
                for v_step, s_map in enumerate(val_loader):
                    if args.max_steps_per_epoch and v_step >= args.max_steps_per_epoch:
                        break
                    s_map = s_map.to(device)
                    x_t, t, v_target, _ = traj_manager.sample_trajectory(s_map)
                    v_pred = model(x_t, t)
                    val_loss_sum += torch.nn.functional.mse_loss(v_pred, v_target).item()
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
