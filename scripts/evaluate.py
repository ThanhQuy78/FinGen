"""
Production Evaluation Script for Cross-Sensor Fingerprint Transfer.

Loads trained MM-DiT + VAE, generates images for the full val/test set,
extracts features, and computes real metrics.

Usage:
    python scripts/evaluate.py --checkpoint outputs/training/best.pt
    python scripts/evaluate.py --checkpoint outputs/training/best.pt --num_samples 50 --save_images
"""

import sys
import os
import argparse
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.mm_dit import DualStreamMMDiT
from src.models.vae import FingerprintVAE
from src.models.flow_matching import RectifiedFlowTrajectoryManager
from src.preprocessing.fingernet_extractor import FingerNetExtractor, build_extractor
from src.evaluation.eval_metrics import FingerprintEvaluator
from src.data.dataset import CrossSensorFingerprintDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fingerprint generation model")
    parser.add_argument("--config", type=str, default="./configs/default_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Number of samples to evaluate (-1 = all)")
    parser.add_argument("--nfe_steps", type=int, default=20)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--small", action="store_true",
                        help="Use small model config (must match training)")
    return parser.parse_args()


def save_image_tensor(tensor: torch.Tensor, path: str):
    """Save single-channel tensor [0,1] as grayscale PNG."""
    import cv2
    img = tensor.squeeze().cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(path, img)


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = args.checkpoint or config["eval"].get("checkpoint", "")
    output_dir = args.output_dir or config["eval"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if args.save_images:
        img_dir = os.path.join(output_dir, "generated_images")
        os.makedirs(img_dir, exist_ok=True)

    print("=" * 60)
    print("EVALUATION: Cross-Sensor Fingerprint Generation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"NFE steps: {args.nfe_steps}")

    # ─── Load VAE ───
    vae = FingerprintVAE(
        latent_channels=config.get("vae", {}).get("latent_channels", 4),
        base_channels=config.get("vae", {}).get("base_channels", 64)
    ).to(device)
    vae_path = config.get("vae", {}).get("weights", "./weights/vae_fingerprint.pt")
    if os.path.exists(vae_path):
        vae.load_pretrained(vae_path)
    else:
        print(f"WARNING: VAE weights not found at {vae_path}, using random VAE")
    vae.eval()

    # ─── Load MM-DiT ───
    if args.small:
        hidden_size, depth, num_heads = 256, 4, 4
    else:
        hidden_size = config["model"]["hidden_size"]
        depth = config["model"]["depth"]
        num_heads = config["model"]["num_heads"]

    model_kwargs = dict(
        in_channels=config["model"]["in_channels"],
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        patch_size=config["model"]["patch_size"],
        num_sensors=config["model"]["num_sensor_classes"],
        rope_offset_delta=config["model"]["rope_offset_delta"]
    )

    ckpt = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        # Checkpoints written by train_mmdit.py carry their own architecture, so a
        # --small run evaluates correctly without repeating the flag here.
        if ckpt.get("model_kwargs"):
            model_kwargs = ckpt["model_kwargs"]
            print(f"Architecture from checkpoint: hidden={model_kwargs['hidden_size']} "
                  f"depth={model_kwargs['depth']} heads={model_kwargs['num_heads']}")

    model = DualStreamMMDiT(**model_kwargs).to(device)

    if ckpt is not None:
        # Try EMA weights first (better quality)
        if "ema_state_dict" in ckpt:
            ema_sd = ckpt["ema_state_dict"]
            model_sd = model.state_dict()
            for name in model_sd:
                if name in ema_sd:
                    model_sd[name] = ema_sd[name]
            model.load_state_dict(model_sd)
            print("Loaded EMA weights from checkpoint")
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            print("Loaded model weights from checkpoint")
        epoch = ckpt.get("epoch", "?")
        print(f"Checkpoint epoch: {epoch}")
    else:
        print("WARNING: No checkpoint loaded, using random weights")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MM-DiT params: {total_params:,}")

    # ─── Feature Extractor (for evaluating generated images) ───
    # Load the pretrained CoarseNet — evaluating orientation/minutiae metrics against
    # a randomly-initialised extractor would produce meaningless numbers.
    extractor = build_extractor(config, device=device)

    # ─── Evaluation Setup ───
    traj_manager = RectifiedFlowTrajectoryManager()
    evaluator = FingerprintEvaluator()

    dataset = CrossSensorFingerprintDataset(config, split="val")
    num_samples = args.num_samples if args.num_samples > 0 else len(dataset)
    num_samples = min(num_samples, len(dataset))
    print(f"Evaluating {num_samples} samples...")

    latent_size = config["model"].get("latent_size", 32)

    # ─── Generate & Evaluate ───
    all_metrics = []
    t_start = time.time()

    for i in range(num_samples):
        sample = dataset[i]
        img_a = sample["img_A"].unsqueeze(0).to(device)
        img_b = sample["img_B"].unsqueeze(0).to(device)
        S_aligned = sample["S_aligned"].unsqueeze(0).to(device)
        sensor_b = sample["sensor_B"].unsqueeze(0).to(device)
        sample_id = sample.get("sample_id", f"sample_{i}")

        # Ensure correct size
        if img_a.shape[-1] != 256:
            img_a = F.interpolate(img_a, size=(256, 256), mode="bilinear", align_corners=False)
            img_b = F.interpolate(img_b, size=(256, 256), mode="bilinear", align_corners=False)

        S_for_model = F.interpolate(S_aligned, size=(latent_size, latent_size),
                                    mode="bilinear", align_corners=False)

        # Generate via Euler sampling
        with torch.no_grad():
            generated_lat = traj_manager.sample_euler(
                model, shape=(1, config["model"]["in_channels"], latent_size, latent_size),
                c=sensor_b, struct_map=S_for_model,
                is_aligned=True, steps=args.nfe_steps
            )
            # Decode latent to image
            generated_img = vae.decode(generated_lat)  # (1, 1, 256, 256)

        # Compute metrics
        # 1. Orientation RMSE
        orient_rmse = evaluator.compute_orientation_rmse(generated_img, img_b)

        # 2. Extract minutiae from generated image
        with torch.no_grad():
            gen_feats = extractor(generated_img)
            tgt_feats = extractor(img_b)

        # Extract minutiae points from heatmaps (threshold > 0.5)
        gen_mnt = gen_feats["minutiae_map"][0, 0]  # presence channel
        tgt_mnt = tgt_feats["minutiae_map"][0, 0]

        gen_pts = torch.nonzero(gen_mnt > 0.5).cpu().numpy()
        tgt_pts = torch.nonzero(tgt_mnt > 0.5).cpu().numpy()

        if len(gen_pts) > 0 and len(tgt_pts) > 0:
            prec, rec, f1 = evaluator.compute_minutiae_precision_recall(gen_pts, tgt_pts)
        else:
            prec, rec, f1 = 0.0, 0.0, 0.0

        # 3. Reconstruction quality (MSE, SSIM-like)
        mse = F.mse_loss(generated_img, img_b).item()
        psnr = -10 * np.log10(max(mse, 1e-10))

        # 4. Identity preservation via embedding cosine similarity
        gen_orient = gen_feats["orientation_map"]
        tgt_orient = tgt_feats["orientation_map"]
        orient_cosine = F.cosine_similarity(
            gen_orient.flatten(1), tgt_orient.flatten(1)
        ).item()

        metrics = {
            "sample_id": sample_id,
            "orientation_rmse_deg": round(orient_rmse, 2),
            "minutiae_precision": round(prec, 4),
            "minutiae_recall": round(rec, 4),
            "minutiae_f1": round(f1, 4),
            "mse": round(mse, 6),
            "psnr_db": round(psnr, 2),
            "orientation_cosine_sim": round(orient_cosine, 4),
            "gen_minutiae_count": len(gen_pts),
            "tgt_minutiae_count": len(tgt_pts),
        }
        all_metrics.append(metrics)

        # Save images
        if args.save_images:
            save_image_tensor(generated_img, os.path.join(img_dir, f"{sample_id}_gen.png"))
            save_image_tensor(img_b, os.path.join(img_dir, f"{sample_id}_target.png"))
            save_image_tensor(img_a, os.path.join(img_dir, f"{sample_id}_source.png"))

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{num_samples}] {sample_id}: "
                  f"RMSE={orient_rmse:.1f} Mnt_F1={f1:.2f} PSNR={psnr:.1f}dB")

    elapsed = time.time() - t_start

    # ─── Aggregate Report ───
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    # Compute aggregated stats
    agg = {}
    numeric_keys = ["orientation_rmse_deg", "minutiae_precision", "minutiae_recall",
                     "minutiae_f1", "mse", "psnr_db", "orientation_cosine_sim"]
    for key in numeric_keys:
        values = [m[key] for m in all_metrics]
        agg[key] = {
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values)), 4),
            "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4),
        }

    report = {
        "num_samples": num_samples,
        "nfe_steps": args.nfe_steps,
        "checkpoint": checkpoint_path,
        "time_seconds": round(elapsed, 1),
        "time_per_sample_ms": round(elapsed * 1000 / max(num_samples, 1), 1),
        "aggregated_metrics": agg,
        "per_sample_metrics": all_metrics,
    }

    # Print summary
    print(f"  Samples evaluated: {num_samples}")
    print(f"  Time: {elapsed:.1f}s ({elapsed*1000/max(num_samples,1):.0f}ms/sample)")
    print(f"  Orientation RMSE: {agg['orientation_rmse_deg']['mean']:.2f} +/- {agg['orientation_rmse_deg']['std']:.2f} deg")
    print(f"  Minutiae F1:      {agg['minutiae_f1']['mean']:.4f} +/- {agg['minutiae_f1']['std']:.4f}")
    print(f"  PSNR:             {agg['psnr_db']['mean']:.2f} +/- {agg['psnr_db']['std']:.2f} dB")
    print(f"  Orient Cosine:    {agg['orientation_cosine_sim']['mean']:.4f} +/- {agg['orientation_cosine_sim']['std']:.4f}")

    # Save report
    report_path = os.path.join(output_dir, "eval_results.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {report_path}")

    if args.save_images:
        print(f"Generated images saved to: {img_dir}")


if __name__ == "__main__":
    main()
