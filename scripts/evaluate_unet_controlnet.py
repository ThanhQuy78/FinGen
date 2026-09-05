"""
Evaluation script for the UNet + ControlNet + Flow-Matching backbone
(src/models/unet_controlnet.py) — mirrors evaluate.py exactly except for the
model class, its config keys, and default paths. See train_unet_controlnet.py
for why this needed its own script rather than reusing evaluate.py: the
`--small` architecture and `config["model"]` keys there are MM-DiT-specific
(hidden_size/depth/num_heads), not backbone-agnostic. Everything downstream
of the model call (VAE decode, CoarseNet feature extraction, metrics) is
unchanged.

Usage:
    python scripts/evaluate_unet_controlnet.py --checkpoint outputs/training_unet_controlnet/best.pt
    python scripts/evaluate_unet_controlnet.py --checkpoint outputs/training_unet_controlnet/best.pt --num_samples 50 --save_images
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

from src.models.unet_controlnet import UNetControlNetDenoiser
from src.models.vae import FingerprintVAE
from src.models.flow_matching import RectifiedFlowTrajectoryManager
from src.preprocessing.fingernet_extractor import FingerNetExtractor, build_extractor
from src.evaluation.eval_metrics import FingerprintEvaluator
from src.data.dataset import CrossSensorFingerprintDataset
from src.losses.identity_loss import IdentityCosineLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UNet+ControlNet fingerprint generation model")
    parser.add_argument("--config", type=str, default="./configs/unet_controlnet_config.yaml")
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
    print("EVALUATION: UNet + ControlNet Cross-Sensor Fingerprint Generation")
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

    # ─── Load UNet+ControlNet ───
    um = config["unet_model"]
    if args.small:
        base_channels, channel_mult, num_res_blocks, attn_resolutions, num_heads, control_levels = \
            32, (1, 2), 1, (), 4, 1
    else:
        base_channels = um["base_channels"]
        channel_mult = tuple(um["channel_mult"])
        num_res_blocks = um["num_res_blocks"]
        attn_resolutions = tuple(um["attn_resolutions"])
        num_heads = um["num_heads"]
        control_levels = um["control_levels"]

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

    ckpt = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        # Checkpoints written by train_unet_controlnet.py carry their own
        # architecture (channel_mult can differ under --small), so a --small
        # run evaluates correctly without repeating the flag here.
        if ckpt.get("model_kwargs"):
            model_kwargs = ckpt["model_kwargs"]
            print(f"Architecture from checkpoint: base_channels={model_kwargs['base_channels']} "
                  f"channel_mult={model_kwargs['channel_mult']} "
                  f"num_res_blocks={model_kwargs['num_res_blocks']} "
                  f"control_levels={model_kwargs['control_levels']}")

    model = UNetControlNetDenoiser(**model_kwargs).to(device)

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
    print(f"UNet+ControlNet params: {total_params:,}")

    # ─── Feature Extractor (for evaluating generated images) ───
    extractor = build_extractor(config, device=device)

    # ─── Identity critic ───
    # Same DMD embedder used as L_Identity during training (loss_builder.py),
    # reused here read-only as an evaluation metric — the training loss never
    # actually got reported as a held-out number anywhere, so "does the
    # generated image keep the source's identity" had no metric of its own.
    # Frozen (IdentityCosineLoss already freezes the embedder's params), so
    # this is purely a critic here, not a training signal.
    identity_loss_fn = IdentityCosineLoss(
        embedder_type=config["losses"].get("identity_embedder", "dmd"),
        checkpoint_path=config["losses"].get("identity_checkpoint", "./weights/dmd.pt"),
    ).to(device)
    identity_loss_fn.eval()

    # ─── Evaluation Setup ───
    traj_manager = RectifiedFlowTrajectoryManager()
    evaluator = FingerprintEvaluator()

    dataset = CrossSensorFingerprintDataset(config, split="val")
    num_samples = args.num_samples if args.num_samples > 0 else len(dataset)
    num_samples = min(num_samples, len(dataset))
    print(f"Evaluating {num_samples} samples...")

    latent_size = um.get("latent_size", 64)

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

        if img_a.shape[-1] != 256:
            img_a = F.interpolate(img_a, size=(256, 256), mode="bilinear", align_corners=False)
            img_b = F.interpolate(img_b, size=(256, 256), mode="bilinear", align_corners=False)

        S_for_model = F.interpolate(S_aligned, size=(latent_size, latent_size),
                                    mode="bilinear", align_corners=False)

        # Generate via Euler sampling — sample_euler's non-MM-DiT branch
        # (model has no `.blocks`/`forward_y_stream`) calls model(x, t, c,
        # struct_map) directly, exactly UNetControlNetDenoiser.forward's
        # signature. `is_aligned` is accepted but unused on this path.
        with torch.no_grad():
            generated_lat = traj_manager.sample_euler(
                model, shape=(1, um["in_channels"], latent_size, latent_size),
                c=sensor_b, struct_map=S_for_model,
                is_aligned=True, steps=args.nfe_steps
            )
            generated_img = vae.decode(generated_lat)  # (1, 1, 256, 256)

        # Compute metrics
        orient_rmse = evaluator.compute_orientation_rmse(generated_img, img_b)

        with torch.no_grad():
            gen_feats = extractor(generated_img)
            tgt_feats = extractor(img_b)

        gen_pts_full = extractor.extract_minutiae(feats=gen_feats)[0]
        tgt_pts_full = extractor.extract_minutiae(feats=tgt_feats)[0]
        gen_pts = gen_pts_full[:, :2]
        tgt_pts = tgt_pts_full[:, :2]

        if len(gen_pts) > 0 and len(tgt_pts) > 0:
            prec, rec, f1 = evaluator.compute_minutiae_precision_recall(gen_pts, tgt_pts)
        else:
            prec, rec, f1 = 0.0, 0.0, 0.0

        mse = F.mse_loss(generated_img, img_b).item()
        psnr = -10 * np.log10(max(mse, 1e-10))

        gen_orient = gen_feats["orientation_map"]
        tgt_orient = tgt_feats["orientation_map"]
        orient_cosine = F.cosine_similarity(
            gen_orient.flatten(1), tgt_orient.flatten(1)
        ).item()

        # 5. Identity preservation via DMD embedding cosine similarity — does the
        # generated image still match the *source* fingerprint's identity?
        # Also score target-vs-source: img_a/img_b are the same finger on two
        # different sensors (directed_pairs), so this is the DMD embedder's own
        # "genuine cross-sensor pair" reference — how far gen-vs-source falls
        # short of that tells you how much identity the model is actually losing,
        # not just what the raw number is in isolation.
        with torch.no_grad():
            identity_sim_gen_src = 1.0 - identity_loss_fn(generated_img, img_a).item()
            identity_sim_tgt_src = 1.0 - identity_loss_fn(img_b, img_a).item()

        metrics = {
            "sample_id": sample_id,
            "orientation_rmse_deg": round(orient_rmse, 2),
            "minutiae_precision": round(prec, 4),
            "minutiae_recall": round(rec, 4),
            "minutiae_f1": round(f1, 4),
            "mse": round(mse, 6),
            "psnr_db": round(psnr, 2),
            "orientation_cosine_sim": round(orient_cosine, 4),
            "identity_cosine_sim_gen_vs_source": round(identity_sim_gen_src, 4),
            "identity_cosine_sim_target_vs_source": round(identity_sim_tgt_src, 4),
            "gen_minutiae_count": len(gen_pts),
            "tgt_minutiae_count": len(tgt_pts),
        }
        all_metrics.append(metrics)

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

    agg = {}
    numeric_keys = ["orientation_rmse_deg", "minutiae_precision", "minutiae_recall",
                     "minutiae_f1", "mse", "psnr_db", "orientation_cosine_sim",
                     "identity_cosine_sim_gen_vs_source", "identity_cosine_sim_target_vs_source"]
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

    print(f"  Samples evaluated: {num_samples}")
    print(f"  Time: {elapsed:.1f}s ({elapsed*1000/max(num_samples,1):.0f}ms/sample)")
    print(f"  Orientation RMSE: {agg['orientation_rmse_deg']['mean']:.2f} +/- {agg['orientation_rmse_deg']['std']:.2f} deg")
    print(f"  Minutiae F1:      {agg['minutiae_f1']['mean']:.4f} +/- {agg['minutiae_f1']['std']:.4f}")
    print(f"  PSNR:             {agg['psnr_db']['mean']:.2f} +/- {agg['psnr_db']['std']:.2f} dB")
    print(f"  Orient Cosine:    {agg['orientation_cosine_sim']['mean']:.4f} +/- {agg['orientation_cosine_sim']['std']:.4f}")
    print(f"  DMD Identity (gen vs source):    {agg['identity_cosine_sim_gen_vs_source']['mean']:.4f} +/- {agg['identity_cosine_sim_gen_vs_source']['std']:.4f}")
    print(f"  DMD Identity (target vs source, reference ceiling): {agg['identity_cosine_sim_target_vs_source']['mean']:.4f} +/- {agg['identity_cosine_sim_target_vs_source']['std']:.4f}")

    report_path = os.path.join(output_dir, "eval_results.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {report_path}")

    if args.save_images:
        print(f"Generated images saved to: {img_dir}")


if __name__ == "__main__":
    main()
