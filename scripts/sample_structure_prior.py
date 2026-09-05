"""
End-to-end new-identity sampling: draw a brand-new structural map from the
unconditional prior (structure_prior_unet.py), then render it into a full
fingerprint image with the trained UNet+ControlNet pipeline
(unet_controlnet.py) + VAE decoder. This is the composition referenced
throughout the "how can I sample a new identity" discussion — neither model
does this alone.

Also reports a DMD identity-similarity sanity check against a handful of
real training-set identities, so you can see whether the sample lands near
any specific real person (possible failure mode: the prior memorizing /
mode-collapsing onto training identities) or sits apart from all of them —
keep the earlier caveat in mind: this repo's DMD critic (pooled, not the
real dense matcher) has limited discriminative power, so treat this as a
rough signal, not proof of novelty.

Usage:
    python scripts/sample_structure_prior.py \
        --prior_checkpoint outputs/training_structure_prior/best.pt \
        --render_checkpoint outputs/training_unet_controlnet/latest.pt \
        --num_samples 4
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.structure_prior_unet import StructurePriorUNet
from src.models.unet_controlnet import UNetControlNetDenoiser
from src.models.vae import FingerprintVAE
from src.models.flow_matching import RectifiedFlowTrajectoryManager
from src.losses.identity_loss import IdentityCosineLoss
from src.data.structure_map_dataset import StructureMapDataset


def parse_args():
    p = argparse.ArgumentParser(description="Sample new fingerprint identities end-to-end")
    p.add_argument("--prior_config", type=str, default="./configs/structure_prior_config.yaml")
    p.add_argument("--render_config", type=str, default="./configs/unet_controlnet_config.yaml")
    p.add_argument("--prior_checkpoint", type=str, default="./outputs/training_structure_prior/best.pt")
    p.add_argument("--render_checkpoint", type=str, default="./outputs/training_unet_controlnet/latest.pt")
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--sensor", type=int, default=0)
    p.add_argument("--nfe_steps", type=int, default=20)
    p.add_argument("--num_reference_identities", type=int, default=10,
                    help="How many real training identities to score the sample against")
    p.add_argument("--output_dir", type=str, default="./outputs/sample_structure_prior")
    return p.parse_args()


def save_image_tensor(tensor: torch.Tensor, path: str):
    import cv2
    img = tensor.squeeze().cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(path, img)


def load_ema(model: torch.nn.Module, ckpt: dict):
    ema_sd = ckpt["ema_state_dict"]
    model_sd = model.state_dict()
    for name in model_sd:
        if name in ema_sd:
            model_sd[name] = ema_sd[name]
    model.load_state_dict(model_sd)


def main():
    args = parse_args()
    with open(args.prior_config) as f:
        prior_config = yaml.safe_load(f)
    with open(args.render_config) as f:
        render_config = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Load prior (structural map generator) ───
    prior_ckpt = torch.load(args.prior_checkpoint, map_location=device)
    prior = StructurePriorUNet(**prior_ckpt["model_kwargs"]).to(device)
    load_ema(prior, prior_ckpt)
    prior.eval()
    print(f"Loaded prior from {args.prior_checkpoint} (epoch {prior_ckpt.get('epoch')}, "
          f"val_loss {prior_ckpt.get('best_val_loss'):.4f})")

    # ─── Load renderer (UNet+ControlNet) + VAE ───
    render_ckpt = torch.load(args.render_checkpoint, map_location=device)
    renderer = UNetControlNetDenoiser(**render_ckpt["model_kwargs"]).to(device)
    load_ema(renderer, render_ckpt)
    renderer.eval()
    print(f"Loaded renderer from {args.render_checkpoint} (epoch {render_ckpt.get('epoch')})")

    vae = FingerprintVAE(
        latent_channels=render_config["vae"]["latent_channels"],
        base_channels=render_config["vae"]["base_channels"],
    ).to(device)
    vae.load_pretrained(render_config["vae"]["weights"])
    vae.eval()

    identity_loss_fn = IdentityCosineLoss(
        embedder_type=render_config["losses"].get("identity_embedder", "dmd"),
        checkpoint_path=render_config["losses"].get("identity_checkpoint", "./weights/dmd.pt"),
    ).to(device).eval()

    traj_manager = RectifiedFlowTrajectoryManager()
    map_size = prior_ckpt["model_kwargs"]["map_size"]
    latent_size = render_config["unet_model"]["latent_size"]
    in_channels = render_config["unet_model"]["in_channels"]
    sensor_c = torch.tensor([args.sensor], device=device)

    # A few real identities to score novelty against — NOT used to condition
    # generation, purely a post-hoc similarity check (see module docstring's
    # caveat on the pooled DMD critic's limited discriminative power).
    ref_dataset = StructureMapDataset(
        cached_prep_dir=prior_config["dataset"]["cached_prep_dir"], map_size=map_size, split="train"
    )
    ref_indices = torch.randperm(len(ref_dataset))[:args.num_reference_identities].tolist()

    for i in range(args.num_samples):
        with torch.no_grad():
            # 1. Sample a brand-new structural map from pure noise (unconditional).
            dummy_struct = torch.zeros(1, 6, map_size, map_size, device=device)
            sampled_struct = traj_manager.sample_euler(
                prior, shape=(1, 6, map_size, map_size),
                c=sensor_c, struct_map=dummy_struct, steps=args.nfe_steps,
            )

            # 2. Render it into a full fingerprint image via the trained
            # UNet+ControlNet pipeline, exactly like a real cached S_aligned would be.
            gen_lat = traj_manager.sample_euler(
                renderer, shape=(1, in_channels, latent_size, latent_size),
                c=sensor_c, struct_map=sampled_struct, steps=args.nfe_steps,
            )
            gen_img = vae.decode(gen_lat)

            # 3. Sanity check: how similar is this to a handful of real identities?
            sims = []
            for ref_idx in ref_indices:
                # Need the *image*, not just the structural map, for the DMD critic —
                # reuse the render pipeline on the real cached map as a stand-in for
                # "what a real identity looks like rendered the same way", so the
                # comparison isn't confounded by the renderer's own texture quality.
                ref_struct = ref_dataset[ref_idx].unsqueeze(0).to(device)
                ref_lat = traj_manager.sample_euler(
                    renderer, shape=(1, in_channels, latent_size, latent_size),
                    c=sensor_c, struct_map=ref_struct, steps=args.nfe_steps,
                )
                ref_img = vae.decode(ref_lat)
                sim = 1.0 - identity_loss_fn(gen_img, ref_img).item()
                sims.append(sim)

        save_image_tensor(gen_img, os.path.join(args.output_dir, f"new_identity_{i:02d}.png"))
        max_sim = max(sims)
        print(f"[{i+1}/{args.num_samples}] saved new_identity_{i:02d}.png | "
              f"DMD sim vs {len(sims)} real identities: mean={np.mean(sims):.4f} "
              f"max={max_sim:.4f} (closest real match)")

    print(f"\nImages saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
