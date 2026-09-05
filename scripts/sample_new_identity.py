"""
Experimental: sample a "new" (not-in-dataset) fingerprint by blending two real
subjects' structural maps and feeding the hybrid through the trained
UNet+ControlNet pipeline.

This is NOT a validated generative-identity mode — see the project discussion
that led here: the model was only ever trained with real, cached S_aligned
maps (no classifier-free-guidance dropout, no unconditional path), so this is
out-of-distribution conditioning for the ControlNet branch. It's a cheap way
to *probe* whether the pipeline can produce a plausible hybrid without any
retraining, not a substitute for a real unconditional prior over structural
maps (the architecturally correct way to generate de-novo identities — see
PrintsGAN-style approaches referenced in identity_loss.py).

Blend recipe, informed by what each of the 6 channels actually is
(fingernet_extractor.py's `combined_structure`):
  ch 0:   segmentation mask           -> linear blend (still a soft mask)
  ch 1-2: orientation (cos2θ, sin2θ)  -> linear blend + L2 renormalize (unit vector)
  ch 3:   minutiae score              -> linear blend
  ch 4-5: minutiae direction (cos,sin)-> linear blend + L2 renormalize (unit vector)

Usage:
    python scripts/sample_new_identity.py --checkpoint outputs/training_unet_controlnet/latest.pt --alpha 0.5
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.unet_controlnet import UNetControlNetDenoiser
from src.models.vae import FingerprintVAE
from src.models.flow_matching import RectifiedFlowTrajectoryManager
from src.losses.identity_loss import IdentityCosineLoss
from src.data.dataset import CrossSensorFingerprintDataset


def parse_args():
    p = argparse.ArgumentParser(description="Blend two identities' structural maps and sample a hybrid fingerprint")
    p.add_argument("--config", type=str, default="./configs/unet_controlnet_config.yaml")
    p.add_argument("--checkpoint", type=str, default="./outputs/training_unet_controlnet/latest.pt")
    p.add_argument("--alpha", type=float, default=0.5, help="Blend weight toward subject B (0=pure A, 1=pure B)")
    p.add_argument("--sensor", type=int, default=0, help="Target sensor class id to render as")
    p.add_argument("--nfe_steps", type=int, default=20)
    p.add_argument("--output_dir", type=str, default="./outputs/sample_new_identity")
    return p.parse_args()


def save_image_tensor(tensor: torch.Tensor, path: str):
    import cv2
    img = tensor.squeeze().cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(path, img)


def blend_structure_maps(s_a: torch.Tensor, s_b: torch.Tensor, alpha: float) -> torch.Tensor:
    """s_a, s_b: (1, 6, H, W). See module docstring for the channel layout."""
    seg = (1 - alpha) * s_a[:, 0:1] + alpha * s_b[:, 0:1]

    orient = (1 - alpha) * s_a[:, 1:3] + alpha * s_b[:, 1:3]
    orient = F.normalize(orient, p=2, dim=1, eps=1e-8)

    mnt_score = (1 - alpha) * s_a[:, 3:4] + alpha * s_b[:, 3:4]

    mnt_dir = (1 - alpha) * s_a[:, 4:6] + alpha * s_b[:, 4:6]
    mnt_dir = F.normalize(mnt_dir, p=2, dim=1, eps=1e-8)

    return torch.cat([seg, orient, mnt_score, mnt_dir], dim=1)


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    vae = FingerprintVAE(
        latent_channels=config["vae"]["latent_channels"], base_channels=config["vae"]["base_channels"]
    ).to(device)
    vae.load_pretrained(config["vae"]["weights"])
    vae.eval()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = UNetControlNetDenoiser(**ckpt["model_kwargs"]).to(device)
    ema_sd = ckpt["ema_state_dict"]
    model_sd = model.state_dict()
    for name in model_sd:
        if name in ema_sd:
            model_sd[name] = ema_sd[name]
    model.load_state_dict(model_sd)
    model.eval()
    print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch')})")

    identity_loss_fn = IdentityCosineLoss(
        embedder_type=config["losses"].get("identity_embedder", "dmd"),
        checkpoint_path=config["losses"].get("identity_checkpoint", "./weights/dmd.pt"),
    ).to(device).eval()

    dataset = CrossSensorFingerprintDataset(config, split="val")

    # Pick two samples from two *different* subjects so the blend isn't trivially
    # "the same identity twice".
    subj_to_idx = {}
    for i, s in enumerate(dataset.samples):
        subj_to_idx.setdefault(s["subject_id"], i)
        if len(subj_to_idx) >= 2:
            break
    subjects = list(subj_to_idx.keys())
    idx_a, idx_b = subj_to_idx[subjects[0]], subj_to_idx[subjects[1]]
    print(f"Subject A: {subjects[0]} (sample idx {idx_a}) | Subject B: {subjects[1]} (sample idx {idx_b})")

    sample_a = dataset[idx_a]
    sample_b = dataset[idx_b]
    img_a = sample_a["img_A"].unsqueeze(0).to(device)
    img_b = sample_b["img_A"].unsqueeze(0).to(device)
    S_a = sample_a["S_aligned"].unsqueeze(0).to(device)
    S_b = sample_b["S_aligned"].unsqueeze(0).to(device)

    latent_size = config["unet_model"]["latent_size"]
    if img_a.shape[-1] != 256:
        img_a = F.interpolate(img_a, size=(256, 256), mode="bilinear", align_corners=False)
        img_b = F.interpolate(img_b, size=(256, 256), mode="bilinear", align_corners=False)
    S_a_lat = F.interpolate(S_a, size=(latent_size, latent_size), mode="bilinear", align_corners=False)
    S_b_lat = F.interpolate(S_b, size=(latent_size, latent_size), mode="bilinear", align_corners=False)

    S_blend = blend_structure_maps(S_a_lat, S_b_lat, args.alpha)

    traj_manager = RectifiedFlowTrajectoryManager()
    sensor_c = torch.tensor([args.sensor], device=device)

    with torch.no_grad():
        gen_lat = traj_manager.sample_euler(
            model, shape=(1, config["unet_model"]["in_channels"], latent_size, latent_size),
            c=sensor_c, struct_map=S_blend, steps=args.nfe_steps
        )
        gen_img = vae.decode(gen_lat)

        sim_to_a = 1.0 - identity_loss_fn(gen_img, img_a).item()
        sim_to_b = 1.0 - identity_loss_fn(gen_img, img_b).item()
        sim_a_to_b = 1.0 - identity_loss_fn(img_a, img_b).item()  # reference: how far apart A/B already are

    save_image_tensor(gen_img, os.path.join(args.output_dir, "blend_gen.png"))
    save_image_tensor(img_a, os.path.join(args.output_dir, "subject_A.png"))
    save_image_tensor(img_b, os.path.join(args.output_dir, "subject_B.png"))

    print("\n" + "=" * 60)
    print(f"alpha={args.alpha} (0=pure A, 1=pure B)")
    print(f"DMD identity cosine sim, blend-gen vs subject A: {sim_to_a:.4f}")
    print(f"DMD identity cosine sim, blend-gen vs subject B: {sim_to_b:.4f}")
    print(f"DMD identity cosine sim, subject A vs subject B (different people, reference floor): {sim_a_to_b:.4f}")
    print("=" * 60)
    print(f"Images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
