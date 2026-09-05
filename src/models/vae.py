"""
Lightweight Convolutional VAE for grayscale fingerprint images.

Encoder: 1ch x 256x256  ->  4ch x 64x64 latent (4x downscale)
Decoder: 4ch x 64x64    ->  1ch x 256x256

4x, not the original 8x (32x32) — see `Encoder`'s docstring: RidgeLoRA-FP
(Fingen.pdf)'s own latent-resolution ablation found an 8x-equivalent bottleneck
collapses ridge/minutiae structure into "broad intensity blobs" (Ridge F1
0.994 -> 0.219), the same failure mode this VAE showed before the DMD
perceptual-loss fine-tune — and that fine-tune alone still leaves some blur/
checkerboard, motivating this second, independent lever (spatial resolution,
not the loss function) on top of it.

Architecture:
  Encoder: 4 downsample blocks (Conv stride=2 + ResBlock + GroupNorm + SiLU) down
           to 16x16 for a deep receptive field, then 2 upsample stages back out to
           64x64 for the actual latent grid. Channels: 1 -> 64 -> 128 -> 256 -> 512
           -> (up) 256 -> 256, then project to 2*latent_ch.
  Decoder: Roughly mirrors the encoder's upsample half (64x64 -> 256x256, 2 stages).
           Final Conv -> Sigmoid for [0,1] pixel output.
  Both up-sampling paths use nearest-upsample + Conv2d, not ConvTranspose2d
  (checkerboard-artifact avoidance, see `UpBlock`).

Designed specifically for fingerprint images (grayscale, structured ridges).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import os


class ResBlock(nn.Module):
    """Residual block with GroupNorm + SiLU."""

    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        groups = min(groups, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h


class DownBlock(nn.Module):
    """Downsample block: Conv(stride=2) + ResBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.res = ResBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.res(x)
        return x


class UpBlock(nn.Module):
    """
    Upsample block: nearest-neighbor upsample + Conv2d(stride=1) + ResBlock.

    Was `ConvTranspose2d(kernel=4, stride=2)`, the textbook source of checkerboard
    artifacts (Odena et al., "Deconvolution and Checkerboard Artifacts") — kernel
    size not a multiple of stride means overlapping "shells" of the transposed
    conv receive different numbers of contributions, producing a periodic pattern.
    Plain MSE training smeared it away; once perceptual/orientation loss pushed the
    decoder to keep real high-frequency ridge detail, the checkerboard came along
    with it (visible in the FFT of generated images as a regular grid). Resize +
    regular conv has no such periodicity by construction.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.res = ResBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.res(x)
        return x


class Encoder(nn.Module):
    """
    Encodes 1ch x 256x256 -> latent_ch x 64x64 (mu + logvar) — 4x spatial
    downsampling, not the earlier 8x (32x32).

    RidgeLoRA-FP (Fingen.pdf, ICLR 2026 submission)'s own latent-resolution
    ablation is the reason for the change: comparing their 128x128 (4x, from a
    512x512 input) latent against a 32x32-equivalent (8x) bottleneck proxy on the
    *same* decoder, Ridge F1 dropped 0.994 -> 0.219 and ridge/minutiae structure
    collapsed into "broad intensity blobs" — matching, almost exactly, what this
    VAE's own 8x reconstructions looked like before the DMD perceptual-loss
    fine-tune (and the mild checkerboard/blur that fine-tune still hasn't fully
    resolved). Their ablation isolates spatial resolution specifically (not
    discrete-vs-continuous latents), so the fix taken here is the same lever
    without adopting their vector-quantized codebook.

    Still 4 downsample stages for a deep receptive field / semantic bottleneck at
    16x16 (unchanged), just *two* upsample stages back out (16->32->64) instead
    of one, landing the mu/logvar grid at 64x64 instead of 32x32.
    """

    def __init__(self, in_channels: int = 1, latent_channels: int = 4,
                 base_channels: int = 64):
        super().__init__()
        ch = base_channels
        self.stem = nn.Conv2d(in_channels, ch, 3, padding=1)

        # 4 downsample blocks: 256->128->64->32->16
        self.down1 = DownBlock(ch, ch * 1)       # 64 -> 64, 256->128
        self.down2 = DownBlock(ch * 1, ch * 2)   # 64 -> 128, 128->64
        self.down3 = DownBlock(ch * 2, ch * 4)   # 128 -> 256, 64->32
        self.down4 = DownBlock(ch * 4, ch * 8)   # 256 -> 512, 32->16

        # Back up to 64x64 for latent space (2 stages: 16->32->64) — resize+conv,
        # not ConvTranspose2d (same checkerboard-artifact reasoning as `UpBlock`
        # in the decoder; doing it here matters even more, since an artifact baked
        # in at this stage would sit inside mu/logvar itself, upstream of
        # everything the decoder and MM-DiT do with it).
        self.up_to_latent1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch * 8, ch * 4, 3, stride=1, padding=1),
        )
        self.up_to_latent2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch * 4, ch * 4, 3, stride=1, padding=1),
        )
        self.res_latent = ResBlock(ch * 4)

        # Project to mu and logvar
        self.norm_out = nn.GroupNorm(32, ch * 4)
        self.act_out = nn.SiLU(inplace=True)
        self.conv_mu = nn.Conv2d(ch * 4, latent_channels, 1)
        self.conv_logvar = nn.Conv2d(ch * 4, latent_channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, logvar) each of shape (B, latent_ch, 64, 64)."""
        h = self.stem(x)       # (B, 64, 256, 256)
        h = self.down1(h)      # (B, 64, 128, 128)
        h = self.down2(h)      # (B, 128, 64, 64)
        h = self.down3(h)      # (B, 256, 32, 32)
        h = self.down4(h)      # (B, 512, 16, 16)
        h = self.up_to_latent1(h)  # (B, 256, 32, 32)
        h = self.up_to_latent2(h)  # (B, 256, 64, 64)
        h = self.res_latent(h)
        h = self.act_out(self.norm_out(h))

        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decodes latent_ch x 64x64 -> 1ch x 256x256.
    2 upsample stages: 64->128->256 spatial (was 3, 32->64->128->256 — matches
    `Encoder`'s move from 8x to 4x downsampling, see its docstring).
    """

    def __init__(self, latent_channels: int = 4, out_channels: int = 1,
                 base_channels: int = 64):
        super().__init__()
        ch = base_channels

        # Project from latent
        self.conv_in = nn.Conv2d(latent_channels, ch * 4, 3, padding=1)
        self.res_in = ResBlock(ch * 4)

        # 2 upsample blocks: 64->128->256
        self.up1 = UpBlock(ch * 4, ch * 2)   # 256 -> 128, 64->128
        self.up2 = UpBlock(ch * 2, ch * 1)   # 128 -> 64, 128->256
        # No 3rd upsample — already at 256

        self.norm_out = nn.GroupNorm(32, ch)
        self.act_out = nn.SiLU(inplace=True)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Input: (B, latent_ch, 64, 64), Output: (B, 1, 256, 256)."""
        h = self.conv_in(z)    # (B, 256, 64, 64)
        h = self.res_in(h)
        h = self.up1(h)        # (B, 128, 128, 128)
        h = self.up2(h)        # (B, 64, 256, 256)
        h = self.act_out(self.norm_out(h))
        h = self.conv_out(h)   # (B, 1, 256, 256)
        return torch.sigmoid(h)


class FingerprintVAE(nn.Module):
    """
    Variational Autoencoder for grayscale fingerprint images.

    Usage:
        vae = FingerprintVAE()
        
        # Training
        recon, mu, logvar = vae(img)
        loss = vae.loss(recon, img, mu, logvar)
        
        # Encoding (for MM-DiT training)
        z = vae.encode(img)        # -> (B, 4, 64, 64)
        
        # Decoding (for inference)
        img = vae.decode(z)        # -> (B, 1, 256, 256)
    """

    def __init__(self, in_channels: int = 1, latent_channels: int = 4,
                 base_channels: int = 64):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_channels, base_channels)
        self.decoder = Decoder(latent_channels, in_channels, base_channels)
        self.latent_channels = latent_channels

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu  # Deterministic at eval time

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent z (using reparameterization during training, mu at eval)."""
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)

    def encode_dist(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image and return (mu, logvar) distribution parameters."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent z to image."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward: encode -> reparameterize -> decode. Returns (recon, mu, logvar)."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    @staticmethod
    def loss(recon: torch.Tensor, target: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             kl_weight: float = 1e-4) -> Dict[str, torch.Tensor]:
        """
        VAE loss = Reconstruction MSE + beta * KL divergence.
        
        kl_weight (beta) is kept small to prioritize reconstruction quality.
        For fingerprint images, sharp reconstruction is critical.
        """
        recon_loss = F.mse_loss(recon, target, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + kl_weight * kl_loss

        return {
            "loss_total": total,
            "loss_recon": recon_loss,
            "loss_kl": kl_loss,
            "kl_weight": torch.tensor(kl_weight),
        }

    def load_pretrained(self, path: str) -> None:
        """Load pretrained VAE weights."""
        if not os.path.exists(path):
            print(f"[FingerprintVAE] Warning: weights not found at {path}")
            return
        state = torch.load(path, map_location="cpu")
        if "model_state_dict" in state:
            self.load_state_dict(state["model_state_dict"])
        elif "state_dict" in state:
            self.load_state_dict(state["state_dict"])
        else:
            self.load_state_dict(state)
        print(f"[FingerprintVAE] Loaded weights from {path}")

    def save_pretrained(self, path: str) -> None:
        """Save VAE weights."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"model_state_dict": self.state_dict()}, path)
        print(f"[FingerprintVAE] Saved weights to {path}")


if __name__ == "__main__":
    vae = FingerprintVAE()
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"FingerprintVAE total params: {total_params:,}")

    # Test forward
    img = torch.randn(2, 1, 256, 256).clamp(0, 1)
    recon, mu, logvar = vae(img)
    print(f"Input: {img.shape} -> Latent mu: {mu.shape} -> Recon: {recon.shape}")

    # Test encode/decode
    z = vae.encode(img)
    print(f"Encode: {img.shape} -> {z.shape}")
    decoded = vae.decode(z)
    print(f"Decode: {z.shape} -> {decoded.shape}")

    # Test loss
    losses = vae.loss(recon, img, mu, logvar)
    print(f"Loss: total={losses['loss_total'].item():.4f} "
          f"recon={losses['loss_recon'].item():.4f} "
          f"kl={losses['loss_kl'].item():.4f}")
