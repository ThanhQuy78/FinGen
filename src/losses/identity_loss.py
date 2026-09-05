import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.losses.dmd.model_zoo import DMD


class AFRNetEmbeddingExtractor(nn.Module):
    """
    AFRNet / DeepPrint fingerprint identity embedding extractor.
    Pretrained on synthetic prints (PrintsGAN) and fine-tuned on real dataset.
    Extracts a 512-dimensional normalized identity vector from fingerprint image x_0.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 512):
        super().__init__()
        # Backbone feature encoder
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embed_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Input: img tensor of shape (B, 1, H, W) normalized to [0, 1]
        Output: L2-normalized identity embedding of shape (B, embed_dim)
        """
        x = F.relu(self.bn1(self.conv1(img)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.global_pool(x).flatten(1)
        embed = self.fc(x)
        embed = F.normalize(embed, p=2, dim=1, eps=1e-8)
        return embed


class DMDEmbeddingExtractor(nn.Module):
    """
    Adapts DMD (Dense Minutia Descriptor — Pan et al., IJCB 2024, arXiv:2405.01199,
    Apache-2.0; vendored under `src/losses/dmd/`, see `dmd/NOTICE.md`) to the single
    global, L2-normalized embedding interface `IdentityCosineLoss` needs.

    DMD's own output is a *dense* per-patch descriptor (a texture branch `feat_t` and
    a minutiae branch `feat_f`, each (B, ndim_feat, h, w)) plus a foreground mask
    `mask_f` (B, 1, h, w) — built for correlation-based dense matching, not a single
    vector. We reduce it to one vector via mask-weighted global-average-pooling of
    both branches, concatenated and L2-normalized. This throws away DMD's spatial
    correspondence — it is a simplification adequate for a scalar `L_Identity`
    training signal, not a substitute for DMD's own matcher.

    DMD was trained on images normalized to [-1, 1] (`(img - 127.5) / 127.5`), so this
    wrapper rescales the [0, 1] tensors the rest of the pipeline uses before calling
    the backbone.
    """

    def __init__(
        self,
        num_in: int = 1,
        ndim_feat: int = 6,
        pos_embed: bool = True,
        input_norm: bool = False,
        tar_shape: tuple = (128, 128),
    ):
        super().__init__()
        self.model = DMD(
            num_in=num_in, ndim_feat=ndim_feat, pos_embed=pos_embed,
            input_norm=input_norm, tar_shape=tar_shape,
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Input: img tensor of shape (B, 1, H, W) normalized to [0, 1]
        Output: L2-normalized identity embedding of shape (B, 2 * ndim_feat)
        """
        x = img * 2.0 - 1.0  # [0, 1] -> [-1, 1], DMD's expected input range
        out = self.model(x)
        mask = out["mask_f"]
        denom = mask.sum(dim=(2, 3)).clamp_min(1e-6)
        pooled_f = (out["feat_f"] * mask).sum(dim=(2, 3)) / denom
        pooled_t = (out["feat_t"] * mask).sum(dim=(2, 3)) / denom
        embed = torch.cat([pooled_f, pooled_t], dim=1)
        return F.normalize(embed, p=2, dim=1, eps=1e-8)


class IdentityCosineLoss(nn.Module):
    """
    Differentiable Identity Preservation Loss L_Identity.
    Computes cosine distance between generated image x_0 and source image I_A embeddings.

    `embedder_type` selects the backbone when no `embedder` instance is passed in:
      - "dmd"    (default): DMDEmbeddingExtractor, pretrained on real fingerprints
                 (NIST SD14) — recommended, see `checkpoint_path`.
      - "afrnet": AFRNetEmbeddingExtractor — the original placeholder CNN. Its
                 pretraining recipe (PrintsGAN -> SD302a/N2N finetune) was never
                 executed in this repo, so without a `checkpoint_path` it is random
                 init and contributes no real identity signal.

    Weights are frozen during diffusion training — gradients flow through the embedder
    to the input x0_est, but do not update embedder parameters.
    """

    def __init__(
        self,
        embedder: Optional[nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        embedder_type: str = "dmd",
    ):
        super().__init__()
        if embedder is not None:
            self.embedder = embedder
        elif embedder_type == "dmd":
            self.embedder = DMDEmbeddingExtractor()
        elif embedder_type == "afrnet":
            self.embedder = AFRNetEmbeddingExtractor()
        else:
            raise ValueError(f"Unknown embedder_type: {embedder_type!r} (expected 'dmd' or 'afrnet')")

        # Load pretrained weights if provided
        if checkpoint_path is not None:
            self.load_pretrained(checkpoint_path)

        # Freeze embedder weights during diffusion training
        for param in self.embedder.parameters():
            param.requires_grad = False

    def load_pretrained(self, checkpoint_path: str) -> None:
        """
        Loads pretrained embedder weights from checkpoint file. Works for both
        DMDEmbeddingExtractor (loads into `self.embedder.model`) and
        AFRNetEmbeddingExtractor (loads into `self.embedder` directly) —
        DMD checkpoints carry a nested 'model' state_dict, so we detect which
        submodule actually matches before loading.
        """
        import os
        if not os.path.exists(checkpoint_path):
            print(f"[IdentityCosineLoss] Warning: checkpoint not found at {checkpoint_path}, using random init")
            return
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

        target = self.embedder.model if isinstance(self.embedder, DMDEmbeddingExtractor) else self.embedder
        missing, unexpected = target.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[IdentityCosineLoss] Warning: {len(missing)} missing / {len(unexpected)} "
                  f"unexpected keys loading {checkpoint_path}")
        print(f"[IdentityCosineLoss] Loaded pretrained embedder from {checkpoint_path}")

    def forward(self, x0_est: torch.Tensor, img_A: torch.Tensor) -> torch.Tensor:
        """
        Input:
          - x0_est: Estimated target image tensor (B, 1, H, W)
          - img_A: Source identity image tensor (B, 1, H, W)
        Output:
          - Cosine distance loss = 1 - cos_sim(emb_x0, emb_IA)
        """
        # Ensure single channel grayscale input
        if x0_est.shape[1] > 1:
            x0_est = x0_est.mean(dim=1, keepdim=True)
        if img_A.shape[1] > 1:
            img_A = img_A.mean(dim=1, keepdim=True)

        emb_x0 = self.embedder(x0_est)
        emb_A = self.embedder(img_A)

        cos_sim = torch.sum(emb_x0 * emb_A, dim=1)
        loss = 1.0 - cos_sim
        return loss.mean()


if __name__ == "__main__":
    loss_fn = IdentityCosineLoss()
    x0_gen = torch.randn(2, 1, 256, 256, requires_grad=True)
    img_src = torch.randn(2, 1, 256, 256)
    
    l_id = loss_fn(x0_gen, img_src)
    l_id.backward()
    print(f"Identity loss: {l_id.item():.4f}, x0 grad norm: {x0_gen.grad.norm().item():.4f}")
