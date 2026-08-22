import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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


class IdentityCosineLoss(nn.Module):
    """
    Differentiable Identity Preservation Loss L_Identity.
    Computes cosine distance between generated image x_0 and source image I_A embeddings.
    
    The embedder should be pretrained on PrintsGAN and fine-tuned on real data (SD302a/N2N).
    Weights are frozen during diffusion training — gradients flow through the embedder
    to the input x0_est, but do not update embedder parameters.
    """

    def __init__(self, embedder: Optional[nn.Module] = None, checkpoint_path: Optional[str] = None):
        super().__init__()
        self.embedder = embedder or AFRNetEmbeddingExtractor()
        
        # Load pretrained weights if provided
        if checkpoint_path is not None:
            self.load_pretrained(checkpoint_path)
        
        # Freeze embedder weights during diffusion training
        for param in self.embedder.parameters():
            param.requires_grad = False

    def load_pretrained(self, checkpoint_path: str) -> None:
        """
        Loads pretrained AFRNet/DeepPrint weights from checkpoint file.
        Expected recipe: pretrain on PrintsGAN (525k synthetic) → fine-tune on SD302a/N2N (~25k real).
        """
        import os
        if not os.path.exists(checkpoint_path):
            print(f"[IdentityCosineLoss] Warning: checkpoint not found at {checkpoint_path}, using random init")
            return
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # Handle wrapped state dicts (e.g., from DDP or lightning)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.embedder.load_state_dict(state_dict, strict=False)
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
