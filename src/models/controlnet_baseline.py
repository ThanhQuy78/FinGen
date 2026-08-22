import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from src.models.rope2d import RotaryEmbedding2D


class TimestepSensorEmbedder(nn.Module):
    """
    Embedder for timestep t and domain sensor condition c using Sinusoidal + MLP.
    """
    def __init__(self, hidden_size: int, num_sensors: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.sensor_embedder = nn.Embedding(num_sensors, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Timestep sinusoidal embedding
        half_dim = self.hidden_size // 2
        emb = torch.exp(-torch.arange(half_dim, device=t.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = t.unsqueeze(-1).float() * emb.unsqueeze(0)
        t_emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Combine timestep embedding with sensor embedding
        c_emb = self.sensor_embedder(c)
        cond = self.mlp(t_emb + c_emb)
        return cond


class DiTBlock(nn.Module):
    """
    Standard DiT Block with Adaptive Layer Norm (AdaLN-Zero).
    
    NOTE (Baseline simplification): This block uses nn.MultiheadAttention which does not
    support 2D-RoPE injection natively. RoPE is intentionally omitted here — the baseline
    exists for fast early comparison numbers. The full MM-DiT model (mm_dit.py) applies
    2D-RoPE at Q/K per layer via custom attention.
    """
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # Self-Attention Branch
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP Branch
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm2)
        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ControlNetTransformerBaseline(nn.Module):
    """
    Baseline architecture: Single-stream DiT + ControlNet-Transformer.
    Replicates the first N blocks of the DiT backbone with zero-initialized linear projections
    for fast early benchmarking numbers.
    """

    def __init__(
        self,
        in_channels: int = 4,
        struct_channels: int = 6,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        patch_size: int = 2,
        control_blocks: int = 4,
        num_sensors: int = 8
    ):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.control_blocks = control_blocks

        # Patch Embeddings
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.struct_embedder = nn.Conv2d(struct_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        # Condition & Position Embeddings
        self.t_c_embedder = TimestepSensorEmbedder(hidden_size, num_sensors=num_sensors)
        self.rope2d = RotaryEmbedding2D(dim=hidden_size // num_heads)

        # Backbone DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        # ControlNet Block Replicas (First control_blocks layers)
        self.control_blocks_list = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(control_blocks)
        ])
        
        # Zero-initialized linear projections for ControlNet features
        self.zero_convs = nn.ModuleList([
            self._zero_init(nn.Linear(hidden_size, hidden_size)) for _ in range(control_blocks)
        ])

        # Final Head
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.final_linear = nn.Linear(hidden_size, patch_size * patch_size * in_channels)

    def _zero_init(self, module: nn.Module) -> nn.Module:
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return module

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        struct_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Input:
          - x: Latent target image (B, 4, H_lat, W_lat)
          - t: Denoising timestep (B,)
          - c: Target sensor class ID (B,)
          - struct_map: Structural conditioning map (B, 6, H_img, W_img)
        Output:
          - Velocity / Noise prediction (B, 4, H_lat, W_lat)
        """
        B, C, H, W = x.shape

        # Patchify inputs
        x_tok = self.x_embedder(x).flatten(2).transpose(1, 2)            # (B, N, hidden_size)
        s_tok = self.struct_embedder(struct_map).flatten(2).transpose(1, 2) # (B, N_s, hidden_size)

        # Match token sequence length if struct_map scale differs
        if s_tok.shape[1] != x_tok.shape[1]:
            s_tok = F.interpolate(s_tok.transpose(1, 2), size=x_tok.shape[1], mode="linear", align_corners=False).transpose(1, 2)

        cond = self.t_c_embedder(t, c)

        # Forward ControlNet branch
        ctrl_x = x_tok + s_tok
        control_outputs = []
        for i in range(self.control_blocks):
            ctrl_x = self.control_blocks_list[i](ctrl_x, cond)
            control_outputs.append(self.zero_convs[i](ctrl_x))

        # Forward Main DiT Backbone
        out_x = x_tok
        for i, block in enumerate(self.blocks):
            out_x = block(out_x, cond)
            if i < self.control_blocks:
                out_x = out_x + control_outputs[i]

        # Unpatchify to image latent dimensions
        out_x = self.final_norm(out_x)
        logits = self.final_linear(out_x)                                # (B, N, patch_size^2 * in_channels)
        
        # Unpatchify tensor (B, N, P*P*C) -> (B, C, H, W)
        P = self.patch_size
        H_tok, W_tok = H // P, W // P
        logits = logits.view(B, H_tok, W_tok, P, P, -1).permute(0, 5, 1, 3, 2, 4).reshape(B, -1, H, W)
        return logits


if __name__ == "__main__":
    model = ControlNetTransformerBaseline()
    x_in = torch.randn(2, 4, 32, 32)
    t_in = torch.tensor([10, 500])
    c_in = torch.tensor([0, 3])
    struct_in = torch.randn(2, 6, 64, 64)
    pred = model(x_in, t_in, c_in, struct_in)
    print("ControlNet Baseline output shape:", pred.shape)
