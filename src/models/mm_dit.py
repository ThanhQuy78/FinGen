import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from src.models.rope2d import RotaryEmbedding2D


class TimestepSensorEmbedder(nn.Module):
    """
    Timestep and Sensor domain embedder for Stream X AdaLN conditioning.
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
        half_dim = self.hidden_size // 2
        emb = torch.exp(-torch.arange(half_dim, device=t.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = t.unsqueeze(-1).float() * emb.unsqueeze(0)
        t_emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        c_emb = self.sensor_embedder(c)
        return self.mlp(t_emb + c_emb)


class MMDiTBlock(nn.Module):
    """
    Dual-stream MM-DiT Block with 1-way Cross-Attention (Stream X -> Stream Y).
    - Stream X: Image stream, modulated by AdaLN (t, c).
    - Stream Y: Identity structural stream, fixed LayerNorm (NO t, c) -> Cacheable!
    - Attention:
        - Stream Y performs Self-Attention (Y self-attends to keep ridge topology intact).
        - Stream X performs Self-Attention AND 1-way Cross-Attention to Stream Y (X takes Q, queries K/V of Y).
    - 2D-RoPE applied to Q & K per layer.
    """

    def __init__(self, hidden_size: int, num_heads: int, rope2d: RotaryEmbedding2D):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rope2d = rope2d

        # Stream X Norms & MLPs
        self.norm1_x = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv_x = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.out_x = nn.Linear(hidden_size, hidden_size, bias=True)
        self.norm2_x = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_x = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        # 9 modulation params: 3 self-attn (shift, scale, gate) + 3 cross-attn + 3 MLP
        self.adaLN_x = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

        # Stream Y Norms & MLPs (No AdaLN - invariant to timestep/sensor!)
        self.norm1_y = nn.LayerNorm(hidden_size, eps=1e-6)
        self.qkv_y = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.out_y = nn.Linear(hidden_size, hidden_size, bias=True)
        self.norm2_y = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp_y = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        # 1-Way Cross-Attention (X takes Q, Y provides K/V)
        self.norm_cross_x = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cross_y = nn.LayerNorm(hidden_size, eps=1e-6)
        self.q_cross_x = nn.Linear(hidden_size, hidden_size, bias=True)
        self.kv_cross_y = nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        self.out_cross_x = nn.Linear(hidden_size, hidden_size, bias=True)

    def _apply_rope_qk(self, q: torch.Tensor, k: torch.Tensor, H: int, W: int, offset_delta: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        q_rope = self.rope2d.apply_rope(q, H, W, offset_delta=0)
        k_rope = self.rope2d.apply_rope(k, H, W, offset_delta=offset_delta)
        return q_rope, k_rope

    def forward_y_stream(self, y: torch.Tensor, H_y: int, W_y: int, offset_delta: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates Y stream self-attention and extracts K_y, V_y for caching.
        """
        B, N_y, _ = y.shape
        y_norm = self.norm1_y(y)
        qkv_y = self.qkv_y(y_norm).reshape(B, N_y, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_y, k_y, v_y = qkv_y[0], qkv_y[1], qkv_y[2]

        q_y, k_y = self._apply_rope_qk(q_y, k_y, H_y, W_y, offset_delta=offset_delta)

        # Y Self-Attention
        attn_y = F.scaled_dot_product_attention(q_y, k_y, v_y)
        attn_y = attn_y.permute(0, 2, 1, 3).reshape(B, N_y, self.hidden_size)
        y = y + self.out_y(attn_y)

        # Y MLP
        y = y + self.mlp_y(self.norm2_y(y))

        # Prepare K_y, V_y for 1-way Cross-Attention
        y_cross_norm = self.norm_cross_y(y)
        kv_y_cross = self.kv_cross_y(y_cross_norm).reshape(B, N_y, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k_y_cross, v_y_cross = kv_y_cross[0], kv_y_cross[1]
        k_y_cross = self.rope2d.apply_rope(k_y_cross, H_y, W_y, offset_delta=offset_delta)

        return y, k_y_cross, v_y_cross

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        cond: torch.Tensor,
        H_x: int, W_x: int,
        H_y: int, W_y: int,
        is_aligned: bool = True,
        offset_delta: int = 100,
        cached_y_kv: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass for Dual-Stream MM-DiT Block.
        """
        B, N_x, _ = x.shape
        rope_offset = 0 if is_aligned else offset_delta

        # Modulate Stream X with AdaLN(t, c) — 9 parameters
        (shift_msa, scale_msa, gate_msa,
         shift_cross, scale_cross, gate_cross,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_x(cond).chunk(9, dim=-1)

        # Process Stream Y (or load from cache)
        if cached_y_kv is None:
            y, k_y_cross, v_y_cross = self.forward_y_stream(y, H_y, W_y, offset_delta=rope_offset)
        else:
            y, k_y_cross, v_y_cross = cached_y_kv

        # 1. Stream X Self-Attention
        x_norm = modulate(self.norm1_x(x), shift_msa, scale_msa)
        qkv_x = self.qkv_x(x_norm).reshape(B, N_x, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_x, k_x, v_x = qkv_x[0], qkv_x[1], qkv_x[2]
        q_x, k_x = self._apply_rope_qk(q_x, k_x, H_x, W_x, offset_delta=0)

        attn_x_self = F.scaled_dot_product_attention(q_x, k_x, v_x)
        attn_x_self = attn_x_self.permute(0, 2, 1, 3).reshape(B, N_x, self.hidden_size)
        x = x + gate_msa.unsqueeze(1) * self.out_x(attn_x_self)

        # 2. Stream X -> Stream Y 1-Way Cross-Attention (dedicated gate_cross)
        x_cross_norm = modulate(self.norm_cross_x(x), shift_cross, scale_cross)
        q_x_cross = self.q_cross_x(x_cross_norm).reshape(B, N_x, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q_x_cross = self.rope2d.apply_rope(q_x_cross, H_x, W_x, offset_delta=0)

        attn_x_cross = F.scaled_dot_product_attention(q_x_cross, k_y_cross, v_y_cross)
        attn_x_cross = attn_x_cross.permute(0, 2, 1, 3).reshape(B, N_x, self.hidden_size)
        x = x + gate_cross.unsqueeze(1) * self.out_cross_x(attn_x_cross)

        # 3. Stream X MLP
        x_norm2 = modulate(self.norm2_x(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp_x(x_norm2)

        return x, y, (y, k_y_cross, v_y_cross)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DualStreamMMDiT(nn.Module):
    """
    Backbone Dual-Stream MM-DiT Architecture.
    Stream X: Noisy Target Image Latent I_B
    Stream Y: Aligned Structural Map S_aligned
    """

    def __init__(
        self,
        in_channels: int = 4,
        struct_channels: int = 6,
        hidden_size: int = 768,
        depth: int = 16,
        num_heads: int = 12,
        patch_size: int = 2,
        num_sensors: int = 8,
        rope_offset_delta: int = 100
    ):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.rope_offset_delta = rope_offset_delta

        # Linear Patch Embedders
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.y_embedder = nn.Conv2d(struct_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        # Condition & Position Encoders
        self.t_c_embedder = TimestepSensorEmbedder(hidden_size, num_sensors=num_sensors)
        self.rope2d = RotaryEmbedding2D(dim=hidden_size // num_heads)

        # MM-DiT Transformer Blocks
        self.blocks = nn.ModuleList([
            MMDiTBlock(hidden_size, num_heads, self.rope2d) for _ in range(depth)
        ])

        # Final Head for Stream X
        self.final_norm_x = nn.LayerNorm(hidden_size, eps=1e-6)
        self.final_linear_x = nn.Linear(hidden_size, patch_size * patch_size * in_channels)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        struct_map: torch.Tensor,
        is_aligned: bool = True,
        cached_y_kv_list: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Input:
          - x: Latent target image (B, 4, H_x, W_x)
          - t: Timestep (B,)
          - c: Target sensor class ID (B,)
          - struct_map: Pre-registered structural map (B, 6, H_y, W_y)
          - is_aligned: bool flag (True -> shared RoPE grid; False -> offset RoPE grid)
          - cached_y_kv_list: Optional cached Y stream feature list across blocks
        Output:
          - velocity / noise estimate (B, 4, H_x, W_x)
          - updated Y stream cached list
        """
        B, C_x, H_x, W_x = x.shape
        _, C_y, H_y, W_y = struct_map.shape

        # Patchify
        x_tok = self.x_embedder(x).flatten(2).transpose(1, 2)
        y_tok = self.y_embedder(struct_map).flatten(2).transpose(1, 2)

        H_patch_x, W_patch_x = H_x // self.patch_size, W_x // self.patch_size
        H_patch_y, W_patch_y = H_y // self.patch_size, W_y // self.patch_size

        cond = self.t_c_embedder(t, c)

        new_cached_y = []
        out_x, out_y = x_tok, y_tok

        for i, block in enumerate(self.blocks):
            block_cache = cached_y_kv_list[i] if cached_y_kv_list is not None else None
            out_x, out_y, block_y_kv = block(
                out_x, out_y, cond,
                H_patch_x, W_patch_x,
                H_patch_y, W_patch_y,
                is_aligned=is_aligned,
                offset_delta=self.rope_offset_delta,
                cached_y_kv=block_cache
            )
            new_cached_y.append(block_y_kv)

        out_x = self.final_norm_x(out_x)
        logits = self.final_linear_x(out_x)

        P = self.patch_size
        logits = logits.view(B, H_patch_x, W_patch_x, P, P, -1).permute(0, 5, 1, 3, 2, 4).reshape(B, -1, H_x, W_x)

        return logits, new_cached_y


if __name__ == "__main__":
    model = DualStreamMMDiT()
    x_in = torch.randn(2, 4, 32, 32)
    t_in = torch.tensor([10, 500])
    c_in = torch.tensor([0, 3])
    struct_in = torch.randn(2, 6, 64, 64)
    
    # 1. First forward pass (computes & caches Stream Y)
    pred_1, y_cache = model(x_in, t_in, c_in, struct_in, is_aligned=True)
    print("MM-DiT pass 1 output shape:", pred_1.shape)

    # 2. Denoising step 2 using cached Stream Y forward pass (saves compute!)
    pred_2, _ = model(x_in, t_in - 1, c_in, struct_in, is_aligned=True, cached_y_kv_list=y_cache)
    print("MM-DiT pass 2 (cached Y) output shape:", pred_2.shape)
