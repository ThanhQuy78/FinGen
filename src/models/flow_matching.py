import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Callable


class RectifiedFlowTrajectoryManager:
    """
    Manager for Rectified Flow / Flow Matching trajectory generation, loss computation,
    $x_0$-estimation, and Euler sampling.
    
    Supports:
    1. Gaussian Noise Trajectory: x_0 ~ N(0, I) -> x_1 = target image I_B
    2. Direct Flow Bridge (I²SB): x_0 = source image I_A -> x_1 = target image I_B
    """

    def __init__(self, sigma_min: float = 1e-5):
        self.sigma_min = sigma_min

    def sample_trajectory(
        self,
        target_latent: torch.Tensor,
        source_latent: Optional[torch.Tensor] = None,
        use_direct_bridge: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples random timestep t in [0, 1] and constructs interpolated state x_t and target velocity v_target.
        
        If use_direct_bridge=False:
            x_0 ~ N(0, I)
            x_1 = target_latent
        If use_direct_bridge=True:
            x_0 = source_latent
            x_1 = target_latent
            
        x_t = (1 - t) * x_0 + t * x_1
        v_target = x_1 - x_0
        """
        B = target_latent.shape[0]
        device = target_latent.device
        dtype = target_latent.dtype

        # Timestep t uniformly sampled in [0, 1]
        t = torch.rand(B, device=device, dtype=dtype)
        t_expand = t.view(B, 1, 1, 1)

        if use_direct_bridge and source_latent is not None:
            x_0 = source_latent
        else:
            x_0 = torch.randn_like(target_latent)

        x_1 = target_latent

        # Linear velocity interpolation trajectory
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
        v_target = x_1 - x_0

        return x_t, t, v_target, x_0

    def compute_x0_estimate(self, x_t: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Calculates estimated reconstructed target x_1 (or noise start x_0) from predicted velocity v_pred.
        
        Since x_t = (1 - t) x_0 + t x_1 and v = x_1 - x_0:
        Estimated x_1 = x_t + (1 - t) * v_pred
        """
        B = x_t.shape[0]
        t_expand = t.view(B, 1, 1, 1)
        x1_est = x_t + (1.0 - t_expand) * v_pred
        return x1_est

    @torch.no_grad()
    def sample_euler(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        c: torch.Tensor,
        struct_map: torch.Tensor,
        is_aligned: bool = True,
        steps: int = 20,
        source_latent: Optional[torch.Tensor] = None,
        use_direct_bridge: bool = False
    ) -> torch.Tensor:
        """
        Euler ODE sampling trajectory from t=0 to t=1 over NFE (Number of Function Evaluations) steps.
        """
        B = shape[0]
        device = struct_map.device

        dt = 1.0 / steps
        if use_direct_bridge and source_latent is not None:
            x = source_latent.clone()
        else:
            x = torch.randn(shape, device=device)

        cached_y_kv = None

        for step in range(steps):
            t_val = step / steps
            t = torch.full((B,), t_val, device=device)

            # Model forward pass (uses Y-stream caching across steps if model is MM-DiT)
            if hasattr(model, "blocks") and hasattr(model.blocks[0], "forward_y_stream"):
                v_pred, cached_y_kv = model(
                    x, t, c, struct_map, is_aligned=is_aligned, cached_y_kv_list=cached_y_kv
                )
            else:
                v_pred = model(x, t, c, struct_map)

            # Euler step: x_{t+dt} = x_t + dt * v_pred
            x = x + dt * v_pred

        return x


if __name__ == "__main__":
    traj = RectifiedFlowTrajectoryManager()
    target_lat = torch.randn(4, 4, 32, 32)
    source_lat = torch.randn(4, 4, 32, 32)
    
    x_t, t, v_tgt, x_0 = traj.sample_trajectory(target_lat, source_lat, use_direct_bridge=True)
    x1_est = traj.compute_x0_estimate(x_t, v_tgt, t)
    
    diff = torch.norm(x1_est - target_lat).item()
    print(f"Trajectory test: x_t shape {x_t.shape}, x1_est reconstruction diff: {diff:.6f}")
