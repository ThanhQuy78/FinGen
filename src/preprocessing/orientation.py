import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple


def compute_gradient_orientation(img: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """
    Computes local ridge orientation map using gradient method.
    Input: img tensor of shape (B, 1, H, W) normalized to [0, 1]
    Output: orientation map tensor of shape (B, 2, H // block_size, W // block_size)
            containing continuous (cos(2θ), sin(2θ)).
    """
    B, C, H, W = img.shape
    
    # Sobel kernels for Gradients
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    
    Gx = F.conv2d(img, sobel_x, padding=1)
    Gy = F.conv2d(img, sobel_y, padding=1)
    
    # Gradient covariance components
    Gxx = Gx * Gx
    Gyy = Gy * Gy
    Gxy = Gx * Gy
    
    # Average over local blocks using average pooling
    avg_pool = nn.AvgPool2d(kernel_size=block_size, stride=block_size)
    
    V_x = 2.0 * avg_pool(Gxy)
    V_y = avg_pool(Gxx - Gyy)
    
    # Orientation angle theta = 0.5 * atan2(V_x, V_y) + pi/2
    # cos(2θ) = -V_y / sqrt(V_x^2 + V_y^2)
    # sin(2θ) = V_x / sqrt(V_x^2 + V_y^2)
    
    norm = torch.sqrt(V_x * V_x + V_y * V_y + 1e-8)
    cos2theta = -V_y / norm
    sin2theta = V_x / norm
    
    orientation_map = torch.cat([cos2theta, sin2theta], dim=1) # (B, 2, h_block, w_block)
    return orientation_map


def orientation_to_angle(orient_map: torch.Tensor) -> torch.Tensor:
    """
    Converts continuous (cos(2θ), sin(2θ)) tensor back to theta angle in [0, pi).
    Input: orient_map of shape (..., 2, H, W) where channel 0 is cos(2θ), channel 1 is sin(2θ)
    Output: theta tensor of shape (..., 1, H, W) in radians [0, pi).
    """
    cos2t = orient_map[..., 0:1, :, :]
    sin2t = orient_map[..., 1:2, :, :]
    two_theta = torch.atan2(sin2t, cos2t)
    theta = 0.5 * two_theta
    theta = torch.where(theta < 0, theta + np.pi, theta)
    return theta


def compute_orientation_coherence(orient1: torch.Tensor, orient2: torch.Tensor) -> torch.Tensor:
    """
    Computes cosine distance / coherence loss between two orientation fields (cos2θ, sin2θ).
    Input: orient1, orient2 of shape (B, 2, H, W)
    Output: scalar loss or map of per-pixel orientation error.
    """
    # Normalize vectors
    o1_norm = F.normalize(orient1, p=2, dim=1, eps=1e-8)
    o2_norm = F.normalize(orient2, p=2, dim=1, eps=1e-8)
    
    # Cosine similarity in (cos2θ, sin2θ) space
    cos_sim = torch.sum(o1_norm * o2_norm, dim=1, keepdim=True)
    # Coherence loss: 1 - cosine similarity
    loss = 1.0 - cos_sim
    return loss.mean()
