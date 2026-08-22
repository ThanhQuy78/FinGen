import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ThinPlateSplineAligner:
    """
    TPS (Thin Plate Spline) Shape Transformer for Pre-Registration of structural maps.
    Aligns source structural map S_A to target sensor geometry producing S_aligned.
    """

    def __init__(self, regularization_parameter: float = 0.1):
        self.reg_param = regularization_parameter

    def fit_transform_points(self, src_pts: np.ndarray, dst_pts: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Fits TPS transformer given paired minutiae source points (N, 2) and target points (N, 2),
        and deforms input image S_A into S_aligned.
        """
        if len(src_pts) < 4 or len(dst_pts) < 4 or len(src_pts) != len(dst_pts):
            # Fallback to affine or identity if insufficient minutiae points
            return image

        # Reshape points for OpenCV ThinPlateSplineShapeTransformer
        src_pts_cv = src_pts.reshape(1, -1, 2).astype(np.float32)
        dst_pts_cv = dst_pts.reshape(1, -1, 2).astype(np.float32)

        matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]

        tps = cv2.createThinPlateSplineShapeTransformer()
        tps.setRegularizationParameter(self.reg_param)
        tps.estimateTransformation(dst_pts_cv, src_pts_cv, matches)

        warped_img = tps.warpImage(image)
        return warped_img

    def align_structure_tensor(
        self,
        src_structure: torch.Tensor,
        src_pts: Optional[np.ndarray] = None,
        dst_pts: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, bool]:
        """
        PyTorch wrapper for structure tensor alignment.
        Input: src_structure of shape (B, C, H, W)
        Returns: (aligned_tensor, is_aligned_flag)
        """
        B, C, H, W = src_structure.shape
        aligned_tensors = []

        if src_pts is None or dst_pts is None or len(src_pts) < 4:
            # Unaligned mode: return original structure unchanged, flag is_aligned = False
            return src_structure, False

        for b in range(B):
            channels_aligned = []
            for c in range(C):
                channel_np = src_structure[b, c].cpu().numpy()
                aligned_np = self.fit_transform_points(src_pts, dst_pts, channel_np)
                channels_aligned.append(torch.from_numpy(aligned_np).to(src_structure.device))
            aligned_tensors.append(torch.stack(channels_aligned, dim=0))

        aligned_batch = torch.stack(aligned_tensors, dim=0)
        return aligned_batch, True


if __name__ == "__main__":
    aligner = ThinPlateSplineAligner()
    dummy_structure = torch.randn(1, 6, 256, 256)
    out, aligned = aligner.align_structure_tensor(dummy_structure)
    print(f"Structure alignment test: output shape {out.shape}, aligned={aligned}")
