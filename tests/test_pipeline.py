import unittest
import torch
import numpy as np

from src.preprocessing.orientation import compute_gradient_orientation, orientation_to_angle, compute_orientation_coherence
from src.preprocessing.fingernet_extractor import FingerNetExtractor
from src.preprocessing.tps_aligner import ThinPlateSplineAligner
from src.models.rope2d import RotaryEmbedding2D
from src.models.controlnet_baseline import ControlNetTransformerBaseline
from src.models.mm_dit import DualStreamMMDiT
from src.models.flow_matching import RectifiedFlowTrajectoryManager
from src.losses.identity_loss import IdentityCosineLoss
from src.losses.orientation_loss import OrientationCoherenceLoss
from src.losses.loss_builder import Stage4CompositeLossBuilder


class TestFingerprintPipeline(unittest.TestCase):

    def test_orientation_continuous_representation(self):
        # Create a structured image with gradients
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij")
        img = torch.sin(10.0 * (grid_x**2 + grid_y**2)).unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1)
        
        orient_map = compute_gradient_orientation(img, block_size=16)
        # Check output channels == 2 for (cos2θ, sin2θ)
        self.assertEqual(orient_map.shape, (2, 2, 4, 4))
        
        # Check continuous orientation vector norm
        norms = torch.sqrt(orient_map[:, 0]**2 + orient_map[:, 1]**2)
        self.assertTrue(torch.all(norms >= 0.0) and torch.all(norms <= 1.05))

    def test_tps_aligner_structure(self):
        aligner = ThinPlateSplineAligner()
        dummy_struct = torch.randn(1, 6, 64, 64)
        src_pts = np.array([[10, 10], [50, 10], [10, 50], [50, 50]])
        dst_pts = np.array([[12, 11], [49, 11], [11, 48], [51, 52]])
        
        aligned_struct, is_aligned = aligner.align_structure_tensor(dummy_struct, src_pts, dst_pts)
        self.assertTrue(is_aligned)
        self.assertEqual(aligned_struct.shape, (1, 6, 64, 64))

    def test_rope2d_offset(self):
        rope = RotaryEmbedding2D(dim=64)
        q = torch.randn(2, 4, 256, 64)
        q_shared = rope.apply_rope(q, H=16, W=16, offset_delta=0)
        q_offset = rope.apply_rope(q, H=16, W=16, offset_delta=100)
        
        self.assertEqual(q_shared.shape, (2, 4, 256, 64))
        self.assertEqual(q_offset.shape, (2, 4, 256, 64))
        self.assertFalse(torch.allclose(q_shared, q_offset))

    def test_controlnet_baseline_forward(self):
        model = ControlNetTransformerBaseline()
        x = torch.randn(2, 4, 32, 32)
        t = torch.tensor([10, 500])
        c = torch.tensor([0, 2])
        s = torch.randn(2, 6, 64, 64)
        
        out = model(x, t, c, s)
        self.assertEqual(out.shape, (2, 4, 32, 32))

    def test_mm_dit_caching_and_forward(self):
        model = DualStreamMMDiT(hidden_size=256, depth=4, num_heads=4)
        x = torch.randn(2, 4, 16, 16)
        t = torch.tensor([100, 200])
        c = torch.tensor([1, 3])
        s = torch.randn(2, 6, 32, 32)
        
        # Pass 1 (uncached)
        out1, y_cache = model(x, t, c, s, is_aligned=True)
        self.assertEqual(out1.shape, (2, 4, 16, 16))
        self.assertEqual(len(y_cache), 4)

        # Pass 2 (cached Stream Y)
        out2, _ = model(x, t - 1, c, s, is_aligned=True, cached_y_kv_list=y_cache)
        self.assertEqual(out2.shape, (2, 4, 16, 16))

    def test_stage4_composite_loss_warmup(self):
        loss_builder = Stage4CompositeLossBuilder(warmup_epochs=3)
        v_pred = torch.randn(2, 4, 16, 16, requires_grad=True)
        v_target = torch.randn(2, 4, 16, 16)
        x0_est = torch.randn(2, 1, 128, 128, requires_grad=True)
        img_a = torch.randn(2, 1, 128, 128)
        s_aligned = torch.randn(2, 6, 128, 128)
        t = torch.tensor([0.1, 0.5])

        # Epoch 0 (Warmup)
        l_dict_ep0 = loss_builder(v_pred, v_target, x0_est, img_a, s_aligned, t, epoch=0)
        self.assertEqual(l_dict_ep0["weight_identity"].item(), 0.0)

        # Epoch 5 (Post Warmup)
        l_dict_ep5 = loss_builder(v_pred, v_target, x0_est, img_a, s_aligned, t, epoch=5)
        self.assertGreater(l_dict_ep5["weight_identity"].item(), 0.0)
        
        l_dict_ep5["loss_total"].backward()
        self.assertIsNotNone(v_pred.grad)

    def test_flow_matching_trajectory_reconstruction(self):
        """Verify x_1 = x_t + (1-t)*v_target exactly reconstructs the target."""
        traj = RectifiedFlowTrajectoryManager()
        target = torch.randn(2, 4, 8, 8)
        source = torch.randn(2, 4, 8, 8)

        x_t, t, v_target, x_0 = traj.sample_trajectory(target, source, use_direct_bridge=True)
        x1_est = traj.compute_x0_estimate(x_t, v_target, t)

        self.assertTrue(torch.allclose(x1_est, target, atol=1e-5))

    def test_fingernet_extractor_output_shapes(self):
        """Verify CoarseNet-based FingerNet outputs correct channel counts and spatial resolution."""
        extractor = FingerNetExtractor()
        # Input must be divisible by 8 (3 MaxPool stages in CoarseNet backbone)
        img = torch.randn(2, 1, 128, 128)
        out = extractor(img)

        self.assertEqual(out["segmentation_map"].shape, (2, 1, 128, 128))
        self.assertEqual(out["orientation_map"].shape, (2, 2, 128, 128))
        self.assertEqual(out["minutiae_map"].shape, (2, 3, 128, 128))
        self.assertEqual(out["combined_structure"].shape, (2, 6, 128, 128))

        # Verify orientation map vectors are approximately unit-normalized
        orient_norms = torch.norm(out["orientation_map"], p=2, dim=1)
        self.assertTrue(torch.allclose(orient_norms, torch.ones_like(orient_norms), atol=0.01))

    def test_identity_loss_gradient_flow_to_x0(self):
        """Verify gradients from IdentityCosineLoss flow back to x0_est input."""
        loss_fn = IdentityCosineLoss()
        x0 = torch.randn(2, 1, 64, 64, requires_grad=True)
        img_a = torch.randn(2, 1, 64, 64)

        loss = loss_fn(x0, img_a)
        loss.backward()

        self.assertIsNotNone(x0.grad)
        self.assertGreater(x0.grad.abs().sum().item(), 0.0)

    def test_orientation_loss_differentiability(self):
        """Verify OrientationCoherenceLoss is differentiable w.r.t. x0_est."""
        loss_fn = OrientationCoherenceLoss()
        x0 = torch.randn(2, 1, 64, 64, requires_grad=True)
        s_aligned = torch.randn(2, 6, 64, 64)

        loss = loss_fn(x0, s_aligned)
        loss.backward()

        self.assertIsNotNone(x0.grad)
        self.assertGreater(x0.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
