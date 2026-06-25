"""Training-time visualization utilities for TensorBoard logging."""

import torch
import matplotlib.pyplot as plt


def depth_to_spectral(depth_tensor: torch.Tensor) -> torch.Tensor:
    """Depth map → Spectral colormap (inverted). Accepts (H,W), (B,H,W) or (B,1,H,W)."""
    if depth_tensor.ndim == 4 and depth_tensor.shape[1] == 1:
        depth_tensor = depth_tensor.squeeze(1)
    if depth_tensor.ndim == 4:
        raise ValueError(f"Expected 1-channel depth, got {depth_tensor.shape}")

    def _single(d: torch.Tensor) -> torch.Tensor:
        d_np = -d.detach().cpu().numpy()
        d_np = (d_np - d_np.min()) / (d_np.max() - d_np.min() + 1e-8)
        c = plt.cm.Spectral(d_np)[:, :, :3]
        return torch.tensor(c, dtype=torch.float32).permute(2, 0, 1)

    if depth_tensor.ndim == 3:
        return torch.stack([_single(d) for d in depth_tensor])
    return _single(depth_tensor)
