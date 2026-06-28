"""Depth evaluation metrics and affine-invariant alignment.

Adapted from the Marigold (CVPR 2024) evaluation protocol. ZipDepth predicts
affine-invariant inverse depth, so predictions are aligned to the ground truth
with a least-squares scale-and-shift before metrics are computed.
"""

from typing import Dict, Optional

import numpy as np


# =============================================================================
# ALIGNMENT
# =============================================================================

def align_depth_least_square(
    gt: np.ndarray,
    pred: np.ndarray,
    valid_mask: np.ndarray,
    max_resolution: Optional[int] = None,
) -> np.ndarray:
    """Find scale ``s`` and shift ``t`` so that ``s * pred + t`` best matches ``gt``.

    Args:
        gt: Ground-truth array (depth or disparity).
        pred: Prediction in the same space as ``gt``.
        valid_mask: Boolean mask of pixels to use for the fit.
        max_resolution: If set, downsample before solving for speed.

    Returns:
        ``s * pred + t`` at the original resolution.
    """
    gt_ds, pred_ds, mask_ds = gt, pred, valid_mask
    if max_resolution is not None:
        h, w = gt.shape[:2]
        if max(h, w) > max_resolution:
            import cv2
            scale_f = max_resolution / max(h, w)
            new_h, new_w = int(h * scale_f), int(w * scale_f)
            gt_ds = cv2.resize(gt, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            pred_ds = cv2.resize(pred, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            mask_ds = cv2.resize(valid_mask.astype(np.uint8), (new_w, new_h),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)

    gt_valid = gt_ds[mask_ds].astype(np.float64)
    pred_valid = pred_ds[mask_ds].astype(np.float64)

    if len(gt_valid) < 10:
        return pred.copy()

    # Solve [pred, 1] @ [s, t]^T = gt in the least-squares sense.
    A = np.stack([pred_valid, np.ones_like(pred_valid)], axis=1)
    scale, shift = np.linalg.lstsq(A, gt_valid, rcond=None)[0]
    return scale * pred + shift


def depth2disparity(depth: np.ndarray, return_mask: bool = False):
    """Convert depth to disparity (1/depth), leaving non-positive values at 0."""
    non_neg = depth > 0
    disparity = np.zeros_like(depth)
    disparity[non_neg] = 1.0 / depth[non_neg]
    if return_mask:
        return disparity, non_neg
    return disparity


def disparity2depth(disparity: np.ndarray) -> np.ndarray:
    """Convert disparity to depth (1/disparity), leaving non-positive values at 0."""
    mask = disparity > 0
    depth = np.zeros_like(disparity)
    depth[mask] = 1.0 / disparity[mask]
    return depth


# =============================================================================
# METRICS  (all operate on the masked, aligned depth)
# =============================================================================

def abs_relative_difference(pred, gt, mask):
    """AbsRel = mean(|pred - gt| / gt)."""
    return float(np.mean(np.abs(pred[mask] - gt[mask]) / gt[mask]))


def squared_relative_difference(pred, gt, mask):
    """SqRel = mean((pred - gt)^2 / gt)."""
    return float(np.mean((pred[mask] - gt[mask]) ** 2 / gt[mask]))


def rmse_linear(pred, gt, mask):
    """RMSE = sqrt(mean((pred - gt)^2))."""
    return float(np.sqrt(np.mean((pred[mask] - gt[mask]) ** 2)))


def rmse_log(pred, gt, mask):
    """RMSElog = sqrt(mean((log(pred) - log(gt))^2))."""
    return float(np.sqrt(np.mean((np.log(pred[mask]) - np.log(gt[mask])) ** 2)))


def log10_error(pred, gt, mask):
    """log10 = mean(|log10(pred) - log10(gt)|)."""
    return float(np.mean(np.abs(np.log10(pred[mask]) - np.log10(gt[mask]))))


def delta_acc(pred, gt, mask, threshold):
    ratio = np.maximum(pred[mask] / gt[mask], gt[mask] / pred[mask])
    return float(np.mean(ratio < threshold))


def delta1_acc(pred, gt, mask):
    return delta_acc(pred, gt, mask, 1.25)


def delta2_acc(pred, gt, mask):
    return delta_acc(pred, gt, mask, 1.25 ** 2)


def delta3_acc(pred, gt, mask):
    return delta_acc(pred, gt, mask, 1.25 ** 3)


METRIC_FUNCTIONS = {
    'abs_rel': abs_relative_difference,
    'sq_rel': squared_relative_difference,
    'rmse': rmse_linear,
    'rmse_log': rmse_log,
    'log10': log10_error,
    'delta1': delta1_acc,
    'delta2': delta2_acc,
    'delta3': delta3_acc,
}

# Whether a lower value is better (for display only).
LOWER_IS_BETTER = {'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10'}
HIGHER_IS_BETTER = {'delta1', 'delta2', 'delta3'}


def compute_depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray,
    metrics: Optional[list] = None,
) -> Dict[str, float]:
    """Compute all metrics on the valid, strictly-positive pixels."""
    if metrics is None:
        metrics = list(METRIC_FUNCTIONS.keys())

    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    mask = valid_mask & (pred > 0) & (gt > 0)

    if mask.sum() < 10:
        return {m: float('nan') for m in metrics}

    results = {}
    for name in metrics:
        try:
            results[name] = METRIC_FUNCTIONS[name](pred, gt, mask)
        except Exception:
            results[name] = float('nan')
    return results


# =============================================================================
# METRIC TRACKER
# =============================================================================

class MetricTracker:
    """Accumulate per-sample metrics and report dataset-level means."""

    def __init__(self, metric_names: list):
        self.names = metric_names
        self.reset()

    def reset(self):
        self._values = {n: [] for n in self.names}

    def update(self, metrics: Dict[str, float]):
        for name, val in metrics.items():
            if name in self._values and not np.isnan(val):
                self._values[name].append(val)

    def result(self) -> Dict[str, float]:
        return {
            name: float(np.mean(vals)) if vals else float('nan')
            for name, vals in self._values.items()
        }

    def count(self) -> int:
        return max((len(v) for v in self._values.values()), default=0)
