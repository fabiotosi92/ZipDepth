"""Scale-and-shift invariant depth loss with multi-scale gradient matching."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZipDepthLoss(nn.Module):
    """
    Combined loss for affine-invariant monocular depth estimation.

    Normalizes predictions and targets via median/MAD before computing:
      - SSI loss:      masked L1 on normalized maps
      - Gradient loss: masked L1 on image gradients at multiple scales
    """

    def __init__(
        self,
        alpha_ssi: float = 1.0,
        alpha_grad: float = 2.0,
        grad_scales: int = 4,
    ):
        super().__init__()
        self.alpha = {'ssi': alpha_ssi, 'grad': alpha_grad}
        self.grad_scales = grad_scales

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        if mask is None:
            mask = torch.ones_like((target > 0).float())

        p_hat, t_hat = self._normalize(pred, target, mask)
        ssi_loss  = self._ssi_loss(p_hat, t_hat, mask)
        grad_loss = self._gradient_loss(p_hat, t_hat, mask)

        total = self.alpha['ssi'] * ssi_loss + self.alpha['grad'] * grad_loss
        return total, {'ssi': ssi_loss.detach().item(), 'grad': grad_loss.detach().item()}

    # ------------------------------------------------------------------
    def _normalize(self, pred, target, mask):
        pred, target, mask = self._squeeze(pred, target, mask)
        B = pred.shape[0]

        pred_nan = pred.clone()
        tgt_nan  = target.clone()
        pred_nan[mask < 0.5] = float('nan')
        tgt_nan[mask < 0.5]  = float('nan')

        p_med = torch.nanmedian(pred_nan.view(B, -1), dim=1).values.view(B, 1, 1)
        t_med = torch.nanmedian(tgt_nan.view(B, -1),  dim=1).values.view(B, 1, 1)

        M     = mask.sum(dim=(1, 2), keepdim=True) + 1e-8
        p_mad = ((pred   - p_med).abs() * mask).sum(dim=(1, 2), keepdim=True) / M
        t_mad = ((target - t_med).abs() * mask).sum(dim=(1, 2), keepdim=True) / M

        p_hat = (pred   - p_med) / p_mad.clamp(min=1e-6)
        t_hat = (target - t_med) / t_mad.clamp(min=1e-6)
        return p_hat, t_hat

    def _ssi_loss(self, pred, target, mask):
        pred, target, mask = self._squeeze(pred, target, mask)
        diff = (pred - target).abs()
        M = mask.sum() + 1e-8
        return (diff * mask).sum() / M

    def _gradient_loss(self, pred, target, mask):
        pred, target, mask = self._squeeze(pred, target, mask)
        total = 0.0
        for scale in range(self.grad_scales):
            k = 2 ** scale
            if k > 1:
                p = F.avg_pool2d(pred.unsqueeze(1),   k).squeeze(1)
                t = F.avg_pool2d(target.unsqueeze(1), k).squeeze(1)
                m = (F.avg_pool2d(mask.unsqueeze(1),  k).squeeze(1) > 0.99).float()
            else:
                p, t, m = pred, target, mask

            p_dx = p[:, :, 1:] - p[:, :, :-1]
            p_dy = p[:, 1:, :] - p[:, :-1, :]
            t_dx = t[:, :, 1:] - t[:, :, :-1]
            t_dy = t[:, 1:, :] - t[:, :-1, :]

            m_x = m[:, :, 1:] * m[:, :, :-1]
            m_y = m[:, 1:, :] * m[:, :-1, :]

            loss_x = ((p_dx - t_dx).abs() * m_x).sum()
            loss_y = ((p_dy - t_dy).abs() * m_y).sum()
            denom  = m_x.sum() + m_y.sum() + 1e-8
            total += (loss_x + loss_y) / denom

        return total / self.grad_scales

    @staticmethod
    def _squeeze(*tensors):
        return tuple(t.squeeze(1) if t.dim() == 4 else t for t in tensors)
