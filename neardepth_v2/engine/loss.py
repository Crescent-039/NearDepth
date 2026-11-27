"""Nonnegative affine SSI/gradients plus a direction-preserving relative L1 anchor."""
import torch
from torch import nn
from ..config import LossConfig


class RelativeDepthLoss(nn.Module):
    def __init__(self, trim=.1, scales=(1, 2, 4, 8),
                 protocol="positive_affine_anchor_v2", anchor_weight=None):
        super().__init__()
        if not 0 <= trim < 1 or not scales or any(s < 1 for s in scales):
            raise ValueError("invalid trim/scales")
        self.trim, self.scales = trim, scales
        self.protocol = protocol
        self.anchor_weight = (0. if protocol == "affine_v1" else 1.) if anchor_weight is None else anchor_weight
        LossConfig(protocol, self.anchor_weight, trim).validate()

    def forward(self, prediction, target, mask, epoch=0):
        if prediction.shape != target.shape or mask.shape != target.shape:
            raise ValueError("loss shapes must match")
        if not torch.isfinite(prediction).all():
            raise ValueError("non-finite prediction")
        losses, gradients, anchors, negative_scales = [], [], [], []
        for p, t, m in zip(prediction.float(), target.float(), mask.bool()):
            m = m & torch.isfinite(t) & (t > 0)
            if not m.any():
                continue
            pv, tv = p[m], t[m]
            pc, tc = pv - pv.mean(), tv - tv.mean()
            denom = pc.square().sum()
            # Identity fallback avoids a zero-gradient trap for constant predictions.
            unconstrained_scale = torch.where(denom > 1e-8, (pc*tc).sum()/denom.clamp_min(1e-8), denom*0 + 1)
            negative_scales.append((unconstrained_scale.detach() < 0).float())
            scale = unconstrained_scale.clamp_min(0) if self.protocol == "positive_affine_anchor_v2" else unconstrained_scale
            shift = tv.mean() - scale*pv.mean()
            residual = torch.where(m, scale*p + shift - torch.nan_to_num(t), 0.)
            values = residual[m].abs()
            keep = max(1, int(values.numel()*(1-self.trim)))
            losses.append(values.topk(keep, largest=False).values.mean())
            # Targets and the bounded head share [rho_min, 1], larger = farther.
            # This unaligned term also calibrates relative range. Unlike correlation
            # alone or clamping the fitted scale, it gives a correcting gradient at
            # perfect anticorrelation and constant predictions. Do not trim it away.
            anchors.append((pv-tv).abs().mean())
            per_scale = []
            for s in self.scales:
                r, valid = residual[:, ::s, ::s], m[:, ::s, ::s]
                if min(r.shape[-2:]) < 2:
                    continue
                mx = valid[:, :, 1:] & valid[:, :, :-1]
                my = valid[:, 1:, :] & valid[:, :-1, :]
                dx = (r[:, :, 1:] - r[:, :, :-1]).abs()
                dy = (r[:, 1:, :] - r[:, :-1, :]).abs()
                per_scale.append((dx*mx).sum()/mx.sum().clamp_min(1) + (dy*my).sum()/my.sum().clamp_min(1))
            gradients.append(torch.stack(per_scale).mean() if per_scale else p.sum()*0)
        if not losses:
            raise ValueError("batch has no valid supervision")
        ssi, grad = torch.stack(losses).mean(), torch.stack(gradients).mean()
        anchor = torch.stack(anchors).mean()
        weight = .2 if epoch < 2 else 2.
        total = ssi + weight*grad + self.anchor_weight*anchor
        return total, {"ssi": ssi.detach().item(), "gradient": grad.detach().item(),
                       "gradient_weight": weight, "anchor": anchor.detach().item(),
                       "anchor_weight": self.anchor_weight,
                       "negative_scale_fraction": torch.stack(negative_scales).mean().item()}
