import torch
import torch.nn as nn
import torch.nn.functional as F
from depth_config import DEFAULT_RHO_MIN


# -------------------- 基础工具 --------------------

def safe_masked_mean(x, mask, eps=1e-6):
    # x, mask shape一致
    num = (x * mask).sum()
    den = mask.sum() + eps
    return num / den


def erode_valid_mask(valid_mask, k=3):
    # valid_mask: float [B,1,H,W], 1有效
    # 腐蚀，避免边界邻域受无效值污染
    return 1.0 - F.max_pool2d(1.0 - valid_mask, kernel_size=k, stride=1, padding=k // 2)


def sobel_grad(x):
    # x: [B,1,H,W]
    kx = torch.tensor([[-1., 0., 1.],
                       [-2., 0., 2.],
                       [-1., 0., 1.]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = torch.tensor([[-1., -2., -1.],
                       [ 0.,  0.,  0.],
                       [ 1.,  2.,  1.]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return gx, gy


def grad_mag(x, eps=1e-6):
    gx, gy = sobel_grad(x)
    return torch.sqrt(gx * gx + gy * gy + eps)


def gaussian_blur5(x):
    # x: [B,1,H,W]
    k = torch.tensor([
        [1., 4., 6., 4., 1.],
        [4.,16.,24.,16., 4.],
        [6.,24.,36.,24., 6.],
        [4.,16.,24.,16., 4.],
        [1., 4., 6., 4., 1.]
    ], device=x.device, dtype=x.dtype) / 256.0
    k = k.view(1,1,5,5)
    return F.conv2d(x, k, padding=2)


# -------------------- 你原来的Chamfer可保留 --------------------
class BinsChamferLoss(nn.Module):
    def __init__(self, resize_resolution=(64, 64)):
        super().__init__()
        self.resize_resolution = resize_resolution

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])   # [B,N]
        gt_small = F.interpolate(target_depth_maps, size=self.resize_resolution, mode='nearest')

        b, n = bin_centers.shape
        x = bin_centers.unsqueeze(2)   # [B,N,1]
        y = gt_small.view(b, 1, -1)    # [B,1,M]

        valid_mask = y > 1e-3
        if valid_mask.sum() < 10:
            return torch.tensor(0.0, device=bins.device, requires_grad=True)

        large_val = 1000.0
        y_masked = torch.where(valid_mask, y, torch.tensor(large_val, device=y.device))

        dist_matrix = torch.abs(x - y_masked)  # [B,N,M]

        min_dist_bin2gt, _ = torch.min(dist_matrix, dim=2)  # [B,N]
        loss_bin2gt = min_dist_bin2gt.mean()

        min_dist_gt2bin, _ = torch.min(dist_matrix, dim=1)  # [B,M]
        valid_mask_flat = valid_mask.squeeze(1).float()
        num_valid = valid_mask_flat.sum() + 1e-6
        loss_gt2bin = (min_dist_gt2bin * valid_mask_flat).sum() / num_valid

        return loss_bin2gt + loss_gt2bin


# -------------------- 改造后的核心损失 --------------------

class SafeSILogLoss(nn.Module):
    def __init__(self, eps=1e-6, lambda_mean=0.15, scale=10.0):
        super().__init__()
        self.eps = eps
        self.lambda_mean = lambda_mean
        self.scale = scale

    def forward(self, pred, target, valid_mask=None):
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)

        pred = torch.clamp(pred, min=1e-4)
        target = torch.clamp(target, min=1e-4)

        if valid_mask is None:
            valid_mask = (target > 1e-4).float()
        else:
            valid_mask = valid_mask.float()

        pv = pred[valid_mask > 0.5]
        tv = target[valid_mask > 0.5]

        if pv.numel() < 2:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        g = torch.log(pv) - torch.log(tv)
        Dg = torch.var(g, unbiased=False) + self.lambda_mean * (torch.mean(g) ** 2)
        return self.scale * torch.sqrt(Dg + self.eps)


class MaskedL1Loss(nn.Module):
    def __init__(self, min_depth=1e-3, max_depth=80.0):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, pred, gt, valid_mask=None):
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(pred, size=gt.shape[-2:], mode='bilinear', align_corners=False)

        mask = ((gt > self.min_depth) & (gt < self.max_depth)).float()
        if valid_mask is not None:
            mask = mask * valid_mask.float()

        return safe_masked_mean(torch.abs(pred - gt), mask)


class LowFreqLoss(nn.Module):
    def forward(self, pred, gt, valid_mask):
        pred_l = gaussian_blur5(pred)
        gt_l = gaussian_blur5(gt)
        return safe_masked_mean(torch.abs(pred_l - gt_l), valid_mask.float())


class EdgeBandBuilder(nn.Module):
    """
    用GT深度梯度构造“边界带”:
    - 只监督真实边界附近，不全图硬拉梯度
    """
    def __init__(self, q=0.88, dilate=1):
        super().__init__()
        self.q = q
        self.dilate = dilate

    def forward(self, gt_depth, valid_mask):
        # gt_depth: [B,1,H,W], valid_mask float
        B = gt_depth.shape[0]
        gmag = grad_mag(gt_depth)  # [B,1,H,W]
        edge_band = torch.zeros_like(gmag)

        safe_valid = erode_valid_mask(valid_mask.float(), k=3)

        with torch.no_grad():
            for b in range(B):
                vb = safe_valid[b] > 0.5
                vals = gmag[b][vb]
                if vals.numel() < 32:
                    continue
                thr = torch.quantile(vals, self.q)
                eb = (gmag[b:b+1] >= thr).float()
                if self.dilate > 0:
                    k = 2 * self.dilate + 1
                    eb = F.max_pool2d(eb, kernel_size=k, stride=1, padding=self.dilate)
                edge_band[b:b+1] = eb

        edge_band = edge_band * safe_valid
        return edge_band, safe_valid


class EdgeBandGradLoss(nn.Module):
    """
    仅在边界带比较梯度差异（pred vs gt）
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, edge_band):
        # 用log-depth梯度更稳定
        pred_log = torch.log(torch.clamp(pred, min=1e-4))
        gt_log = torch.log(torch.clamp(gt, min=1e-4))

        pgx, pgy = sobel_grad(pred_log)
        ggx, ggy = sobel_grad(gt_log)

        loss_map = F.smooth_l1_loss(pgx, ggx, reduction='none') + \
                   F.smooth_l1_loss(pgy, ggy, reduction='none')

        return safe_masked_mean(loss_map, edge_band.float())


class NonEdgeSmoothLoss(nn.Module):
    """
    非边界区域惩罚深度梯度：专门抑制纹理噪声
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, edge_band, safe_valid):
        pred_log = torch.log(torch.clamp(pred, min=1e-4))
        g = grad_mag(pred_log)
        non_edge = (1.0 - edge_band) * safe_valid
        return safe_masked_mean(g, non_edge.float())


# 新增的anydepth的loss
def safe_masked_mean(x, mask, eps=1e-6):
    return (x * mask).sum() / (mask.sum() + eps)


def zero_loss_like(x):
    return x.sum() * 0.0


def masked_trimmed_mean(x, mask, trim=0.0, eps=1e-6):
    vals = x[mask > 0.5]
    if vals.numel() == 0:
        return zero_loss_like(x)
    if trim > 0 and vals.numel() > 1:
        keep = max(int(vals.numel() * (1.0 - trim)), 1)
        vals, _ = torch.topk(vals, k=keep, largest=False)
    return vals.mean()


def solve_scale_shift(pred, target, mask, eps=1e-6):
    """
    pred/target/mask: [B,1,H,W]
    对每张图求解:
      s, t = argmin || m * (s*pred + t - target) ||^2
    """
    a00 = torch.sum(mask * pred * pred, dim=(1, 2, 3))   # [B]
    a01 = torch.sum(mask * pred, dim=(1, 2, 3))          # [B]
    a11 = torch.sum(mask, dim=(1, 2, 3))                 # [B]

    b0 = torch.sum(mask * pred * target, dim=(1, 2, 3))  # [B]
    b1 = torch.sum(mask * target, dim=(1, 2, 3))         # [B]

    det = a00 * a11 - a01 * a01

    scale = torch.ones_like(det)
    shift = torch.zeros_like(det)

    valid = (det.abs() > eps) & (a11 > 1.0)
    scale[valid] = (a11[valid] * b0[valid] - a01[valid] * b1[valid]) / det[valid]
    shift[valid] = (-a01[valid] * b0[valid] + a00[valid] * b1[valid]) / det[valid]

    scale = scale.view(-1, 1, 1, 1)
    shift = shift.view(-1, 1, 1, 1)
    return scale, shift


class ScaleShiftInvariantLoss(nn.Module):
    def __init__(self, eps=1e-6, use_l1=False):
        super().__init__()
        self.eps = eps
        self.use_l1 = use_l1  # False=L2(MSE), True=L1

    def forward(self, pred, target, mask, return_aligned=False):
        scale, shift = solve_scale_shift(pred, target, mask, eps=self.eps)
        pred_aligned = scale * pred + shift

        diff = pred_aligned - target
        if self.use_l1:
            loss_map = diff.abs()
        else:
            loss_map = diff * diff

        loss = safe_masked_mean(loss_map, mask, eps=self.eps)

        if return_aligned:
            return loss, pred_aligned
        return loss


class GradientMatchingLoss(nn.Module):
    """
    多尺度梯度匹配（DAv2/MiDaS常见做法）
    """
    def __init__(self, scales=(1, 2, 4, 8), eps=1e-6, use_smooth_l1=False):
        super().__init__()
        self.scales = scales
        self.eps = eps
        self.use_smooth_l1 = use_smooth_l1

    def _grad_loss_once(self, p, t, m):
        # x方向
        pdx = p[:, :, :, 1:] - p[:, :, :, :-1]
        tdx = t[:, :, :, 1:] - t[:, :, :, :-1]
        mx = m[:, :, :, 1:] * m[:, :, :, :-1]

        # y方向
        pdy = p[:, :, 1:, :] - p[:, :, :-1, :]
        tdy = t[:, :, 1:, :] - t[:, :, :-1, :]
        my = m[:, :, 1:, :] * m[:, :, :-1, :]

        if self.use_smooth_l1:
            ex = F.smooth_l1_loss(pdx, tdx, reduction='none')
            ey = F.smooth_l1_loss(pdy, tdy, reduction='none')
        else:
            ex = (pdx - tdx).abs()
            ey = (pdy - tdy).abs()

        lx = safe_masked_mean(ex, mx, eps=self.eps)
        ly = safe_masked_mean(ey, my, eps=self.eps)
        return lx + ly

    def forward(self, pred_aligned, target, mask):
        total = pred_aligned.sum() * 0.0
        used = 0

        H, W = pred_aligned.shape[-2:]
        for s in self.scales:
            if H // s < 2 or W // s < 2:
                continue

            if s == 1:
                p, t, m = pred_aligned, target, mask
            else:
                # 用步长抽样做多尺度
                p = pred_aligned[:, :, ::s, ::s]
                t = target[:, :, ::s, ::s]
                m = mask[:, :, ::s, ::s]

            total = total + self._grad_loss_once(p, t, m)
            used += 1

        if used == 0:
            return pred_aligned.sum() * 0.0
        return total / used


class DAv2ScaleShiftInvariantLoss(nn.Module):
    """Scale-and-shift invariant loss in normalized relative-depth space."""

    def __init__(self, min_depth=1e-3, max_depth=1.0, eps=1e-6, trim=0.1):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.eps = eps
        self.trim = trim

    def forward(self, pred, target, mask, return_aligned=False):
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)

        mask = mask.float()
        finite = torch.isfinite(pred) & torch.isfinite(target)
        mask = mask * finite.float()

        scale, shift = solve_scale_shift(pred, target, mask, eps=self.eps)
        pred_aligned = scale * pred + shift

        residual = pred_aligned - target
        loss = masked_trimmed_mean(residual.abs(), mask, trim=self.trim, eps=self.eps)

        if return_aligned:
            return loss, pred_aligned, target, mask
        return loss


class DAv2GradientMatchingLoss(nn.Module):
    def __init__(self, scales=(1, 2, 4, 8), eps=1e-6):
        super().__init__()
        self.scales = scales
        self.eps = eps

    def _grad_loss_once(self, residual, mask):
        dx = residual[:, :, :, 1:] - residual[:, :, :, :-1]
        mx = mask[:, :, :, 1:] * mask[:, :, :, :-1]

        dy = residual[:, :, 1:, :] - residual[:, :, :-1, :]
        my = mask[:, :, 1:, :] * mask[:, :, :-1, :]

        return safe_masked_mean(dx.abs(), mx, eps=self.eps) + safe_masked_mean(dy.abs(), my, eps=self.eps)

    def forward(self, pred_aligned, target, mask):
        residual = pred_aligned - target
        total = zero_loss_like(residual)
        used = 0

        H, W = residual.shape[-2:]
        for s in self.scales:
            if H // s < 2 or W // s < 2:
                continue

            if s == 1:
                r, m = residual, mask
            else:
                r = residual[:, :, ::s, ::s]
                m = mask[:, :, ::s, ::s]

            total = total + self._grad_loss_once(r, m)
            used += 1

        if used == 0:
            return zero_loss_like(residual)
        return total / used



# -------------------- 新的总损失 --------------------
class _LegacyAincradLoss(nn.Module):
    """
    两阶段推荐：
    stage='coarse': 先学几何，不强推边缘
    stage='edge':   再学边界，仍然约束非边界平滑
    """
    def __init__(self, stage='coarse',
                 min_depth=1e-3, max_depth=50.0):
        super().__init__()

        self.stage = stage
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.eps = 1e-6
        self.gm_scales = (1, 2, 4, 8)

        self.loss_l1 = MaskedL1Loss(min_depth=min_depth, max_depth=max_depth)
        self.loss_silog = SafeSILogLoss()
        self.loss_low = LowFreqLoss()

        self.edge_builder = EdgeBandBuilder(q=0.88, dilate=1)
        self.loss_edge = EdgeBandGradLoss()
        self.loss_nonedge = NonEdgeSmoothLoss()

        self.loss_chamfer = BinsChamferLoss()


        dav2_min_depth = max(float(min_depth), 0.1)
        self.loss_ssi = DAv2ScaleShiftInvariantLoss(
            min_depth=dav2_min_depth,
            max_depth=max_depth,
            eps=self.eps,
            trim=0.1,
        )
        self.loss_gm = DAv2GradientMatchingLoss(scales=self.gm_scales, eps=self.eps)

        # 默认权重
        self.set_stage(stage)

    def set_stage(self, stage='coarse'):
        self.stage = stage
        if stage == 'coarse':
            self.w_l1 = 1.0
            self.w_silog = 1.0
            self.w_low = 0.0
            self.w_nonedge = 0.0
            self.w_edge = 3.5
            self.w_chamfer = 0.0

            self.w_ssi = 0.0
            self.w_gm = 2.0
        elif stage == 'edge':
            self.w_l1 = 1.0
            self.w_silog = 1.0
            self.w_low = 0.0
            self.w_nonedge = 0.0
            self.w_edge = 3.5
            self.w_chamfer = 0.4
            
            self.w_ssi = 0.0
            self.w_gm = 2.0
        else:
            raise ValueError("stage must be 'coarse' or 'edge'")

    def forward(self, pred_depth, bin_edges, gt_depth, valid_mask=None):
        # 现已改为从训练循环中给loss赋mask
        #valid_mask = ((gt_depth > self.min_depth) & (gt_depth < self.max_depth)).float()
        # s_valid_mask = ((s_depth > self.min_depth) & (s_depth < self.max_depth)).float()

        if valid_mask is None:
            valid_mask = ((gt_depth > self.min_depth) & (gt_depth < self.max_depth)).float()
        else:
            valid_mask = valid_mask.float()

        l1 = self.loss_l1(pred_depth, gt_depth, valid_mask) if self.w_l1 > 0 else zero_loss_like(pred_depth)
        silog = self.loss_silog(pred_depth, gt_depth, valid_mask) if self.w_silog > 0 else zero_loss_like(pred_depth)
        low = self.loss_low(pred_depth, gt_depth, valid_mask) if self.w_low > 0 else zero_loss_like(pred_depth)

        if self.w_nonedge > 0 or self.w_edge > 0:
            edge_band, safe_valid = self.edge_builder(gt_depth, valid_mask)
            nonedge = self.loss_nonedge(pred_depth, edge_band, safe_valid) if self.w_nonedge > 0 else zero_loss_like(pred_depth)
        else:
            edge_band = torch.zeros_like(gt_depth)
            nonedge = zero_loss_like(pred_depth)
        # 新增的anydepth的loss
        l_ssi, pred_aligned_inv, target_inv, dav2_mask = self.loss_ssi(
            pred_depth, gt_depth, valid_mask, return_aligned=True
        )
        l_gm = self.loss_gm(pred_aligned_inv, target_inv, dav2_mask)
        
        if self.w_edge > 0:
            edge = self.loss_edge(pred_depth, gt_depth, edge_band)
        else:
            edge = torch.tensor(0.0, device=gt_depth.device)

        if (self.w_chamfer > 0) and (bin_edges is not None):
            chamfer = self.loss_chamfer(bin_edges, gt_depth)
        else:
            chamfer = torch.tensor(0.0, device=gt_depth.device)

        total = self.w_l1 * l1 + \
                self.w_silog * silog + \
                self.w_low * low + \
                self.w_nonedge * nonedge + \
                self.w_edge * edge + \
                self.w_chamfer * chamfer + \
                self.w_ssi * l_ssi + \
                self.w_gm * l_gm

        return total, {
            "MAE": float(l1.detach()),
            "silog": float(silog.detach()),
            "low": float(low.detach()),
            "nonedge": float(nonedge.detach()),
            "edge": float(edge.detach()),
            "chamfer": float(chamfer.detach()),
            "ssi": float(l_ssi.detach()),
            "gm": float(l_gm.detach()),
            "total": float(total.detach())
        }


def gradient_weight_for_epoch(epoch):
    """Return lambda_g for a zero-based epoch index."""
    return 0.2 if epoch < 2 else 2.0


class NearDepthLoss(nn.Module):
    """SSI + multi-scale residual-gradient loss on relative depth."""

    def __init__(self, rho_min=DEFAULT_RHO_MIN, lambda_grad=0.2, eps=1e-6,
                 trim=0.1, gm_scales=(1, 2, 4, 8)):
        super().__init__()
        if not 0.0 < rho_min < 1.0:
            raise ValueError(f"rho_min must be in (0, 1), got {rho_min}")
        self.rho_min = float(rho_min)
        self.eps = eps
        self.lambda_grad = float(lambda_grad)
        self.loss_ssi = DAv2ScaleShiftInvariantLoss(
            min_depth=self.rho_min,
            max_depth=1.0,
            eps=eps,
            trim=trim,
        )
        self.loss_gm = DAv2GradientMatchingLoss(
            scales=gm_scales, eps=eps
        )

    def set_epoch(self, epoch):
        self.lambda_grad = gradient_weight_for_epoch(epoch)

    def set_stage(self, stage='coarse'):
        """Compatibility shim for callers using the old stage vocabulary."""
        if stage == 'coarse':
            self.lambda_grad = 0.2
        elif stage == 'edge':
            self.lambda_grad = 2.0
        else:
            raise ValueError("stage must be 'coarse' or 'edge'")

    def forward(self, pred_depth, bin_edges, gt_depth, valid_mask=None):
        del bin_edges
        if valid_mask is None:
            valid_mask = gt_depth > 0.0
        valid_mask = valid_mask.float()
        finite = torch.isfinite(pred_depth) & torch.isfinite(gt_depth)
        valid_mask = valid_mask * finite.float()

        l_ssi, pred_aligned, target, loss_mask = self.loss_ssi(
            pred_depth, gt_depth, valid_mask, return_aligned=True
        )
        l_gm = self.loss_gm(pred_aligned, target, loss_mask)
        total = l_ssi + self.lambda_grad * l_gm

        return total, {
            "ssi": float(l_ssi.detach()),
            "gm": float(l_gm.detach()),
            "lambda_grad": self.lambda_grad,
            "total": float(total.detach()),
        }


# Backward-compatible import only. New code must use ``NearDepthLoss``.
AincradLoss = NearDepthLoss
        





