"""Versioned affine alignment in z-depth; sums are accumulated in float64."""
import csv
import json
from pathlib import Path
import numpy as np


def evaluate_image(prediction, gt, mask, alignment="positive_scale_shift", min_depth=1e-3,
                   max_depth=3.0):
    prediction, gt = np.asarray(prediction, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if prediction.shape != gt.shape or mask.shape != gt.shape:
        raise ValueError("prediction, GT and mask shapes must match")
    if alignment not in {"none", "scale", "scale_shift", "positive_scale_shift"}:
        raise ValueError("unknown alignment")
    if not 0 <= min_depth < max_depth or not np.isfinite(max_depth):
        raise ValueError("invalid evaluation range")
    valid = mask & np.isfinite(gt) & (gt > min_depth) & (gt < max_depth)
    n = int(valid.sum())
    if not n:
        return {"status": "no_valid_gt", "n": 0}
    p, g = prediction[valid], gt[valid]
    if not np.isfinite(p).all():
        raise ValueError("non-finite prediction on valid GT; refusing to hide model failures")
    scale, shift, degenerate = 1., 0., False
    unconstrained_scale, scale_clamped = 1., False
    if alignment in {"scale_shift", "positive_scale_shift"}:
        pc, gc = p - p.mean(), g - g.mean()
        denominator = float(pc @ pc)
        degenerate = denominator <= 1e-12 * max(1., float(p @ p))
        unconstrained_scale = 0. if degenerate else float(pc @ gc) / denominator
        scale_clamped = alignment == "positive_scale_shift" and unconstrained_scale < 0
        scale = max(0., unconstrained_scale) if alignment == "positive_scale_shift" else unconstrained_scale
        shift = float(g.mean() - scale * p.mean())
    elif alignment == "scale":
        denominator = float(p @ p)
        degenerate = denominator <= 1e-12
        scale = 0. if degenerate else float(p @ g) / denominator
        unconstrained_scale = scale
    p = np.clip(scale * p + shift, max(min_depth, 1e-6), max_depth)
    error = p - g
    ratio = np.maximum(p / g, g / p)
    row = {"status": "ok", "n": n, "scale": scale, "shift": shift,
           "degenerate_alignment": degenerate,
           # Keep the unconstrained sign visible even when the applied scale is 0.
           "negative_scale": unconstrained_scale < 0,
           "unconstrained_scale": unconstrained_scale, "scale_clamped": scale_clamped,
           "abs_rel_sum": float(np.sum(np.abs(error) / g)),
           "sq_err_sum": float(error @ error)}
    row.update({f"d{k}_sum": int(np.sum(ratio < 1.25 ** k)) for k in (1, 2, 3)})
    row.update({"AbsRel": row["abs_rel_sum"] / n, "RMSE": np.sqrt(row["sq_err_sum"] / n)})
    row.update({f"Delta{k}": row[f"d{k}_sum"] / n for k in (1, 2, 3)})
    return row


def summarize(rows):
    valid = [r for r in rows if r["status"] == "ok"]
    if not valid:
        raise ValueError("no evaluable images")
    n = sum(r["n"] for r in valid)
    macro = {k: float(np.mean([r[k] for r in valid]))
             for k in ("AbsRel", "RMSE", "Delta1", "Delta2", "Delta3")}
    micro = {"AbsRel": sum(r["abs_rel_sum"] for r in valid) / n,
             "RMSE": float(np.sqrt(sum(r["sq_err_sum"] for r in valid) / n))}
    micro.update({f"Delta{k}": sum(r[f"d{k}_sum"] for r in valid) / n for k in (1, 2, 3)})
    return {"images": len(valid), "skipped_images": len(rows) - len(valid), "pixels": n,
            "negative_scale_images": sum(r["negative_scale"] for r in valid),
            "clamped_scale_images": sum(r.get("scale_clamped", False) for r in valid),
            "degenerate_images": sum(r["degenerate_alignment"] for r in valid),
            "macro": macro, "micro": micro}


def save_results(rows, directory, protocol):
    summary = summarize(rows)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (directory / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (directory / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"protocol": protocol, **summary}, handle, indent=2, allow_nan=False)
    return summary
