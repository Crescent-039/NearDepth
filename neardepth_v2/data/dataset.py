"""Manifest-based loading with separate supervision and evaluation masks."""
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import load_manifest, resolve_path
from .geometry import to_axial_depth, validate_geometry


def normalize_depth(depth, mask, rho_min=1e-3):
    if not 0 < rho_min < 1:
        raise ValueError("rho_min must be in (0, 1)")
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > 0)
    relative = np.zeros_like(depth)
    if valid.any():
        values = depth[valid]
        span = float(values.max() - values.min())
        relative[valid] = rho_min if span <= 1e-6 else (
            rho_min + (1 - rho_min) * (values - values.min()) / span)
    return relative, valid


def read_depth(path, scale, key="dataset", channels="single"):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("depth scale must be positive and finite")
    path = Path(path)
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in {".h5", ".hdf5"}:
        import h5py
        with h5py.File(path, "r") as handle:
            value = handle[key][:]
    else:
        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if value is None:
            raise ValueError(f"cannot read depth: {path}")
    if channels not in {"single", "replicated"}:
        raise ValueError("unknown depth channel policy")
    if channels == "replicated" and value.ndim == 3:
        if value.shape[2] != 3 or not np.allclose(value, value[..., :1], rtol=0, atol=1e-6, equal_nan=True):
            raise ValueError(f"depth channels are not replicated: {path}")
        value = value[..., 0]
    if value.ndim != 2:
        raise ValueError(f"expected single-channel depth, got {value.shape}: {path}")
    return value.astype(np.float32) * scale


class ManifestDepthDataset(Dataset):
    """Training resizes paired targets; evaluation keeps GT at native resolution.

    Required metadata: target_scale, metric_scale, min_depth, max_depth,
    depth_convention='z' or 'radial'. max_depth may be null (no upper bound).
    Radial inputs require per-scene camera rays and are converted at native
    resolution before any flip or resize. Stored source files are not changed.
    """
    def __init__(self, manifest, input_size=(448, 448), rho_min=1e-3,
                 training=False, horizontal_flip=True, data_root=None):
        self.manifest = Path(manifest).resolve()
        self.records = load_manifest(self.manifest)
        self.input_size = tuple(input_size)
        if len(self.input_size) != 2 or min(self.input_size) <= 0:
            raise ValueError("input_size must contain positive height and width")
        self.rho_min = rho_min
        self.training = training
        self.horizontal_flip = horizontal_flip
        self.data_root = data_root
        for record in self.records:
            m = record.metadata
            if not {"target_scale", "metric_scale", "min_depth", "max_depth"} <= m.keys():
                raise ValueError(f"missing unit/range metadata: {record.sample_id}")
            validate_geometry(m)
            if training and record.split != "train":
                raise ValueError("training loader requires train records")
            if record.supervision_source == "pseudo_label" and not record.metric_depth_path:
                raise ValueError("pseudo labels require an independent metric_depth_path")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        m = r.metadata
        path = lambda item: resolve_path(self.manifest, item, self.data_root)
        rgb = cv2.imread(str(path(r.rgb_path)))
        if rgb is None:
            raise ValueError(f"cannot read RGB: {r.sample_id}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        target = read_depth(path(r.target_depth_path), m["target_scale"],
                            m.get("hdf5_key", "dataset"), m.get("depth_channels", "single"))
        target = to_axial_depth(target, m)
        metric_path = r.metric_depth_path or r.target_depth_path
        metric = target.copy() if metric_path == r.target_depth_path and m["metric_scale"] == m["target_scale"] else to_axial_depth(read_depth(
            path(metric_path), m["metric_scale"], m.get("hdf5_key", "dataset"), m.get("depth_channels", "single")), m)
        # Refuse unverified registration rather than silently resizing GT onto RGB.
        registration = m.get("registration", "pixel_aligned")
        if registration not in {"pixel_aligned", "same_fov"}:
            raise ValueError("unknown registration policy")
        for depth in (target, metric):
            if depth.shape != rgb.shape[:2]:
                h, w = depth.shape
                rh, rw = rgb.shape[:2]
                if registration != "same_fov" or w*rh != rw*h:
                    raise ValueError(f"RGB/depth registration mismatch: {r.sample_id}")
        if self.training and self.horizontal_flip and torch.rand(()).item() < 0.5:
            rgb, target, metric = (np.flip(x, axis=1).copy() for x in (rgb, target, metric))
        height, width = self.input_size
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        if self.training:
            target = cv2.resize(target, (width, height), interpolation=cv2.INTER_NEAREST)
            metric = cv2.resize(metric, (width, height), interpolation=cv2.INTER_NEAREST)
        def clean(depth):
            upper = float("inf") if m["max_depth"] is None else m["max_depth"]
            mask = np.isfinite(depth) & (depth > max(0, m["min_depth"])) & (depth < upper)
            return np.where(mask, depth, 0).astype(np.float32), mask
        target, target_mask = clean(target)
        metric, metric_mask = clean(metric)
        relative, target_mask = normalize_depth(target, target_mask, self.rho_min)
        rgb = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255
        rgb = (rgb - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
        return {"rgb": rgb, "target_relative": torch.from_numpy(relative[None]),
                "target_mask": torch.from_numpy(target_mask[None]),
                "metric_depth": torch.from_numpy(metric[None]),
                "metric_mask": torch.from_numpy(metric_mask[None]),
                "sample_id": r.sample_id, "dataset": r.dataset}
