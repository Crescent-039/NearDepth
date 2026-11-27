"""Resumable Depth Pro labels for extracted GraspNet train_1..4 packages.

Uses the local teacher implementation previously called by infer_folder.py.
PNG labels store millimeters. Depth units and inference settings are logged.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import numpy as np
from PIL import Image


def encode_depth(depth):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2 or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("teacher returned invalid/nonpositive depth")
    millimeters = np.rint(depth*1000)
    if np.any(millimeters > 65535):
        raise ValueError("depth exceeds uint16-mm range; keep source and report, never wrap/truncate")
    return np.maximum(millimeters, 1).astype(np.uint16)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="prepared datasets/GraspNet")
    p.add_argument("--teacher-src", required=True, help="teacher_depthpro/src")
    p.add_argument("--weights", required=True)
    p.add_argument("--reports", required=True)
    p.add_argument("--limit", type=int, help="small pilot before full run")
    p.add_argument("--focal-source", choices=["estimated", "camk"], default="estimated",
                   help="estimated preserves the old folder script policy for PNG inputs")
    args = p.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch required for batch labeling; current environment has no CUDA")
    root, reports = Path(args.root).resolve(), Path(args.reports).resolve()
    files = sorted(root.glob("train_*/scene_*/realsense/rgb/*.png"))
    if not files:
        raise FileNotFoundError("no prepared GraspNet RGB frames")
    weights = Path(args.weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    sys.path.insert(0, str(Path(args.teacher_src).resolve()))
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
    reports.mkdir(parents=True, exist_ok=True)
    settings = {"weights_sha256": digest(weights), "precision": "fp16", "focal_source": args.focal_source,
                "output": "uint16_png_mm", "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0), "source_root": str(root),
                "teacher_source_sha256": {f.relative_to(Path(args.teacher_src).resolve()).as_posix(): digest(f)
                    for f in sorted((Path(args.teacher_src).resolve()/"depth_pro").rglob("*.py"))}}
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    (reports/"settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    model, transform = depth_pro.create_model_and_transforms(
        config=replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(weights)),
        device=torch.device("cuda"), precision=torch.float16)
    model.eval()
    db = sqlite3.connect(reports/"labels.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS labels (path TEXT PRIMARY KEY, settings TEXT, rgb_sha TEXT, output_sha TEXT)")
    success, skipped, failed = 0, 0, 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        for rgb in files[:args.limit] if args.limit else files:
            output = rgb.parent.parent/"synthetic_depth"/(rgb.stem+".png")
            if not output.resolve().is_relative_to(root):
                raise ValueError("output escaped dataset root")
            key, input_hash = rgb.relative_to(root).as_posix(), digest(rgb)
            previous = db.execute("SELECT settings,rgb_sha,output_sha FROM labels WHERE path=?", (key,)).fetchone()
            if output.exists():
                if previous == (fingerprint, input_hash, digest(output)):
                    skipped += 1
                    continue
                raise FileExistsError(f"existing labels lack matching provenance: {output}")
            if shutil.disk_usage(root).free < 25*1024**3:
                raise OSError("below 25GiB reserve; pause labeling before disk migration")
            try:
                sample_started = time.perf_counter()
                image, _, _ = depth_pro.load_rgb(str(rgb))
                focal = None
                if args.focal_source == "camk":
                    intrinsics = np.load(rgb.parent.parent/"camK.npy", allow_pickle=False)
                    focal = torch.tensor(float(intrinsics[0, 0]), device="cuda")
                with torch.inference_mode():
                    result = model.infer(transform(image), f_px=focal)
                depth = result["depth"].float().cpu().numpy().squeeze()
                if depth.shape != image.shape[:2]:
                    raise ValueError("prediction/RGB dimensions differ")
                encoded = encode_depth(depth)
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix(".partial.png")
                Image.fromarray(encoded).save(temporary)
                with Image.open(temporary) as check:
                    if not np.array_equal(np.asarray(check), encoded):
                        raise ValueError("saved depth verification failed")
                temporary.rename(output)
                db.execute("INSERT INTO labels VALUES (?,?,?,?)", (key, fingerprint, input_hash, digest(output)))
                db.commit()
                success += 1
                with (reports/"samples.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"rgb": key, "shape": list(depth.shape),
                        "min_m": float(depth.min()), "max_m": float(depth.max()),
                        "positive_fraction": float((encoded > 0).mean()),
                        "seconds": time.perf_counter()-sample_started,
                        "peak_cuda_mib": torch.cuda.max_memory_allocated()/1024**2})+"\n")
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:
                failed += 1
                with (reports/"failures.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"rgb": key, "error": str(exc)})+"\n")
                if "uint16-mm range" in str(exc):
                    raise
            if (success+failed) % 50 == 0:
                status = {"success": success, "skipped": skipped, "failed": failed, "last_rgb": key}
                (reports/"progress.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
                print(json.dumps(status), flush=True)
        status = {"success": success, "skipped": skipped, "failed": failed, "time": time.time(),
                  "elapsed_seconds": time.perf_counter()-started,
                  "peak_cuda_mib": torch.cuda.max_memory_allocated()/1024**2}
        (reports/"last_run.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status), flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    main()
