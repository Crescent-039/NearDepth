"""Build scene-disjoint manifests from the project's three dataset layouts."""
import argparse
import json
import os
from pathlib import Path
import random
from .manifest import ManifestRecord, save_manifest, manifest_sha256, validate_disjoint


def discover(root, dataset):
    """Return (scene, RGB, sensor/rendered GT, optional pseudo-label) tuples."""
    root = Path(root)
    found = []
    if dataset == "graspnet":
        rgb_files = list(root.glob("train/scene_*/realsense/rgb/*.png"))
        for part in range(1, 5):
            rgb_files.extend(root.glob(f"train_{part}/scene_*/realsense/rgb/*.png"))
        seen = set()
        for rgb in sorted(rgb_files):
            scene = rgb.parents[2].name
            frame_key = (scene, rgb.name)
            if frame_key in seen:
                raise ValueError(f"duplicate GraspNet frame across packages: {frame_key}")
            seen.add(frame_key)
            # Preserve the existing official train pool, never include test scenes.
            if not 0 <= int(scene.split("_")[-1]) < 100:
                continue
            camera = rgb.parents[1]
            found.append((scene, rgb, camera / "depth" / rgb.name,
                          camera / "synthetic_depth" / rgb.name))
    elif dataset == "dreds":
        for rgb in sorted(list(root.glob("train_part*/part*/*/*_color.jpg")) + list(root.glob("train_part*/part*/*/*_color.png"))):
            found.append((rgb.parent.relative_to(root).as_posix(), rgb,
                          rgb.with_name(rgb.stem.removesuffix("_color") + "_depth_120.exr"), None))
    elif dataset == "hypersim":
        for rgb in sorted(root.glob("ai_*/images/scene_cam_*_final_preview/frame.*.color.jpg")):
            camera = rgb.parent.name.removesuffix("_final_preview")
            gt = rgb.parent.parent / (camera + "_geometry_hdf5") / rgb.name.replace(".color.jpg", ".depth_meters.hdf5")
            found.append((rgb.parents[2].name, rgb, gt, None))
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    if not found:
        raise ValueError(f"no samples found for {dataset} under {root}")
    return found


def build(root, dataset, output, seed=42, val_fraction=.1, train_limit=None,
          val_limit=None, min_depth=.001, max_depth=3., target_scale=1.,
          metric_scale=1., depth_convention="z", portable_root=None,
          depth_channels="single", registration="pixel_aligned"):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0,1)")
    if min_depth < 0 or (max_depth is not None and max_depth <= min_depth):
        raise ValueError("invalid depth range")
    if target_scale <= 0 or metric_scale <= 0:
        raise ValueError("unit scales must be positive")
    if depth_convention != "z":
        raise ValueError("convert radial distance to camera-axis z before building a training manifest")
    if depth_channels not in {"single", "replicated"} or registration not in {"pixel_aligned", "same_fov"}:
        raise ValueError("invalid depth channel/registration policy")
    root, output = Path(root).absolute(), Path(output).resolve()
    portable_root = Path(portable_root).absolute() if portable_root else None
    found = discover(root, dataset)
    scenes = sorted({x[0] for x in found})
    if len(scenes) < 2:
        raise ValueError("at least two scenes required")
    if dataset == "graspnet":
        # Retain the established 0..89 / 90..99 split.
        validation = {s for s in scenes if int(s.split("_")[-1]) >= 90}
    else:
        shuffled = scenes.copy()
        random.Random(seed).shuffle(shuffled)
        validation = set(shuffled[:max(1, min(len(scenes) - 1, round(len(scenes) * val_fraction)))])
    records = {"train": [], "val": []}
    missing = []
    for scene, rgb, gt, pseudo in found:
        split = "val" if scene in validation else "train"
        target = pseudo if pseudo is not None and split == "train" else gt
        if not gt.is_file() or not target.is_file():
            missing.append(str(rgb))
            continue
        rel = lambda p: ("data://" + p.relative_to(portable_root).as_posix()) if portable_root else os.path.relpath(p, output).replace("\\", "/")
        records[split].append(ManifestRecord(
            sample_id=dataset + "/" + rgb.relative_to(root).as_posix(),
            dataset=dataset, split=split, scene_id=scene, rgb_path=rel(rgb),
            target_depth_path=rel(target), metric_depth_path=rel(gt),
            supervision_source="pseudo_label" if target == pseudo else (
                "sensor_gt" if dataset == "graspnet" else "rendered_gt"),
            metadata={"target_scale": target_scale if target == pseudo else metric_scale,
                      "metric_scale": metric_scale, "min_depth": min_depth,
                      "max_depth": max_depth, "depth_convention": depth_convention,
                      "depth_channels": depth_channels, "registration": registration}))
    if missing:
        raise FileNotFoundError(f"{len(missing)} RGB frames missing required depths; first: {missing[0]}")
    for split, limit in (("train", train_limit), ("val", val_limit)):
        if not records[split]:
            raise ValueError(f"no {split} records; check scene layout")
        if limit is not None:
            if limit <= 0 or limit > len(records[split]):
                raise ValueError(f"invalid {split} limit {limit}, available {len(records[split])}")
            records[split] = random.Random(seed + (split == "val")).sample(records[split], limit)
    validate_disjoint(records["train"], records["val"])
    if output.exists():
        raise FileExistsError(f"choose a new manifest directory: {output}")
    output.mkdir(parents=True)
    report = {"dataset": dataset, "seed": seed, "root": str(root),
              "split_method": "scene_id_90" if dataset == "graspnet" else "seeded_scene_shuffle",
              "validation_scenes": sorted(validation),
              "training_scenes": sorted(set(scenes) - validation), "splits": {}}
    for split in records:
        path = output / f"{split}.jsonl"
        save_manifest(records[split], path)
        report["splits"][split] = {"samples": len(records[split]), "sha256": manifest_sha256(path)}
    (output / "manifest_info.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["graspnet", "dreds", "hypersim"])
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=.1)
    p.add_argument("--train-limit", type=int)
    p.add_argument("--val-limit", type=int)
    p.add_argument("--min-depth", type=float, default=.001)
    p.add_argument("--max-depth", type=float, required=True)
    p.add_argument("--target-scale", type=float, required=True, help="stored target units to meters")
    p.add_argument("--metric-scale", type=float, required=True, help="stored GT units to meters")
    p.add_argument("--depth-convention", required=True, choices=["z", "radial"],
                   help="explicitly confirm geometry; radial inputs currently rejected")
    p.add_argument("--portable-root", help="common datasets directory; emit data:// paths for transfer")
    p.add_argument("--depth-channels", choices=["single", "replicated"], default="single")
    p.add_argument("--registration", choices=["pixel_aligned", "same_fov"], default="pixel_aligned",
                   help="same_fov is an explicit registration assertion, not inferred from aspect ratio")
    print(json.dumps(build(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
