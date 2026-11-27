"""Single training entry for V2 SDF removals and Simple Fusion pilots."""
import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .config import load_experiment_config, save_experiment_config
from .data.dataset import ManifestDepthDataset
from .data.manifest import manifest_sha256, validate_disjoint
from .data.sampling import make_training_sampler
from .engine.loss import RelativeDepthLoss
from .engine.checkpoint import atomic_save, capture_rng, restore_rng, validate_resume, run_lease
from .evaluation.runner import evaluate_dataset
from .evaluation.metrics import save_results
from .models.model import build_model, module_state_sha256
from .reproducibility import seed_everything, make_dataloader_generator, seed_dataloader_worker


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--backbone-repo", required=True)
    parser.add_argument("--backbone-weights", required=True)
    parser.add_argument("--resume", help="trusted latest.pth; restart at next completed epoch boundary")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def run(args):
    config = load_experiment_config(args.config)
    if not config.datasets:
        raise ValueError("add real train/val manifests to the config first")
    seed_everything(config.runtime.seed, config.runtime.deterministic)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if (config.runtime.require_cuda or device.type == "cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA required but unavailable")
    if config.runtime.require_cuda and device.type != "cuda":
        raise RuntimeError("this config requires CUDA")
    precision = config.optimization.precision
    if precision != "fp32" and device.type != "cuda":
        raise ValueError("fp16/bf16 training requires CUDA; use fp32 for CPU contract tests")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support BF16")
    autocast = lambda: (torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)
                       if precision != "fp32" else nullcontext())
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    size = (config.input_height, config.input_width)
    config_dir = Path(args.config).resolve().parent
    resolve = lambda p: Path(p).resolve() if Path(p).is_absolute() else (config_dir / p).resolve()
    training, validation, hashes = [], [], {}
    for d in config.datasets:
        train_path, val_path = resolve(d.train_manifest), resolve(d.val_manifest)
        train_data = ManifestDepthDataset(train_path, size, config.model.rho_min, training=True, data_root=config.runtime.data_root)
        val_data = ManifestDepthDataset(val_path, size, config.model.rho_min, data_root=config.runtime.data_root)
        if any(r.split != "val" or r.dataset != d.name for r in val_data.records):
            raise ValueError(f"validation manifest does not match {d.name}/val")
        if any(r.dataset != d.name for r in train_data.records):
            raise ValueError(f"training manifest does not match {d.name}")
        validate_disjoint(train_data.records, val_data.records)
        training.append(train_data)
        evaluation = config.evaluation if d.eval_min_depth is None else replace(
            config.evaluation, min_depth=d.eval_min_depth, max_depth=d.eval_max_depth)
        evaluation.validate()
        for r in val_data.records:
            bounds = r.metadata
            if bounds["min_depth"] > evaluation.min_depth or (bounds["max_depth"] is not None and bounds["max_depth"] < evaluation.max_depth):
                raise ValueError(f"manifest validity range is narrower than evaluation: {d.name}")
        validation.append((d.name, val_data, evaluation))
        hashes[d.name] = {"train": manifest_sha256(train_path), "val": manifest_sha256(val_path)}
    model = build_model(config.model, args.backbone_repo, args.backbone_weights).to(device)
    initialization = {"common_init_seed": config.model.common_init_seed,
                      "backbone_state_sha256": module_state_sha256(model.backbone),
                      "head_state_sha256": module_state_sha256(model.head),
                      "decoder": getattr(model, "decoder_initialization", None)}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": config.optimization.backbone_lr},
        {"params": list(model.decoder.parameters()) + list(model.head.parameters()), "lr": config.optimization.decoder_lr}],
        weight_decay=config.optimization.weight_decay)
    generator = make_dataloader_generator(config.runtime.seed)
    sampler = make_training_sampler(training, [d.sampling_weight for d in config.datasets], config.sampling, generator)
    loader = DataLoader(ConcatDataset(training), batch_size=config.optimization.batch_size,
                        shuffle=sampler is None, sampler=sampler, num_workers=config.optimization.num_workers,
                        generator=generator, pin_memory=device.type == "cuda",
                        worker_init_fn=seed_dataloader_worker)
    output = Path(config.runtime.output_dir) / config.experiment_name / f"seed_{config.runtime.seed}"
    if args.resume:
        if Path(args.resume).name != "latest.pth":
            raise ValueError("resume only from latest.pth; best.pth is for evaluation")
        output = Path(args.resume).resolve().parents[1]
        if not (output / "config.json").is_file():
            raise ValueError("resume checkpoint must be inside a run/checkpoints directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "checkpoints").mkdir()
        save_experiment_config(config, output / "config.json")
        (output / "data_manifest_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    def git(*parts):
        try:
            result = subprocess.run(["git", *parts], capture_output=True, text=True)
        except OSError:
            return "unavailable"
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    # Record source bytes, including untracked V2 files which are not in git diff.
    source = Path(__file__).resolve().parent
    source_hashes = {p.relative_to(source).as_posix(): manifest_sha256(p) for p in sorted(source.rglob("*.py"))}
    for relative in (() if args.resume else source_hashes):
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    for dataset in ([] if args.resume else training + [item[1] for item in validation]):
        name = dataset.records[0].dataset + "_" + dataset.records[0].split
        destination = output / "manifests" / (name + ".jsonl")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Archive paths as absolute paths so the copied manifest still resolves.
        from .data.manifest import save_manifest, resolve_path
        archive_path = lambda item: item if item.startswith("data://") else str(resolve_path(dataset.manifest, item))
        records = [replace(r, rgb_path=archive_path(r.rgb_path),
                           target_depth_path=archive_path(r.target_depth_path),
                           metric_depth_path=archive_path(r.metric_depth_path) if r.metric_depth_path else None)
                   for r in dataset.records]
        save_manifest(records, destination)
    environment = {"python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(),
                   "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
                   "source_sha256": source_hashes,
                   "backbone_weights_sha256": manifest_sha256(args.backbone_weights),
                   "initialization": initialization,
                   "sampling": {**asdict(config.sampling), "replacement": sampler is not None,
                                "dataset_weights": {d.name: d.sampling_weight for d in config.datasets},
                                "dataset_lengths": {d.name: len(data) for d, data in zip(config.datasets, training)}},
                   "loss": {**asdict(config.loss), "trim_reduction": "per_image", "gradient_scales": [1, 2, 4, 8],
                            "direction": "normalized_z_depth_larger_is_farther",
                            "gradient_weight": {"epochs_0_1": .2, "later": 2.}},
                   "checkpoint_selection": "equal_dataset_mean_macro_AbsRel",
                   "augmentation": "resize_and_random_horizontal_flip", "scheduler": "none",
                   "precision": precision, "accumulation_steps": config.optimization.accumulation_steps,
                   "resume_boundary": "completed_epoch_only", "cuda": torch.version.cuda,
                   "effective_batch_size": config.optimization.batch_size*config.optimization.accumulation_steps}
    criterion = RelativeDepthLoss(**asdict(config.loss))
    best, start_epoch, global_step, history = float("inf"), 0, 0, []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        validate_resume(checkpoint, config, hashes, source_hashes, environment["backbone_weights_sha256"])
        if checkpoint["training_environment"] != {"torch": str(torch.__version__), "device": str(device),
                                                 "gpu": environment["gpu"]}:
            raise ValueError("resume requires the recorded torch/device/GPU environment")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"], generator)
        best, start_epoch = checkpoint["best_score"], checkpoint["epoch"]+1
        history, global_step = checkpoint["history"], checkpoint["global_step"]
        if checkpoint["selection_score"] == best:
            atomic_save(checkpoint, output / "checkpoints" / "best.pth")
        (output / f"resume_epoch_{start_epoch:03d}.json").write_text(json.dumps(
            {"config": config.to_dict(), "environment": environment}, indent=2), encoding="utf-8")
    else:
        (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    with (output / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        component_names = ("ssi", "gradient", "anchor", "negative_scale_fraction")
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_macro_absrel"] +
                                ["train_"+name for name in component_names])
        writer.writeheader()
        writer.writerows(history)
        handle.flush()
        for epoch in range(start_epoch, config.optimization.epochs):
            model.train()
            total, samples = 0., 0
            component_sums = {name: 0. for name in component_names}
            dataset_counts = {d.name: 0 for d in config.datasets}
            sample_stream = hashlib.sha256()
            epoch_started, group_samples = time.perf_counter(), 0
            optimizer.zero_grad(set_to_none=True)
            batches = min(len(loader), config.optimization.max_train_batches or len(loader))
            for batch_index, batch in enumerate(loader):
                if not batch["target_mask"].flatten(1).any(dim=1).all():
                    raise ValueError("sample has no valid supervision; run preflight before training")
                with autocast():
                    prediction = model(batch["rgb"].to(device, non_blocking=True))["prediction"]
                loss, components = criterion(prediction.float(), batch["target_relative"].to(device), batch["target_mask"].to(device), epoch)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss")
                count = batch["rgb"].shape[0]
                for dataset_name, sample_id in zip(batch["dataset"], batch["sample_id"]):
                    dataset_counts[dataset_name] += 1
                    sample_stream.update((dataset_name+":"+sample_id+"\n").encode())
                scaler.scale(loss*count).backward()
                group_samples += count
                if (batch_index+1) % config.optimization.accumulation_steps == 0 or batch_index+1 == batches:
                    scaler.unscale_(optimizer)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(group_samples)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimization.grad_clip_norm or float("inf"))
                    if not torch.isfinite(norm):
                        raise ValueError("non-finite gradient norm")
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    group_samples = 0
                    global_step += 1
                total += loss.detach().item()*count
                for name in component_names:
                    component_sums[name] += components[name]*count
                samples += count
                if (batch_index+1) % config.runtime.log_every == 0 or batch_index+1 == batches:
                    progress = {"epoch": epoch, "batch": batch_index+1, "batches": batches,
                                "global_step": global_step, "train_loss": total/samples,
                                **{"train_"+name: value/samples for name, value in component_sums.items()},
                                "dataset_counts": dataset_counts,
                                "elapsed_seconds": time.perf_counter()-epoch_started}
                    (output / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
                    print(json.dumps(progress), flush=True)
                if batch_index+1 >= batches:
                    break
            sampling_report = {"epoch": epoch, "samples": samples, "dataset_counts": dataset_counts,
                               "sample_stream_sha256": sample_stream.hexdigest(),
                               "train_seconds": time.perf_counter()-epoch_started}
            (output/"sampling").mkdir(exist_ok=True)
            (output/"sampling"/f"epoch_{epoch:03d}.json").write_text(json.dumps(sampling_report, indent=2), encoding="utf-8")
            scores = []
            for name, data, evaluation in validation:
                rows = evaluate_dataset(model, data, device, evaluation)
                summary = save_results(rows, output / "validation" / f"epoch_{epoch:03d}" / name, asdict(evaluation))
                scores.append(summary["macro"]["AbsRel"])
            score = sum(scores)/len(scores)
            if not math.isfinite(score):
                raise ValueError("non-finite validation selection score")
            improved = score < best
            best = min(best, score)
            row = {"epoch": epoch, "train_loss": total/samples, "val_macro_absrel": score}
            row.update({"train_"+name: value/samples for name, value in component_sums.items()})
            history.append(row)
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "epoch": epoch, "config": config.to_dict(), "manifest_hashes": hashes,
                          "selection_score": score, "best_score": best, "format_version": 2,
                          "source_sha256": source_hashes, "backbone_weights_sha256": environment["backbone_weights_sha256"],
                          "rng": capture_rng(generator), "scaler": scaler.state_dict(),
                          "global_step": global_step, "history": history,
                          "initialization": initialization, "epoch_sampling": sampling_report,
                          "training_environment": {"torch": str(torch.__version__), "device": str(device), "gpu": environment["gpu"]}}
            atomic_save(checkpoint, output / "checkpoints" / "latest.pth")
            if improved:
                atomic_save(checkpoint, output / "checkpoints" / "best.pth")
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)


def main():
    args = parse_args()
    config = load_experiment_config(args.config)
    output = (Path(args.resume).resolve().parents[1] if args.resume else
              Path(config.runtime.output_dir)/config.experiment_name/f"seed_{config.runtime.seed}")
    with run_lease(output):
        run(args)


if __name__ == "__main__":
    main()
