"""Evaluate a V2 checkpoint with the same metric implementation as training."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import torch
from .config import ExperimentConfig
from .data.dataset import ManifestDepthDataset
from .data.manifest import manifest_sha256
from .models.model import build_model
from .evaluation.runner import evaluate_dataset
from .evaluation.metrics import save_results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--backbone-repo", required=True)
    p.add_argument("--backbone-weights", required=True)
    p.add_argument("--min-depth", required=True, type=float)
    p.add_argument("--max-depth", required=True, type=float)
    p.add_argument("--data-root", help="dataset directory resolving data:// paths")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ExperimentConfig.from_dict(checkpoint["config"])
    evaluation = replace(config.evaluation, min_depth=args.min_depth, max_depth=args.max_depth)
    evaluation.validate()
    dataset = ManifestDepthDataset(args.manifest, (config.input_height, config.input_width), config.model.rho_min,
                                   data_root=args.data_root or config.runtime.data_root)
    if any(r.split == "train" for r in dataset.records):
        raise ValueError("evaluation requires val/test manifest")
    for r in dataset.records:
        m = r.metadata
        if m["min_depth"] > args.min_depth or (m["max_depth"] is not None and m["max_depth"] < args.max_depth):
            raise ValueError("manifest validity range narrower than requested evaluation")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    model = build_model(config.model, args.backbone_repo, args.backbone_weights).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = evaluate_dataset(model, dataset, device, evaluation)
    protocol = {**asdict(evaluation), "manifest_sha256": manifest_sha256(args.manifest),
                "data_root": args.data_root or config.runtime.data_root,
                "checkpoint_sha256": manifest_sha256(args.checkpoint)}
    print(save_results(rows, output, protocol))


if __name__ == "__main__":
    main()
