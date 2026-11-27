"""Bind the paper's 448x448 training protocol to local data and generate its ablations."""

import argparse
from dataclasses import replace
from pathlib import Path

from neardepth_v2.config import load_experiment_config, save_experiment_config
from neardepth_v2.configs.make_ablation_configs import generate


TEMPLATE = Path(__file__).resolve().parents[1] / "neardepth_v2/configs/paper_448.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True, type=Path, help="directory containing the six archived JSONL manifests")
    parser.add_argument("--data-root", required=True, type=Path, help="prepared datasets directory resolving data:// paths")
    parser.add_argument("--output", required=True, type=Path, help="new directory for base.json and variants/")
    parser.add_argument("--seed", type=int, help="override the paper configuration's training and model initialization seeds")
    args = parser.parse_args()

    config = load_experiment_config(TEMPLATE)
    seed = config.runtime.seed if args.seed is None else args.seed
    manifest_root = args.manifest_root.resolve()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    if not data_root.is_dir():
        parser.error(f"data directory does not exist: {data_root}")
    if manifest_root.anchor != output.anchor:
        parser.error("place --manifest-root and --output on the same drive; variant configs use relative manifest paths")

    datasets = []
    for dataset in config.datasets:
        train_manifest = manifest_root / dataset.train_manifest
        val_manifest = manifest_root / dataset.val_manifest
        for manifest in (train_manifest, val_manifest):
            if not manifest.is_file():
                parser.error(f"missing manifest: {manifest}")
        datasets.append(replace(dataset, train_manifest=train_manifest.as_posix(), val_manifest=val_manifest.as_posix()))

    config = replace(
        config,
        datasets=tuple(datasets),
        model=replace(config.model, common_init_seed=seed, decoder_init_seed=seed),
        runtime=replace(config.runtime, seed=seed, data_root=data_root.as_posix()),
    )
    config.validate()
    base = output / "base.json"
    variants = output / "variants"
    output.mkdir(parents=True, exist_ok=False)
    save_experiment_config(config, base)
    matrix = generate(base, variants, prefix="paper_448")
    print(f"Seed {seed}: generated {len(matrix['variants'])} configs in {variants}")


if __name__ == "__main__":
    main()
