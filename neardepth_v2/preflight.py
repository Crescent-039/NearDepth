"""Validate a transferred training configuration without starting training."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np

from .config import load_experiment_config
from .data.dataset import ManifestDepthDataset
from .data.manifest import load_manifest, manifest_sha256, validate_disjoint


def inspect(config_path, samples_per_split=16, full=False, data_root=None):
    if samples_per_split < 1:
        raise ValueError('samples_per_split must be positive')
    config_path = Path(config_path).resolve()
    config = load_experiment_config(config_path)
    if data_root is not None:
        config = replace(config, runtime=replace(config.runtime, data_root=data_root))
    if not config.datasets:
        raise ValueError('no datasets configured')
    report = {'status': 'passed', 'full_decode': full, 'datasets': [], 'errors': [],
              'data_root': config.runtime.data_root,
              'effective_batch_size': config.optimization.batch_size*config.optimization.accumulation_steps}
    for dataset in config.datasets:
        splits = []
        for split, manifest in [('train', dataset.train_manifest), ('val', dataset.val_manifest)]:
            path = Path(manifest)
            if not path.is_absolute():
                path = config_path.parent/path
            records = load_manifest(path, check_files=full, data_root=config.runtime.data_root)
            if any(r.split != split or r.dataset != dataset.name for r in records):
                raise ValueError(f'{dataset.name}/{split}: wrong dataset/split records')
            splits.append(records)
            data = ManifestDepthDataset(path, (config.input_height, config.input_width), config.model.rho_min,
                                        training=split == 'train', horizontal_flip=False, data_root=config.runtime.data_root)
            indices = range(len(data)) if full else sorted(set(np.linspace(
                0, len(data)-1, min(samples_per_split, len(data)), dtype=int).tolist()))
            fractions = []
            for index in indices:
                try:
                    sample = data[index]
                    tf, mf = (float(sample[k].float().mean()) for k in ('target_mask', 'metric_mask'))
                    if min(tf, mf) <= 0:
                        raise ValueError('empty supervision/evaluation mask')
                    fractions.append((tf, mf))
                except Exception as exc:
                    report['errors'].append({'sample_id': records[index].sample_id, 'error': str(exc)})
            report['datasets'].append({'dataset': dataset.name, 'split': split, 'records': len(data),
                'scenes': len({r.scene_id for r in records}), 'checked': len(indices),
                'sha256': manifest_sha256(path),
                'min_target_valid_fraction': min((x[0] for x in fractions), default=0),
                'min_metric_valid_fraction': min((x[1] for x in fractions), default=0)})
        validate_disjoint(*splits)
    report['status'] = 'failed' if report['errors'] else 'passed'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples-per-split', type=int, default=16)
    parser.add_argument('--full', action='store_true', help='decode/check every record, not just a sample')
    parser.add_argument('--data-root', help='override destination root for local/remote data checks')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('choose a new preflight report path')
    report = inspect(args.config, args.samples_per_split, args.full, args.data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if report['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
