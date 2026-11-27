"""Freeze the 13 predefined ablations from one common data/training protocol."""
import argparse
from dataclasses import replace
from pathlib import Path
import os
import json
from neardepth_v2.config import load_experiment_config, save_experiment_config
from neardepth_v2.data.manifest import manifest_sha256
from neardepth_v2.ablations import SPECS, COMPARISONS, ablation_models


def generate(base, output, prefix='ablation_v1'):
    config = load_experiment_config(base)
    if not config.datasets:
        raise ValueError("fill in manifests before generating runnable experiment configs")
    base_dir = Path(base).resolve().parent
    output = Path(output).resolve()
    def relative(path):
        if os.name == 'nt' and path.startswith('/'):
            raise ValueError('generate Linux-target configs on the server; refusing to reinterpret /root as a Windows path')
        source = Path(path).resolve() if Path(path).is_absolute() else (base_dir / path).resolve()
        return os.path.relpath(source, output).replace("\\", "/")
    datasets = tuple(replace(d, train_manifest=relative(d.train_manifest), val_manifest=relative(d.val_manifest)) for d in config.datasets)
    variants = ablation_models(config.model, config.runtime.seed)
    output.mkdir(parents=True, exist_ok=False)
    entries = {}
    for name, model in variants.items():
        path = output/(name+'.json')
        value = replace(config, datasets=datasets, experiment_name=prefix+'_'+name, model=model)
        save_experiment_config(value, path)
        entries[name] = {'group': SPECS[name][0], 'meaning': SPECS[name][1],
                         'config': path.name, 'sha256': manifest_sha256(path), 'model': value.to_dict()['model']}
    matrix = {'version': 'ablation_matrix_v1', 'base_config_sha256': manifest_sha256(base),
              'variants': entries, 'comparisons': COMPARISONS,
              'initialization_policy': 'same backbone/head and canonical full-SDF retained tensors; explicit fusion input slices for single branches',
              'budget': {'optimization': config.to_dict()['optimization'], 'sampling': config.to_dict()['sampling']},
              'note': 'Inherits the supplied base budget. Generation does not start training or approve a formal budget. This is a predefined matrix, not every possible combination.'}
    (output/'matrix.json').write_text(json.dumps(matrix, indent=2, ensure_ascii=False), encoding='utf-8')
    return matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--prefix', default='ablation_v1')
    args = p.parse_args()
    print(json.dumps(generate(args.base, args.output, args.prefix), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
