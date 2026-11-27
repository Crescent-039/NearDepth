"""Prepare audited three-dataset manifests and a bounded GPU acceptance pair."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import random
import shutil

from .config import (DatasetConfig, ExperimentConfig, ModelConfig, OptimizationConfig,
                     RuntimeConfig, SamplingConfig, save_experiment_config)
from .data.build_manifest import discover
from .data.geometry import validate_geometry
from .data.manifest import ManifestRecord, manifest_sha256, save_manifest, validate_disjoint


def hypersim_metadata(camera_parameters, official_splits):
    cameras = {}
    with Path(camera_parameters).open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            name = row['scene_name']
            if name in cameras:
                raise ValueError('duplicate camera scene metadata')
            entry = {'camera_native_size': [int(float(row['settings_output_img_height'])),
                                             int(float(row['settings_output_img_width']))],
                     'camera_ray_matrix': [[float(row[f'M_cam_from_uv_{i}{j}']) for j in range(3)] for i in range(3)]}
            validate_geometry({'depth_convention': 'radial', **entry})
            cameras[name] = entry
    splits, scene_partitions = {}, {}
    with Path(official_splits).open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if row['included_in_public_release'] != 'True':
                continue
            key = (row['scene_name'], row['camera_name'], int(row['frame_id']))
            split = row['split_partition_name']
            if split not in {'train', 'val', 'test'} or key in splits:
                raise ValueError('invalid/duplicate public Hypersim split row')
            if row['scene_name'] in scene_partitions and scene_partitions[row['scene_name']] != split:
                raise ValueError('official Hypersim scene crosses partitions')
            scene_partitions[row['scene_name']] = split
            splits[key] = split
    return cameras, splits


def prepare(data_root, camera_parameters, official_splits, output, train_limit=64,
            val_limit=16, epoch_samples=96, runtime_data_root=None, output_dir='experiments_v2', seed=42):
    data_root, output = Path(data_root).absolute(), Path(output).absolute()
    if output.exists():
        raise FileExistsError('choose a new joint preparation directory')
    if min(train_limit, val_limit, epoch_samples) <= 0:
        raise ValueError('joint acceptance counts must be positive')
    cameras, official = hypersim_metadata(camera_parameters, official_splits)
    all_pools, reports = {}, {}
    roots = {'graspnet': 'GraspNet', 'dreds': 'DREDS', 'hypersim': 'Hypersim'}
    for name, directory in roots.items():
        root = data_root/directory
        found = discover(root, name)
        scenes = sorted({r[0] for r in found})
        if len(scenes) < 2:
            raise ValueError(f'{name}: at least two scenes needed')
        validation = set()
        if name == 'dreds':
            shuffled = scenes.copy(); random.Random(seed).shuffle(shuffled)
            validation = set(shuffled[:max(1, min(len(scenes)-1, round(len(scenes)*.1)))])
        pools = {'train': [], 'val': [], 'test': []}
        seen_hypersim = set()
        for scene, rgb, gt, pseudo in found:
            if name == 'hypersim':
                camera_name = rgb.parent.name.removeprefix('scene_').removesuffix('_final_preview')
                frame_id = int(rgb.name.split('.')[1])
                split = official.get((scene, camera_name, frame_id))
                if split is None or scene not in cameras:
                    raise ValueError(f'unknown/unreleased Hypersim frame or camera: {rgb}')
                seen_hypersim.add((scene, camera_name, frame_id))
            else:
                split = 'val' if (int(scene.split('_')[-1]) >= 90 if name == 'graspnet' else scene in validation) else 'train'
            target = pseudo if pseudo is not None and split == 'train' else gt
            if not gt.is_file() or not target.is_file():
                raise FileNotFoundError(f'missing required depth: {rgb}')
            metadata = {'target_scale': .001 if name == 'graspnet' else 1.,
                        'metric_scale': .001 if name == 'graspnet' else 1.,
                        'min_depth': .001, 'max_depth': 80. if name == 'hypersim' else 3.,
                        'depth_convention': 'radial' if name == 'hypersim' else 'z',
                        'depth_channels': 'replicated' if name == 'dreds' else 'single',
                        'registration': 'same_fov' if name == 'dreds' else 'pixel_aligned'}
            if name == 'hypersim':
                metadata.update(cameras[scene])
            path = lambda p: 'data://'+p.relative_to(data_root).as_posix()
            pools[split].append(ManifestRecord(name+'/'+rgb.relative_to(root).as_posix(), name,
                split, scene, path(rgb), path(target), 'pseudo_label' if target == pseudo else (
                    'sensor_gt' if name == 'graspnet' else 'rendered_gt'), path(gt), metadata))
        if name == 'hypersim' and seen_hypersim != set(official):
            raise ValueError(f'Hypersim public metadata/data coverage differs: missing={len(set(official)-seen_hypersim)}')
        if not pools['train'] or not pools['val']:
            raise ValueError(f'{name}: missing train/val pool')
        if train_limit > len(pools['train']) or val_limit > len(pools['val']):
            raise ValueError(f'{name}: acceptance limits exceed available samples')
        validate_disjoint(*(p for p in pools.values() if p))
        all_pools[name] = pools
        reports[name] = {'split_method': 'official_scene_v1' if name == 'hypersim' else (
                        'scene_0_89_vs_90_99' if name == 'graspnet' else 'seed42_internal_scene_holdout_from_train_packages'),
                        'pools': {s: {'samples': len(p), 'scenes': len({r.scene_id for r in p})} for s,p in pools.items()}}
    # Complete validation above before creating any output; never modify source data.
    output.mkdir(parents=True)
    (output/'metadata').mkdir()
    for source, name in [(camera_parameters, 'hypersim_camera_parameters.csv'), (official_splits, 'hypersim_official_splits.csv')]:
        shutil.copy2(source, output/'metadata'/name)
    for name, pools in all_pools.items():
        for split, records in pools.items():
            if records:
                path = output/'manifests/full'/name/(split+'.jsonl')
                save_manifest(records, path)
                reports[name]['pools'][split]['sha256'] = manifest_sha256(path)
        reports[name]['acceptance'] = {}
        for split, count in [('train', train_limit), ('val', val_limit)]:
            if count > len(pools[split]):
                raise ValueError(f'{name}/{split}: requested {count}, available {len(pools[split])}')
            selected = random.Random(seed+(split == 'val')).sample(pools[split], count)
            path = output/'manifests/smoke'/name/(split+'.jsonl')
            save_manifest(selected, path)
            reports[name]['acceptance'][split] = {'samples': count, 'sha256': manifest_sha256(path)}
    datasets = tuple(DatasetConfig(name, f'../manifests/smoke/{name}/train.jsonl',
                         f'../manifests/smoke/{name}/val.jsonl', sampling_weight=1.,
                         eval_min_depth=.001, eval_max_depth=80. if name == 'hypersim' else 3.) for name in roots)
    common = ExperimentConfig(datasets=datasets, model=ModelConfig(common_init_seed=seed),
        sampling=SamplingConfig('dataset_balanced', epoch_samples),
        optimization=OptimizationConfig(epochs=2, batch_size=4, accumulation_steps=4,
                                        num_workers=4, precision='fp32'),
        runtime=RuntimeConfig(data_root=str(runtime_data_root or data_root), output_dir=str(output_dir),
                              require_cuda=True, seed=seed, log_every=4))
    for variant, decoder in [('simple', 'simple'), ('full', 'sdf')]:
        config = replace(common, experiment_name='joint_v1_smoke_'+variant,
                         model=replace(common.model, decoder=decoder))
        save_experiment_config(config, output/'configs'/('smoke_'+variant+'.json'))
    report = {'status': 'prepared_not_trained', 'datasets': reports, 'source_metadata_sha256': {
              'camera': manifest_sha256(camera_parameters), 'official_splits': manifest_sha256(official_splits)},
              'sampling': {'dataset_probabilities': {name: 1/3 for name in roots}, 'replacement': True,
                           'epoch_samples': epoch_samples, 'not_a_full_dataset_epoch': True},
              'geometry': {'hypersim': 'native scene rays -> positive camera-axis z before resize/flip; meters already stored',
                           'dreds': 'replicated axial z meters, half-resolution same-FOV target; no radial conversion',
                           'graspnet': 'uint16 millimeters -> meters; pseudo train, sensor val'},
              'limitations': ['Engineering smoke and draft full manifests only; no formal training launched',
                              'Depth cutoffs 3/3/80m are provisional and recorded, not an official unified benchmark',
                              'DREDS validation is an internal scene holdout from available training archives, not official test']}
    (output/'preparation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--camera-parameters', required=True)
    parser.add_argument('--official-splits', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--runtime-data-root')
    parser.add_argument('--output-dir', default='experiments_v2')
    parser.add_argument('--train-limit', type=int, default=64)
    parser.add_argument('--val-limit', type=int, default=16)
    parser.add_argument('--epoch-samples', type=int, default=96)
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
