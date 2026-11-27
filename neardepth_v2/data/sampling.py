"""Explicit dataset probabilities, independent of the sizes of its manifests."""
import math
import torch
from torch.utils.data import WeightedRandomSampler


def make_training_sampler(datasets, weights, config, generator):
    config.validate()
    if len(datasets) != len(weights) or not datasets or any(len(d) == 0 for d in datasets):
        raise ValueError('sampling requires nonempty datasets with matching weights')
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('dataset sampling weights must be finite and positive')
    if config.mode == 'concat':
        if any(w != 1 for w in weights):
            raise ValueError('non-unit weights require dataset_balanced sampling')
        return None
    # Per-image mass w_d/N_d gives total dataset mass w_d, not w_d*N_d.
    per_image = torch.cat([torch.full((len(d),), w/len(d), dtype=torch.float64)
                           for d, w in zip(datasets, weights)])
    return WeightedRandomSampler(per_image, config.epoch_samples, replacement=True, generator=generator)
