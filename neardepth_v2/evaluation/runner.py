"""Evaluate each image at native GT resolution, independent of batch size."""
import torch
from torch.nn import functional as F
from .metrics import evaluate_image


@torch.no_grad()
def evaluate_dataset(model, dataset, device, config):
    model.eval()
    rows = []
    for sample in dataset:
        result = model(sample["rgb"].unsqueeze(0).to(device))
        prediction = F.interpolate(result["prediction"], size=sample["metric_depth"].shape[-2:],
                                   mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        row = evaluate_image(prediction, sample["metric_depth"][0].numpy(),
                             sample["metric_mask"][0].numpy(), config.alignment,
                             config.min_depth, config.max_depth)
        rows.append({"sample_id": sample["sample_id"], "dataset": sample["dataset"], **row})
    return rows
