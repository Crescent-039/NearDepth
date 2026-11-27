"""Serializable and validated configuration for NearDepth V2 experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any
from .models.sdf import SDFOptions


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    train_manifest: str
    val_manifest: str
    sampling_weight: float = 1.0
    eval_min_depth: float | None = None
    eval_max_depth: float | None = None

    def validate(self) -> None:
        if not self.name:
            raise ValueError("dataset name must not be empty")
        if not self.train_manifest or not self.val_manifest:
            raise ValueError(f"dataset {self.name!r} requires train and val manifests")
        if not math.isfinite(self.sampling_weight) or self.sampling_weight <= 0:
            raise ValueError("sampling_weight must be positive")
        if (self.eval_min_depth is None) != (self.eval_max_depth is None):
            raise ValueError("set both dataset evaluation depth bounds or neither")
        if self.eval_min_depth is not None and not 0 <= self.eval_min_depth < self.eval_max_depth:
            raise ValueError("invalid dataset evaluation range")


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "dinov2_vits14_reg"
    decoder: str = "sdf"
    head: str = "adabins"
    feature_dim: int = 256
    n_bins: int = 100
    rho_min: float = 1e-3
    common_init_seed: int | None = None
    decoder_init_seed: int | None = None
    use_rgb_guidance: bool = False
    sdf: SDFOptions = field(default_factory=SDFOptions)

    def validate(self) -> None:
        self.sdf.validate()
        if self.decoder not in {"simple", "dpt", "sdf"}:
            raise ValueError(f"unsupported decoder: {self.decoder}")
        if self.head != "adabins":
            raise ValueError("NearDepth V2 comparisons must use the shared AdaBins-style head")
        if self.feature_dim <= 0 or self.n_bins < 2:
            raise ValueError("feature_dim must be positive and n_bins must be at least 2")
        if not 0.0 < self.rho_min < 1.0:
            raise ValueError("rho_min must be in (0, 1)")
        if self.use_rgb_guidance:
            raise ValueError("RGB guidance is intentionally disabled in NearDepth V2")
        if self.common_init_seed is not None and (not isinstance(self.common_init_seed, int) or self.common_init_seed < 0):
            raise ValueError("common_init_seed must be a nonnegative integer")
        if self.decoder_init_seed is not None and (not isinstance(self.decoder_init_seed, int) or self.decoder_init_seed < 0):
            raise ValueError("decoder_init_seed must be a nonnegative integer")


@dataclass(frozen=True)
class OptimizationConfig:
    epochs: int = 30
    batch_size: int = 16
    backbone_lr: float = 1e-5
    decoder_lr: float = 1e-4
    weight_decay: float = 1e-2
    num_workers: int = 4
    precision: str = "fp32"
    accumulation_steps: int = 1
    grad_clip_norm: float = 1.0
    max_train_batches: int | None = None

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.backbone_lr <= 0 or self.decoder_lr <= 0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0 or self.num_workers < 0:
            raise ValueError("weight_decay and num_workers must be non-negative")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16")
        if self.accumulation_steps < 1 or self.grad_clip_norm < 0:
            raise ValueError("invalid accumulation_steps or grad_clip_norm")
        if self.max_train_batches is not None and self.max_train_batches < 1:
            raise ValueError("max_train_batches must be positive (smoke runs only)")


@dataclass(frozen=True)
class SamplingConfig:
    mode: str = "concat"
    epoch_samples: int | None = None

    def validate(self):
        if self.mode not in {"concat", "dataset_balanced"}:
            raise ValueError("unknown sampling mode")
        if self.mode == "concat" and self.epoch_samples is not None:
            raise ValueError("concat sampling does not accept epoch_samples")
        if self.mode == "dataset_balanced" and (not isinstance(self.epoch_samples, int) or self.epoch_samples <= 0):
            raise ValueError("dataset_balanced sampling requires positive epoch_samples")


@dataclass(frozen=True)
class LossConfig:
    protocol: str = "positive_affine_anchor_v2"
    anchor_weight: float = 1.0
    trim: float = .1

    def validate(self) -> None:
        if self.protocol not in {"affine_v1", "positive_affine_anchor_v2"}:
            raise ValueError("unknown loss protocol")
        if not math.isfinite(self.trim) or not 0 <= self.trim < 1:
            raise ValueError("loss trim must be in [0, 1)")
        if not math.isfinite(self.anchor_weight) or self.anchor_weight < 0:
            raise ValueError("loss anchor_weight must be finite and nonnegative")
        if self.protocol == "positive_affine_anchor_v2" and self.anchor_weight <= 0:
            raise ValueError("direction-aware loss requires a positive anchor_weight")
        if self.protocol == "affine_v1" and self.anchor_weight != 0:
            raise ValueError("legacy affine_v1 has no depth anchor")


@dataclass(frozen=True)
class EvaluationConfig:
    alignment: str = "positive_scale_shift"
    min_depth: float = 1e-3
    max_depth: float = 3.0
    report_macro: bool = True
    report_micro: bool = True
    save_per_image: bool = True

    def validate(self) -> None:
        if self.alignment not in {"none", "scale", "scale_shift", "positive_scale_shift"}:
            raise ValueError(f"unsupported alignment: {self.alignment}")
        if not 0 <= self.min_depth < self.max_depth:
            raise ValueError("evaluation depth range is invalid")
        if not (self.report_macro or self.report_micro):
            raise ValueError("at least one aggregation method must be enabled")
        if not self.save_per_image:
            raise ValueError("NearDepth V2 requires per-image evaluation records")


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    deterministic: bool = True
    output_dir: str = "experiments_v2"
    data_root: str | None = None
    require_cuda: bool = False
    log_every: int = 50

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.output_dir:
            raise ValueError("output_dir must not be empty")
        if self.log_every < 1:
            raise ValueError("log_every must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str = "sdf_baseline"
    input_height: int = 448
    input_width: int = 448
    datasets: tuple[DatasetConfig, ...] = field(default_factory=tuple)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        if not self.experiment_name:
            raise ValueError("experiment_name must not be empty")
        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError("input size must be positive")
        if self.input_height % 14 or self.input_width % 14:
            raise ValueError("DINOv2 input height and width must be divisible by 14")
        tokens = (self.input_height//16)*(self.input_width//16)
        if not 129 <= tokens <= 1024:
            raise ValueError("shared AdaBins head requires 129..1024 tokens after H/4, patch4")
        names = [dataset.name for dataset in self.datasets]
        if len(names) != len(set(names)):
            raise ValueError("dataset names must be unique")
        for dataset in self.datasets:
            dataset.validate()
        self.model.validate()
        self.optimization.validate()
        self.sampling.validate()
        self.loss.validate()
        self.evaluation.validate()
        if self.loss.protocol == "positive_affine_anchor_v2" and self.evaluation.alignment != "positive_scale_shift":
            raise ValueError("direction-aware loss requires positive_scale_shift evaluation")
        self.runtime.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        data = dict(payload)
        data["datasets"] = tuple(DatasetConfig(**item) for item in data.get("datasets", ()))
        model = dict(data.get("model", {}))
        model["sdf"] = SDFOptions(**model.get("sdf", {}))
        data["model"] = ModelConfig(**model)
        data["optimization"] = OptimizationConfig(**data.get("optimization", {}))
        data["sampling"] = SamplingConfig(**data.get("sampling", {}))
        # Old configs/checkpoints remain explicitly legacy; never relabel old results as V2.
        legacy = "loss" not in data
        data["loss"] = LossConfig(protocol="affine_v1", anchor_weight=0.) if legacy else LossConfig(**data["loss"])
        data["evaluation"] = EvaluationConfig(**data.get("evaluation", {"alignment": "scale_shift"} if legacy else {}))
        data["runtime"] = RuntimeConfig(**data.get("runtime", {}))
        config = cls(**data)
        config.validate()
        return config


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ExperimentConfig.from_dict(payload)


def save_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    config.validate()
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
