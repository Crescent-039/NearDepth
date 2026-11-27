"""JSONL manifests that make every experiment's sample set auditable."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


VALID_SPLITS = {"train", "val", "test"}
VALID_SUPERVISION_SOURCES = {"pseudo_label", "sensor_gt", "rendered_gt"}


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    dataset: str
    split: str
    scene_id: str
    rgb_path: str
    target_depth_path: str
    supervision_source: str
    metric_depth_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        required = {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "scene_id": self.scene_id,
            "rgb_path": self.rgb_path,
            "target_depth_path": self.target_depth_path,
        }
        empty = [name for name, value in required.items() if not value]
        if empty:
            raise ValueError(f"manifest record has empty fields: {empty}")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"unsupported split: {self.split}")
        if self.supervision_source not in VALID_SUPERVISION_SOURCES:
            raise ValueError(f"unsupported supervision source: {self.supervision_source}")


def save_manifest(records: Iterable[ManifestRecord], path: str | Path) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = sorted(records, key=lambda record: record.sample_id)

    if not materialized:
        raise ValueError("cannot save an empty manifest")
    seen: set[str] = set()
    for record in materialized:
        record.validate()
        if record.sample_id in seen:
            raise ValueError(f"duplicate sample_id: {record.sample_id}")
        seen.add(record.sample_id)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in materialized:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def load_manifest(path: str | Path, check_files: bool = False, data_root=None) -> list[ManifestRecord]:
    manifest_path = Path(path)
    records: list[ManifestRecord] = []
    seen: set[str] = set()

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = ManifestRecord(**json.loads(line))
                record.validate()
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid manifest line {line_number}: {exc}") from exc
            if record.sample_id in seen:
                raise ValueError(f"duplicate sample_id at line {line_number}: {record.sample_id}")
            seen.add(record.sample_id)

            if check_files:
                paths = [record.rgb_path, record.target_depth_path]
                if record.metric_depth_path:
                    paths.append(record.metric_depth_path)
                missing = [item for item in paths if not resolve_path(manifest_path, item, data_root).is_file()]
                if missing:
                    raise FileNotFoundError(f"missing files for {record.sample_id}: {missing}")
            records.append(record)

    if not records:
        raise ValueError(f"manifest is empty: {manifest_path}")
    return records


def resolve_path(manifest_path: str | Path, item: str, data_root=None) -> Path:
    """Relative sample paths are always relative to the manifest, never cwd."""
    if item.startswith("data://"):
        if data_root is None:
            raise ValueError("data:// paths require an explicit data_root")
        relative = item[len("data://"):]
        path = Path(relative)
        if not relative or relative.startswith("/") or path.is_absolute() or ".." in path.parts or ":" in relative or "\\" in relative:
            raise ValueError("unsafe data:// path")
        root = Path(data_root).absolute()
        # Do not resolve symlinks here: a verified DREDS junction may live on another disk.
        return root / path
    path = Path(item)
    return path if path.is_absolute() else Path(manifest_path).resolve().parent / path


def validate_disjoint(*manifests: list[ManifestRecord]) -> None:
    """Reject frame and scene overlap between train/val/test manifests."""
    scenes, samples = {}, {}
    for records in manifests:
        for record in records:
            for registry, key in ((scenes, (record.dataset, record.scene_id)),
                                  (samples, (record.dataset, record.sample_id))):
                if key in registry and registry[key] != record.split:
                    raise ValueError(f"split overlap: {key}")
                registry[key] = record.split


def manifest_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
