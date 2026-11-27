"""Atomic epoch-boundary checkpoints with weights_only-compatible RNG state."""
import os
from pathlib import Path
import random
import json
from contextlib import contextmanager
import numpy as np
import torch


@contextmanager
def run_lease(output):
    """Never auto-remove a crash lock: a user must confirm the owner exited."""
    output = Path(output).absolute()
    lock = output.with_name(output.name + '.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = lock.open('x', encoding='utf-8')
    except FileExistsError as exc:
        raise RuntimeError(f'run locked: {lock}; verify recorded PID exited before removing stale lock') from exc
    try:
        with handle:
            json.dump({'pid': os.getpid(), 'run': str(output)}, handle)
        yield
    finally:
        lock.unlink()


def capture_rng(generator):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(),
            numpy_state[2], numpy_state[3], numpy_state[4]], "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("resume requires the same CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_resume(checkpoint, config, hashes, source_hashes, backbone_hash):
    if checkpoint.get("format_version") != 2:
        raise ValueError("checkpoint lacks epoch-boundary resume state (format_version=2)")
    old, new = checkpoint["config"], config.to_dict()
    # Epoch extension and relocating run/data/manifests are allowed only when content hashes agree.
    import copy
    old, new = copy.deepcopy(old), copy.deepcopy(new)
    for payload in (old, new):
        payload["optimization"].pop("epochs")
        payload["runtime"].pop("output_dir", None)
        payload["runtime"].pop("data_root", None)
        for dataset in payload["datasets"]:
            dataset.pop("train_manifest")
            dataset.pop("val_manifest")
    if old != new or checkpoint["manifest_hashes"] != hashes:
        raise ValueError("resume config or manifest hashes differ from checkpoint")
    if checkpoint["source_sha256"] != source_hashes or checkpoint["backbone_weights_sha256"] != backbone_hash:
        raise ValueError("resume source or backbone weights changed")
    if config.optimization.epochs <= checkpoint["epoch"]+1:
        raise ValueError("requested epochs already completed")
