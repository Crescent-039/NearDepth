"""Read-only full label/provenance audit; writes reports only, never data."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
from PIL import Image

from .generate_graspnet_labels import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--reports', required=True)
    args = parser.parse_args()
    root, reports = Path(args.root).resolve(), Path(args.reports).resolve()
    settings = json.loads((reports/'settings.json').read_text(encoding='utf-8'))
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    with sqlite3.connect((reports/'labels.sqlite').as_uri()+'?mode=ro', uri=True) as db:
        records = {r[0]: r[1:] for r in db.execute('SELECT path,settings,rgb_sha,output_sha FROM labels')}
    images = sorted(root.glob('train_*/scene_*/realsense/rgb/*.png'))
    keys = {p.relative_to(root).as_posix() for p in images}
    outputs = list(root.glob('train_*/scene_*/realsense/synthetic_depth/*.png'))
    expected = {p.parent.parent/'synthetic_depth'/p.name for p in images}
    errors = []
    if len(images) != 25600 or keys != set(records):
        errors.append('RGB/ledger keys or expected 25600 count differ')
    if set(outputs) != expected:
        errors.append('Output set differs from expected RGB set')
    package_counts = Counter()
    label_bytes = 0
    minimum, maximum = 65535, 0
    started = time.perf_counter()

    def check(rgb):
        key = rgb.relative_to(root).as_posix()
        try:
            output = rgb.parent.parent/'synthetic_depth'/rgb.name
            record = records[key]
            if record != (fingerprint, digest(rgb), digest(output)):
                raise ValueError('Provenance/hash mismatch')
            with Image.open(rgb) as image:
                size = image.size
            raw = rgb.parent.parent/'depth'/rgb.name
            with Image.open(raw) as image:
                image.load()
                if image.size != size:
                    raise ValueError('Raw depth/RGB dimensions differ')
            with Image.open(output) as image:
                depth = np.asarray(image)
                if image.format != 'PNG' or depth.dtype != np.uint16 or depth.shape != size[::-1]:
                    raise ValueError('Label must be matched 2D uint16 PNG')
                low, high = int(depth.min()), int(depth.max())
                if low <= 0:
                    raise ValueError('Nonpositive label pixel')
            return key, output.stat().st_size, low, high, None
        except Exception as exc:
            return key, 0, 0, 0, str(exc)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (key, size, low, high, error) in enumerate(pool.map(check, images), 1):
            if error:
                errors.append({'rgb': key, 'error': error})
            else:
                package_counts[key.split('/')[0]] += 1
                label_bytes += size
                minimum, maximum = min(minimum, low), max(maximum, high)
            if index % 500 == 0:
                progress = {'checked': index, 'total': len(images), 'errors': len(errors)}
                (reports/'audit_progress.json').write_text(json.dumps(progress), encoding='utf-8')
                print(json.dumps(progress), flush=True)
    report = {'status': 'passed' if not errors else 'failed', 'rgb_count': len(images),
              'ledger_count': len(records), 'output_count': len(outputs),
              'verified_by_package': dict(package_counts), 'label_bytes': label_bytes,
              'label_gib': label_bytes/1024**3, 'min_mm': minimum, 'max_mm': maximum,
              'elapsed_seconds': time.perf_counter()-started, 'errors': errors,
              'scope': 'All RGB/output SHA256 and settings provenance, raw depth decode, output decode/dtype/size/strictly-positive pixels; not teacher accuracy validation'}
    (reports/'full_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
