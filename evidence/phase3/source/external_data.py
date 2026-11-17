"""Pinned public datasets. Files stay in an ignored cache, never silently updated."""

import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_bytes(content: bytes, spec: dict) -> None:
    """Integrity is against a pinned HTTPS snapshot, not a publisher signature."""
    if len(content) != spec["bytes"] or hashlib.sha256(content).hexdigest() != spec["sha256"]:
        raise ValueError("Dataset integrity mismatch; preserve cache and investigate upstream changes")


def parse_data(content: bytes, spec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Validate schema; remove feature-identical rows before any partitioning.

    First-occurrence raw row IDs are retained. Conflicting duplicate labels fail.
    No fitted transformation or outcome-based selection is performed here.
    """
    verify_bytes(content, spec)
    rows = list(csv.reader(io.StringIO(content.decode("ascii"))))
    if len(rows) != spec["rows"] or any(len(r) != spec["features"] + 1 for r in rows):
        raise ValueError("Dataset schema or row count changed")
    try:
        x = np.asarray([[float(v) for v in r[:-1]] for r in rows])
        y = np.asarray([spec["labels"][r[-1].strip()] for r in rows], dtype=int)
    except (ValueError, KeyError) as exc:
        raise ValueError("Invalid numeric value or target label") from exc
    if not np.isfinite(x).all() or set(y) != {0, 1}:
        raise ValueError("Expected finite features and both binary classes")
    seen, keep, removed = {}, [], []
    for i, row in enumerate(x):
        key = tuple(row)
        if key in seen:
            if y[seen[key]] != y[i]:
                raise ValueError("Conflicting labels for feature-identical records")
            removed.append(i)
        else:
            seen[key] = i
            keep.append(i)
    ids = np.asarray(keep, dtype=int)
    return x[ids], y[ids], ids, {"raw_rows": len(rows), "unique_rows": len(ids),
                                "removed_duplicate_raw_ids": removed}


def load_dataset(name: str, manifest: dict, cache: str | Path, download: bool = True) -> tuple:
    """Verify existing cache every time; write a download only after verification."""
    spec = manifest[name]
    path = Path(cache) / f"{name}.data"
    if path.exists():
        content = path.read_bytes()
    elif download:
        with urlopen(spec["url"], timeout=60) as response:
            content = response.read()
        verify_bytes(content, spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as f:
            f.write(content)
    else:
        raise FileNotFoundError(f"Verified cache required: {path}")
    return parse_data(content, spec)
