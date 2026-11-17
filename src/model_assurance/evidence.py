"""Content manifests for new evidence; never reseal a historical run implicitly."""

import json
from pathlib import Path

from .external_data import sha256


def seal(root: Path) -> None:
    target = root / "artifact-hashes.json"
    if target.exists():
        raise FileExistsError("Refusing to replace an existing evidence manifest")
    hashes = {
        p.relative_to(root).as_posix(): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p != target
    }
    target.write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
