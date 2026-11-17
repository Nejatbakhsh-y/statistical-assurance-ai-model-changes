"""Verify pristine checkout byte hashes before tools generate local artifacts."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "release-manifest.json").read_text(encoding="utf-8-sig"))
actual = {p.relative_to(root).as_posix(): p for p in root.rglob("*")
          if p.is_file() and ".git" not in p.relative_to(root).parts
          and p != root / "release-manifest.json"}
if set(actual) != set(manifest["included"]):
    raise SystemExit("FAIL: unlisted or missing files")
for name, path in actual.items():
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["included"][name]:
        raise SystemExit("FAIL: altered file: " + name)
print("PASS: " + str(len(actual)) + " listed files match SHA256")
