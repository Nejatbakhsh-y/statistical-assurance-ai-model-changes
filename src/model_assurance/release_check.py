"""Read-only release inventory, privacy, link, lineage and legacy-evidence checks."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from sklearn.datasets import load_breast_cancer

from .external_data import sha256
from .prepare_release import canonical
from .strict_verification import check_hashes, read_json, safe_member
from .verification_math import model_metrics, require_equal


def legacy_loss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -y * np.log(p) - (1 - y) * np.log1p(-p)


def legacy_ci(d, seed, reps, confidence):
    rng = np.random.default_rng(seed)
    draws = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)]
    return {
        "estimate": float(d.mean()),
        "lower": float(np.quantile(draws, (1 - confidence) / 2)),
        "upper": float(np.quantile(draws, 1 - (1 - confidence) / 2)),
        "confidence": confidence,
        "replicates": reps,
    }


def legacy_checks(root):
    p = root / "evidence/phase1"
    m = read_json(p / "results/metrics.json")
    rows = list(csv.DictReader((p / "results/case_predictions.csv").open()))
    y = np.array([int(r["true_label"]) for r in rows])
    probs = []
    for model in ("baseline", "challenger"):
        prob = np.array([float(r[model + "_probability"]) for r in rows])
        score = np.array([float(r[model + "_score"]) for r in rows])
        values = model_metrics(y, score)["metrics"]
        values["log_loss"] = float(legacy_loss(y, prob).mean())
        require_equal(m[model + "_metrics"], values, "Phase1 metrics")
        probs.append(prob)
    c = m["configuration"]
    require_equal(
        m["paired_logloss_difference_ci"],
        legacy_ci(
            legacy_loss(y, probs[1]) - legacy_loss(y, probs[0]),
            c["seed"],
            c["bootstrap_replicates"],
            c["confidence"],
        ),
        "Phase1 CI",
    )
    p = root / "evidence/phase2"
    m = read_json(p / "metrics.json")
    c = m["configuration"]
    data = load_breast_cancer()
    if (
        hashlib.sha256(data.data.tobytes() + data.target.tobytes()).hexdigest()
        != m["dataset_sha256"]
    ):
        raise ValueError("Built-in breast-cancer dataset version/hash mismatch")
    if sha256(p / "source/robustness.py") != m["implementation_sha256"]:
        raise ValueError("Phase2 source lineage mismatch")
    rows = list(csv.DictReader((p / "predictions.csv").open()))
    lookup = {
        (r["challenger"], int(r["split_seed"]), int(r["training_seed"]), int(r["subject_id"])): r
        for r in rows
    }
    for name, summary in m["summary"].items():
        differences = []
        for split in c["split_seeds"]:
            for seed in c["training_seeds"]:
                rr = [lookup[name, split, seed, i] for i in m["evaluation_ids"]]
                y = np.array([int(r["true_label"]) for r in rr])
                if not np.array_equal(y, data.target[m["evaluation_ids"]]):
                    raise ValueError("Phase2 label mismatch")
                record = next(
                    r
                    for r in m["runs"]
                    if (r["challenger"], r["split_seed"], r["training_seed"]) == (name, split, seed)
                )
                probs = []
                for model in ("baseline", "challenger"):
                    prob = np.array([float(r[model + "_probability"]) for r in rr])
                    score = np.array([float(r[model + "_score"]) for r in rr])
                    derived = model_metrics(y, score)
                    derived["metrics"]["log_loss"] = float(legacy_loss(y, prob).mean())
                    require_equal(record[model + "_metrics"], derived["metrics"], "Phase2 metrics")
                    require_equal(
                        record["baseline_reliability" if model == "baseline" else "reliability"],
                        derived["calibration"]["bins"],
                        "Phase2 bins",
                    )
                    probs.append(prob)
                differences.append(legacy_loss(y, probs[1]) - legacy_loss(y, probs[0]))
        require_equal(
            summary["paired_logloss_ci"],
            legacy_ci(
                np.mean(differences, axis=0),
                c["bootstrap_seed"],
                c["bootstrap_replicates"],
                1 - (1 - c["family_confidence"]) / 2,
            ),
            "Phase2 CI",
        )


def check(root):
    root = Path(root).resolve()
    m = read_json(root / "release-manifest.json")
    if m.get("schema") != "publication-manifest-v1" or not isinstance(m.get("excluded"), dict):
        raise ValueError("Missing publication manifest policy")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
        and ".git" not in p.relative_to(root).parts
        and p != root / "release-manifest.json"
    }
    if set(m["included"]) != actual:
        raise ValueError("Unlisted or missing files in proposed repository")
    for relative, digest in m["included"].items():
        if sha256(safe_member(root, relative)) != digest:
            raise ValueError(f"Publication hash mismatch: {relative}")
    forbidden = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "offline-wheels"}
    pattern = re.compile(
        r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|[A-Za-z]:[\\/]Users[\\/]|"
        + "One"
        + "Drive"
        + ")"
    )
    link_count = 0
    for relative in actual:
        p = safe_member(root, relative)
        if (
            set(Path(relative).parts) & forbidden
            or p.name.startswith(".env")
            or p.suffix in (".key", ".pem")
        ):
            raise ValueError("Excluded/private artifact in proposed release")
        if p.suffix in (".py", ".ps1", ".md", ".json", ".yaml", ".toml", ".txt", ".yml"):
            text = p.read_text(encoding="utf-8-sig")
            if pattern.search(text):
                raise ValueError(f"Credential or machine-local path candidate: {relative}")
            if p.suffix == ".md":
                for target in re.findall(r"\]\(([^)]+)\)", text):
                    target = target.strip("<>").split("#")[0]
                    if not target or re.match(r"[A-Za-z]+:", target):
                        continue
                    link_count += 1
                    if not (p.parent / unquote(target)).resolve().exists():
                        raise ValueError(f"Broken local Markdown link: {relative}: {target}")
    lineage = read_json(root / "evidence/lineage.json")
    for relative, record in lineage["artifacts"].items():
        path = safe_member(root, relative)
        if sha256(path) != record["staged_sha256"]:
            raise ValueError("Lineage staged hash mismatch")
        if (
            record["operation"] == "byte-identical"
            and record["original_sha256"] != record["staged_sha256"]
        ):
            raise ValueError("Copied artifact no longer matches archive")
        if "redacted_fields" in record:
            env = read_json(path)
            kept = {k: v for k, v in env.items() if k not in record["redacted_fields"]}
            if canonical(kept) != record["unchanged_public_fields_sha256"]:
                raise ValueError("Redaction changed scientific environment metadata")
    check_hashes(root / "evidence/phase3")
    legacy_checks(root)
    return {
        "status": "PASS",
        "included_files": len(actual),
        "lineage_records": len(lineage["artifacts"]),
        "local_markdown_links_checked": link_count,
        "privacy_scan": "specified high-specificity patterns; not a security certification",
        "phase1_phase2_reconstruction": "PASS",
        "phase3_semantic_verification": "run the separate read-only verifier",
        "automatic_deployment_allowed": False,
        "remote_actions": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps(check(args.root), indent=2))


if __name__ == "__main__":
    main()
