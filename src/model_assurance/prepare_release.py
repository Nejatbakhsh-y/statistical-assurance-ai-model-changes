"""Build a new allowlisted, unpublished repository and traceable evidence derivative."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .evidence import seal
from .external_data import sha256
from .external_validation import write_dataset_report

ROOT_FILES = [
    "README.md",
    "AGENTS.md",
    "pyproject.toml",
    "requirements.txt",
    ".gitignore",
    "windows-runtime.ps1",
    "setup-project.ps1",
    "run-all.ps1",
    "run-phase2.ps1",
    "run-phase3.ps1",
    "publish-github.ps1",
    "bootstrap-windows.ps1",
]
DOCS = [
    "datasets.md",
    "reproducibility.md",
    "research-limitations.md",
    "phase2_protocol.md",
    "phase3_protocol.md",
    "mathematical_foundations.md",
    "methodology.md",
    "validation_protocol.md",
]


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def summary(repo):
    p1 = json.loads((repo / "evidence/phase1/results/metrics.json").read_text())
    p2 = json.loads((repo / "evidence/phase2/metrics.json").read_text())
    p3 = json.loads((repo / "evidence/phase3/integrated.json").read_text())
    lines = [
        "# Archived empirical findings",
        "",
        "All categories are exploratory and conditional. Coverage, power and false-release control have not been established. Human review is mandatory; no automatic deployment authorization.",
        "",
        "| Phase/task | Challenger | Mean log-loss change | Interval | Interpretation |",
        "|---|---|---:|---|---|",
    ]
    ci = p1["paired_logloss_difference_ci"]
    lines.append(
        f"| 1 / breast cancer | Fixed transform | {ci['estimate']:.8f} | [{ci['lower']:.8f}, {ci['upper']:.8f}] | 95%, additional evaluation |"
    )
    for model, s in p2["summary"].items():
        ci = s["paired_logloss_ci"]
        lines.append(
            f"| 2 / breast cancer | {model} | {ci['estimate']:.8f} | [{ci['lower']:.8f}, {ci['upper']:.8f}] | 97.5%, exploratory |"
        )
    for name, models in p3["datasets"].items():
        for model, s in models.items():
            ci = s["joint_refit_and_test_ci"]
            lines.append(
                f"| 3 / {name} | {model} | {ci['estimate']:.8f} | [{ci['lower']:.8f}, {ci['upper']:.8f}] | 98.75%, conditional {s['assurance']['category']} |"
            )
    lines += [
        "",
        "Negative change favors challenger. The seven rows are not one controlled family. Phase 3 alone allocates alpha across four primary contrasts. Separate domains are not independent-site validation of a frozen medical model.",
        "",
        "The proposed evidence is linked to immutable local runs by [lineage](../evidence/lineage.json). Model data and raw score arrays are copied unchanged; redacted environment fields and regenerated narrative are explicitly listed derivatives. Original archives were not rewritten. SHA256 does not authenticate a publisher.",
        "",
        "Phase 1/2 use clipped probability loss. Phase 3 uses score-space loss and a different forest sigmoid input; these are not causal ablations of an unchanged intervention. Phase 2 forest saturation is influenced by one high-leverage case. Ionosphere has only 88 primary test cases and unstable split/tail behavior. No case was deleted or tuning performed against these outcomes in Phase 4.",
    ]
    (repo / "docs/validation-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(output, source="."):
    source = Path(source).resolve()
    outer = Path(output).resolve()
    if outer.exists():
        raise FileExistsError("Release staging must be new")
    repo = outer / "repository"
    repo.mkdir(parents=True)
    selected = [source / p for p in ROOT_FILES]
    selected += list((source / "src/model_assurance").glob("*.py"))
    selected += list((source / "tests").glob("*.py"))
    selected += [p for p in (source / "configs").iterdir() if p.suffix in (".yaml", ".json")]
    selected += [source / "docs" / p for p in DOCS]
    selected += [source / ".github/workflows/ci.yml"]
    for p in selected:
        if p.is_symlink():
            raise ValueError("Symlink in release source")
        target = repo / p.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    origins = {}

    def copy(origin, relative):
        src = source / origin
        dst = repo / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        origins[relative] = {
            "original_relative_path": origin,
            "original_sha256": sha256(src),
            "staged_sha256": sha256(dst),
            "operation": "byte-identical",
        }

    phase1 = {
        "results/metrics.json": "results/metrics.json",
        "results/case_predictions.csv": "results/case_predictions.csv",
        "reports/validation.html": "reports/validation.html",
        "figures/reliability.png": "figures/reliability.png",
    }
    for origin, relative in phase1.items():
        copy(origin, "evidence/phase1/" + relative)
    for name in ("metrics.json", "predictions.csv", "validation.md"):
        copy("results/phase2-20261117-075627-011/" + name, "evidence/phase2/" + name)
    copy(
        "reports/phase4-20261117/before/model_assurance/robustness.py",
        "evidence/phase2/source/robustness.py",
    )
    original = source / "results/phase3-20261117-095543-473"
    target = repo / "evidence/phase3"
    for name in ("banknote", "ionosphere"):
        for file in ("metrics.json", "predictions.npz"):
            copy(
                f"results/phase3-20261117-095543-473/{name}/{file}",
                f"evidence/phase3/{name}/{file}",
            )
        m = json.loads((target / name / "metrics.json").read_text())
        write_dataset_report(m, target / name / "validation.md")
        origins[f"evidence/phase3/{name}/validation.md"] = {
            "original_relative_path": f"results/phase3-20261117-095543-473/{name}/validation.md",
            "original_sha256": sha256(original / name / "validation.md"),
            "staged_sha256": sha256(target / name / "validation.md"),
            "operation": "regenerated narrative; unchanged numeric evidence",
        }
    copy("results/phase3-20261117-095543-473/integrated.json", "evidence/phase3/integrated.json")
    for p in (original / "source").glob("*.py"):
        copy(p.relative_to(source).as_posix(), f"evidence/phase3/source/{p.name}")
    env = json.loads((original / "environment.json").read_text())
    fields = ("executable", "platform", "argv")
    public_fields = {k: v for k, v in env.items() if k not in fields}
    env.update(
        executable="python",
        platform="archived Windows runtime; host-specific detail omitted",
        argv=[
            "python",
            "-m",
            "model_assurance.external_validation",
            "--config",
            "configs/phase3.yaml",
            "--output",
            "results/new-phase3",
            "--offline",
        ],
    )
    (target / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    origins["evidence/phase3/environment.json"] = {
        "original_relative_path": "results/phase3-20261117-095543-473/environment.json",
        "original_sha256": sha256(original / "environment.json"),
        "staged_sha256": sha256(target / "environment.json"),
        "operation": "redacted execution descriptor; not the raw invocation",
        "redacted_fields": list(fields),
        "unchanged_public_fields_sha256": canonical(public_fields),
    }
    copy("configs/phase3.yaml", "evidence/phase3/configuration.yaml")
    integrated = json.loads((target / "integrated.json").read_text())
    rows = [
        "# Exploratory Phase 3 evidence",
        "",
        "Conditional research classifications only. Coverage, power and false-release control have not been established. Human review is mandatory; automatic deployment authorization is false.",
        "",
        "| Dataset | Challenger | Mean log-loss change | Joint interval | Assurance |",
        "|---|---|---:|---|---|",
    ]
    for name, models in integrated["datasets"].items():
        for model, s in models.items():
            ci = s["joint_refit_and_test_ci"]
            rows.append(
                f"| {name} | {model} | {ci['estimate']:.8f} | [{ci['lower']:.8f}, {ci['upper']:.8f}] | {s['assurance']['category']} |"
            )
    rows += [
        "",
        "See the repository scientific limitations and dataset-specific reports. These are external tasks, not frozen-model site validation. Source snapshots are provenance evidence, not the current verifier implementation. A detached semantic-verification record does not certify coverage.",
    ]
    (target / "validation.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    origins["evidence/phase3/validation.md"] = {
        "original_relative_path": "results/phase3-20261117-095543-473/validation.md",
        "original_sha256": sha256(original / "validation.md"),
        "staged_sha256": sha256(target / "validation.md"),
        "operation": "regenerated conditional narrative; unchanged table values",
    }
    seal(target)
    for name in ("banknote", "ionosphere"):
        copy(f"results/external-cache/{name}.data", f"evidence/datasets/{name}.data")
    (repo / "evidence/lineage.json").write_text(
        json.dumps(
            {
                "schema": "release-derivative-v1",
                "artifacts": origins,
                "trust": "Local source hashes checked at export; not publisher signatures. Original private metadata remains local.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary(repo)
    (repo / "repro").mkdir()
    # Runtime and test distributions actually present; build tools explicitly pinned.
    import importlib.metadata

    requirements = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
        and d.metadata["Name"].lower()
        not in ("pip", "setuptools", "wheel", "model-assurance-research")
    }
    requirements.update(setuptools="80.9.0", wheel="0.45.1")
    (repo / "repro/requirements-win-py311.txt").write_text(
        "\n".join(f"{n}=={v}" for n, v in sorted(requirements.items())) + "\n"
    )
    manifest = {
        "schema": "publication-manifest-v1",
        "purpose": "Unpublished research-prototype review tree",
        "license": "Code license pending author approval; no open-source grant",
        "included": {
            p.relative_to(repo).as_posix(): sha256(p)
            for p in sorted(repo.rglob("*"))
            if p.is_file()
        },
        "excluded": {
            ".venv/**": "local environment",
            "**/__pycache__/**": "bytecode",
            "**/.pytest_cache/**": "test cache",
            "**/.ruff_cache/**": "lint cache",
            "**/*.egg-info/**": "build metadata",
            ".env*": "private environment",
            "**/*.key": "private-key candidates",
            "reports/**": "working archives/audits, including falsified scratch fixtures",
            "results/**": "unreviewed generated outputs and caches",
            "figures/**": "working figures except explicitly selected evidence",
            "docs/research_demo.md": "historical conversational context not needed for public release",
            "offline-wheels/**": "separate local verification kit, not a proposed Git commit",
        },
        "scientific_status": "exploratory conditional; coverage/power/false-release control unestablished",
    }
    (repo / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return repo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build(args.output))


if __name__ == "__main__":
    main()
