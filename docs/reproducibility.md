# Reproducibility and safe execution

The archived Windows replay uses Python 3.11.9 and exact constraints in repro/.
Different patch/library versions are not silently asserted equivalent. Python
>=3.10 is the source compatibility target. Hosted CI is unverified until run.

## Windows short path

From the project in VS Code's Windows PowerShell terminal:

```powershell
& .\setup-project.ps1 -Python311 'C:\path\to\Python311\python.exe'
& .\run-all.ps1 -Python311 'C:\path\to\Python311\python.exe'
```

The path is a user-supplied example, not the developer's computer. Discovery tests
an actual 3.11 interpreter, not launcher presence. The shared runtime validates
the actual venv. It creates R: only if unused and refuses to replace a different
mapping. It refuses a wrong-version existing environment; use `-EnvironmentPath
.venv311` to create a separate one. No registry, PATH, security or Python 3.13
change occurs. Keep the temporary mapping for the session. Setup does not run an
experiment. Baseline CLI uses a new timestamped output by default.

## Verify archived evidence without a full refit rerun

```powershell
Set-Location R:\
& .\.venv\Scripts\python.exe -m model_assurance.release_check --root .
& .\.venv\Scripts\python.exe -m model_assurance.verify_external --output evidence/phase3 --data-dir evidence/datasets --read-only --record reports/verification-new.json
```

The strict verifier recalculates every stored metric/calibration/decision field,
all interval metadata and values, summaries, role RNG streams and categories. It
replays original model bundles and every bootstrap logistic validation selection.
It reconstructs stored refit-score distributions but does not retrain every forest.
It hashes archived source without executing it; its own version is recorded.
Passing is scoped consistency, not coverage or publisher authentication. Detached
records are exclusive-create outside evidence. Archives are never written by it.

Full studies use run-phase2.ps1 and run-phase3.ps1, always with new outputs.
Phase 3 is expensive. Use -Offline with the included pinned data. Overlapping
splits are descriptive; repeated predictions never become independent subjects.

The offline dependency kit is a sibling of the proposed repository and excluded
from the proposed Git commit. Clean-room replay copies only staged source/evidence,
creates a new environment, installs wheels with --no-index, unsets PYTHONPATH and
checks that imports resolve inside the new environment. This is not a Linux or
hosted-CI result.

`evidence/lineage.json` records original hashes and copied/redacted/regenerated
derivatives. Original archives remain unchanged. Public metadata omits executable/
host paths but preserves scientific versions/configuration. The malformed historical
`Set-Location R:& ...` command is superseded by these commands, not rewritten.
