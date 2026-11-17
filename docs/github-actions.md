# Hosted checks after separately authorized repository creation

These are instructions only; no remote action was performed during preparation.
After this reviewed candidate is present on the repository's default branch and
GitHub Actions is enabled:

1. Open **Actions → Research prototype checks → Run workflow**.
2. Select the intended default branch, then **Run workflow**.
3. Require all four Linux/Windows × Python 3.10/3.11 jobs to pass. Record the
   commit SHA, run URL, resolved versions and logs; investigate every failure.

Push and pull-request events also trigger the workflow. Manual dispatch requires
the workflow file on the default branch; see the [official instructions](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).

No secrets are needed. Token permissions are contents:read; checkout does not
persist credentials. There are no publishing, deployment, release or write steps.
The first step checks pristine file hashes before installation creates build/cache
files. Subsequent steps install declared dependencies, run pip check, pinned Ruff,
offline fixture tests and the baseline CLI. Dependencies are resolved online;
this compatibility matrix is not exact historical Windows replay. Exact archive
reconstruction remains the separate recorded Phase 4 clean-room check. Hosted
success has not been observed. Actions use reviewed major-version tags, not
immutable commit pins; pinning approved revisions is optional hardening.

CI success is software evidence only and cannot authorize model deployment or
replace author review, licensing decisions or statistical validation.
