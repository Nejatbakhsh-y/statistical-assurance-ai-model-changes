# Dataset attribution and licenses

Original project code is MIT licensed with author authorization. This grant does
not relicense third-party datasets, dependency code or their bundled notices. The official UCI
pages list these datasets under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

| Dataset | Attribution and source | Representation |
|---|---|---|
| Wisconsin Diagnostic Breast Cancer | Wolberg, Mangasarian, N. Street and W. Street (1993). [UCI](https://archive.ics.uci.edu/dataset/17/breast%2Bcancer%2Bwisconsin%2Bdiagnostic), DOI 10.24432/C5DW2B | scikit-learn built-in data, malignant=0/benign=1; 569 cases, 30 features; runtime 1.9.1, X/y hash in Phase 2 metrics |
| Banknote Authentication | Lohweg (2012). [UCI](https://archive.ics.uci.edu/dataset/267/banknote%2Bauthentication), DOI 10.24432/C55P57 | Original 0/1 labels; semantic names not inferred. Snapshot 2026-11-17 |
| Ionosphere | Sigillito, Wing, Hutton and Baker (1989). [UCI](https://archive.ics.uci.edu/dataset/52/ionosphere), DOI 10.24432/C5W01B | b=0, g=1 (good return); snapshot 2026-11-17 |

`configs/datasets-phase3.json` records raw download URLs, byte counts, schema,
label mappings and SHA256. No upstream release number was advertised: versions
are content-addressed snapshots. Hashes came from official HTTPS downloads, not
independent publisher signatures. Reviewed raw files under evidence/datasets are
selected reproduction inputs, not a copied cache directory.

Changes: 24 Banknote and one Ionosphere feature-identical duplicates were removed
before partitioning, retaining first raw IDs. Conflicting labels would fail.
Unchanged raw downloads are included; removed IDs are in metrics. Missing specimen/
session IDs mean within-dataset independence remains unverified.

Python dependency wheels retain their licenses and bundled notices. The optional
offline wheelhouse is outside the proposed Git repository, not an instruction to
commit binary dependencies or an additional code-license grant.
