"""Render observed Phase 3 calibration without fitting or pooling repeated cases.

Run from any directory with the project's numerical dependencies installed:
    python repro/plot_calibration.py
Only the two named README derivatives are written; archived evidence is read-only.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

MODELS = ("baseline", "tuned_logistic", "calibrated_forest")
LABELS = ("Baseline logistic", "Validation-tuned logistic", "Sigmoid-calibrated forest")
COLORS = ("#2166AC", "#D97917", "#12816C")
MARKERS = ("o", "s", "D")
BINS = 10


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summaries(root, dataset):
    """Check source hashes and saved calibration, then derive one original fit."""
    evidence = root / "evidence/phase3"
    hashes = json.loads((evidence / "artifact-hashes.json").read_text(encoding="utf-8-sig"))
    # Match the archived manifest's path spelling on Windows and POSIX.
    for name in ("metrics.json", "predictions.npz"):
        relative = dataset + "/" + name
        expected = next(v for k, v in hashes.items() if k.replace("\\", "/") == relative)
        if sha256(evidence / relative) != expected:
            raise ValueError("Archived evidence hash mismatch: " + relative)
    metadata = json.loads((evidence / dataset / "metrics.json").read_text())
    c = metadata["configuration"]
    split, seed = c["split_seeds"][0], c["training_seeds"][0]
    record = next(
        r for r in metadata["fits"] if (r["split_seed"], r["training_seed"]) == (split, seed)
    )
    with np.load(evidence / dataset / "predictions.npz", allow_pickle=False) as archive:
        scores = archive["observed_scores"][0, 0].copy()
        y = archive["test_labels"][0].copy()
        ids = archive["test_ids"][0].copy()
    if scores.shape != (len(MODELS), len(y)) or len(np.unique(ids)) != len(y):
        raise ValueError("Expected one original prediction per model per held-out subject")
    if not np.isfinite(scores).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Finite scores and binary labels required")
    rows = []
    for name, score in zip(MODELS, scores, strict=True):
        probability = expit(score)
        assignment = np.minimum((probability * BINS).astype(int), BINS - 1)
        bins = []
        for index in range(BINS):
            mask = assignment == index
            count = int(mask.sum())
            bins.append(
                {
                    "bin": index,
                    "count": count,
                    "mean_probability": float(probability[mask].mean()) if count else None,
                    "observed_rate": float(y[mask].mean()) if count else None,
                }
            )
        archived = record["models"][name]["calibration"]
        for calculated, saved in zip(bins, archived["bins"], strict=True):
            if calculated["count"] != saved["count"]:
                raise ValueError("Bin-count mismatch")
            for field in ("mean_probability", "observed_rate"):
                if calculated[field] is None:
                    if saved[field] is not None:
                        raise ValueError("Empty-bin mismatch")
                elif not np.isclose(calculated[field], saved[field], rtol=1e-12, atol=1e-12):
                    raise ValueError("Saved calibration mismatch")
        ece = sum(
            b["count"] / len(y) * abs(b["mean_probability"] - b["observed_rate"])
            for b in bins
            if b["count"]
        )
        if not np.isclose(ece, archived["ece_10_equal_width"], rtol=1e-12, atol=1e-12):
            raise ValueError("Saved ECE mismatch")
        rows.append({"model": name, "ece_10_equal_width": ece, "bins": bins})
    return {
        "dataset": dataset,
        "split_seed": split,
        "training_seed": seed,
        "unique_test_cases": len(y),
        "positive_cases": int(y.sum()),
        "prediction_sha256": sha256(evidence / dataset / "predictions.npz"),
        "metrics_sha256": sha256(evidence / dataset / "metrics.json"),
        "models": rows,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    data = [summaries(root, name) for name in ("banknote", "ionosphere")]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#B7C3CD",
            "text.color": "#172F42",
            "axes.labelcolor": "#172F42",
            "xtick.color": "#526575",
            "ytick.color": "#526575",
        }
    )
    fig = plt.figure(figsize=(12.8, 8.0), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[3.5, 1], hspace=0.12, wspace=0.23)
    fig.subplots_adjust(top=0.745, bottom=0.15, left=0.075, right=0.97)
    fig.text(
        0.075,
        0.955,
        "PHASE 3  /  OBSERVED HELD-OUT PREDICTIONS",
        fontsize=10,
        color="#526575",
        weight="bold",
    )
    fig.text(0.075, 0.905, "Calibration on held-out observations", fontsize=22, weight="bold")
    fig.text(
        0.075,
        0.865,
        "One original fit per model; no pooling across repeated splits or seeds.",
        fontsize=11,
        color="#526575",
    )
    handles = []
    for col, result in enumerate(data):
        axis = fig.add_subplot(grid[0, col])
        counts = fig.add_subplot(grid[1, col], sharex=axis)
        (reference,) = axis.plot(
            [0, 1],
            [0, 1],
            color="#65727D",
            linestyle=":",
            linewidth=1.8,
            label="Perfect calibration",
            zorder=1,
        )
        for index, row in enumerate(result["models"]):
            populated = [b for b in row["bins"] if b["count"]]
            (line,) = axis.plot(
                [b["mean_probability"] for b in populated],
                [b["observed_rate"] for b in populated],
                color=COLORS[index],
                marker=MARKERS[index],
                linewidth=2.0,
                markersize=5.5,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=LABELS[index],
                zorder=3,
            )
            counts.stairs(
                [b["count"] for b in row["bins"]],
                np.linspace(0, 1, BINS + 1),
                color=COLORS[index],
                linewidth=1.6,
            )
            if col == 0:
                handles.append(line)
        title = "Banknote Authentication" if col == 0 else "Ionosphere"
        axis.set_title(
            f"{title}  |  n = {result['unique_test_cases']}",
            loc="left",
            fontsize=12,
            weight="bold",
            pad=12,
        )
        axis.set(xlim=(-0.02, 1.02), ylim=(-0.035, 1.035), yticks=np.linspace(0, 1, 6))
        axis.grid(color="#E4EBF0", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.tick_params(labelbottom=False)
        counts.set(
            xlabel="Mean predicted probability of class 1",
            ylim=(0, None),
            xticks=np.linspace(0, 1, 6),
        )
        counts.grid(axis="y", color="#E4EBF0", linewidth=0.7)
        counts.set_axisbelow(True)
        if col == 0:
            axis.set_ylabel("Observed fraction of class 1")
            counts.set_ylabel("Cases / bin")
    fig.legend(
        handles + [reference],
        LABELS + ("Perfect calibration",),
        loc="upper left",
        bbox_to_anchor=(0.068, 0.835),
        ncol=4,
        frameon=False,
        fontsize=9.4,
        handlelength=2.3,
        columnspacing=1.6,
    )
    fig.text(
        0.075,
        0.075,
        "10 fixed equal-width bins; p = 1 belongs to the last bin. Empty bins omitted; lines are visual guides.",
        fontsize=9,
        color="#526575",
    )
    fig.text(
        0.075,
        0.048,
        "Split seed 1701; training seed 101. Descriptive diagnostics only: no uncertainty bands or certification.",
        fontsize=9,
        color="#526575",
    )
    directory = root / "figures"
    directory.mkdir(exist_ok=True)
    output = directory / "phase3_calibration.png"
    fig.savefig(output, dpi=220, facecolor="white", metadata={"Software": "Matplotlib"})
    plt.close(fig)
    metadata = {
        "method": "expit(saved observed scores); floor(p*10), capped at 9; empty bins omitted",
        "selection": "first configured split and first training seed; no model refitting or pooling",
        "datasets": data,
    }
    (directory / "phase3_calibration.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(output)
    for result in data:
        print(
            result["dataset"],
            result["unique_test_cases"],
            {r["model"]: round(r["ece_10_equal_width"], 8) for r in result["models"]},
        )


if __name__ == "__main__":
    main()
