"""Trivial baselines, used as lower bounds for the results.

- majority: every exam gets the share of positive exams in the train set as its probability. The AUC is 0.5
  by construction, and at threshold 0.5 every exam is predicted as the most common class.
- slices: logistic regression on the number of slices of the three series, fitted on the train set. It shows
  how much of a label can be guessed from the length of the scan alone, without looking at the images.

Output: the common prediction format (plane = "combined", plus split and method) for the tuning and test sets.
It contains the labels, so it is written outside the repository.

Usage:
    python -m src.eval.trivial_baseline --kind majority --out /path/runs/trivial_majority.csv
    python -m src.eval.trivial_baseline --kind slices --out /path/runs/trivial_slices.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MRNET_DIR, PLANES, SEED, SPLITS_CSV, TABLES_DIR, TASKS
from src.model.dataset import load_table
from src.model.fuse_planes import check_outside_repo

KINDS = ["majority", "slices"]


def slice_counts(checks_csv):
    """Number of slices of each series: one row per exam, one column per plane (from the data checks table)."""
    checks = pd.read_csv(checks_csv, dtype={"exam_id": str})
    return checks.pivot(index="exam_id", columns="plane", values="slices")[PLANES]


def scores(kind, train, part, task, counts=None):
    """Probabilities of one task for the exams in part, learned from the train exams only."""
    y_train = train[task].to_numpy()
    if kind == "majority":
        return [float(y_train.mean())] * len(part)
    if kind == "slices":
        model = make_pipeline(StandardScaler(), LogisticRegression())
        model.fit(counts.loc[train["exam_id"]].to_numpy(), y_train)
        return model.predict_proba(counts.loc[part["exam_id"]].to_numpy())[:, 1]
    raise ValueError(f"unknown kind {kind}; choose one of {KINDS}")


def predict(table, kind, counts=None, seed=SEED):
    """Predictions for the tuning and test sets. table has one row per exam: exam_id, split and the labels."""
    train = table[table["split"] == "train"]
    rows = []
    for task in TASKS:
        for split in ["tuning", "test"]:
            part = table[table["split"] == split]
            probs = scores(kind, train, part, task, counts)
            rows += [{"exam_id": e, "task": task, "plane": "combined", "prob": float(p), "label": int(y),
                      "seed": seed, "split": split, "method": kind}
                     for e, p, y in zip(part["exam_id"], probs, part[task])]
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checks", type=Path, default=TABLES_DIR / "data_checks.csv",
                        help="table with the number of slices of each series (from src/data/check_data.py)")
    args = parser.parse_args()

    check_outside_repo(args.out)
    table = load_table(MRNET_DIR, SPLITS_CSV)
    counts = slice_counts(args.checks) if args.kind == "slices" else None
    out = predict(table, args.kind, counts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.6f")
    print(out.groupby(["task", "split"]).size().to_string())
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
