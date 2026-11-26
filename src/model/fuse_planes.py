"""Combine the three plane predictions of each task into one probability per exam.

Methods:
- lr_train: logistic regression fitted on the train-set predictions, as in Bien et al. (2018), then applied to
  the tuning and test sets. For fine-tuned networks the train predictions come from exams the network was
  trained on.
- lr_tuning: logistic regression fitted on the tuning-set predictions. The tuning predictions come from a
  5-fold stratified cross-validation inside the tuning set (seed 42), so no exam is scored by a model that saw
  it. The test predictions come from the model fitted on the whole tuning set.
- mean: average of the three probabilities. Nothing is fitted.

Input: one or more prediction files with the columns exam_id, split, task, plane, prob and label.
Output: the common prediction format (CONVENTIONS.md) with plane = "combined", for the tuning and test sets,
plus the column method. Predictions contain the labels, so they are written outside the repository.

Usage:
    python -m src.model.fuse_planes --predictions /path/runs/*_seed42/predictions.csv --method lr_train --out /path/runs/mrnet_fused.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import PLANES, ROOT, SEED, TASKS

METHODS = ["lr_train", "lr_tuning", "mean"]
SPLITS_OUT = ["tuning", "test"]


def load_predictions(paths):
    """Plane predictions from one or more CSV files. exam_id is read as text to keep the leading zeros."""
    df = pd.concat([pd.read_csv(p, dtype={"exam_id": str}) for p in paths], ignore_index=True)
    missing = {"exam_id", "split", "task", "plane", "prob", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return df


def wide(df, task, split):
    """Probabilities of one task and split as a table: one row per exam, one column per plane.

    Returns (probabilities, labels), both sorted by exam_id. Every exam must have all three planes,
    and the three planes must agree on the label.
    """
    part = df[(df["task"] == task) & (df["split"] == split) & df["plane"].isin(PLANES)]
    if part.duplicated(["exam_id", "plane"]).any():
        raise ValueError(f"{task}, {split}: an exam has two predictions for the same plane")
    probs = part.pivot(index="exam_id", columns="plane", values="prob").sort_index()
    if list(sorted(probs.columns)) != sorted(PLANES) or probs.isna().any().any():
        raise ValueError(f"{task}, {split}: not every exam has the three planes")
    labels = part.groupby("exam_id")["label"].agg(["min", "max"]).loc[probs.index]
    if (labels["min"] != labels["max"]).any():
        raise ValueError(f"{task}, {split}: the planes of an exam have different labels")
    return probs[PLANES], labels["min"].astype(int)


def logistic():
    # scikit-learn defaults, as in Bien et al. (2018), who give no other setting.
    return LogisticRegression()


def fuse_task(x, y, method, seed=SEED):
    """Combined probabilities of one task.

    x and y are dicts keyed by split ("train", "tuning", "test") with the outputs of wide().
    Returns (dict of probabilities for the tuning and test sets, fitted weights or None).
    """
    present = [s for s in SPLITS_OUT if s in x]
    if method == "mean":
        return {s: x[s].mean(axis=1).to_numpy() for s in present}, None

    if method == "lr_train":
        model = logistic().fit(x["train"].to_numpy(), y["train"].to_numpy())
        return {s: model.predict_proba(x[s].to_numpy())[:, 1] for s in present}, model

    if method == "lr_tuning":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        out = {"tuning": cross_val_predict(logistic(), x["tuning"].to_numpy(), y["tuning"].to_numpy(),
                                           cv=cv, method="predict_proba")[:, 1]}
        model = logistic().fit(x["tuning"].to_numpy(), y["tuning"].to_numpy())
        if "test" in x:
            out["test"] = model.predict_proba(x["test"].to_numpy())[:, 1]
        return out, model

    raise ValueError(f"unknown method {method}; choose one of {METHODS}")


def fuse(df, method, seed=SEED):
    """Combined predictions for every task in df. Returns (predictions, weights of the fitted models)."""
    needed = {"lr_train": "train", "lr_tuning": "tuning", "mean": None}[method]
    rows, weights = [], []
    for task in [t for t in TASKS if t in set(df["task"])]:
        splits = [s for s in ["train"] + SPLITS_OUT if ((df["task"] == task) & (df["split"] == s)).any()]
        tables = {s: wide(df, task, s) for s in splits}
        if needed and needed not in tables:
            raise ValueError(f"{task}: method {method} needs predictions on the {needed} set")
        x = {s: t[0] for s, t in tables.items()}
        y = {s: t[1] for s, t in tables.items()}
        probs, model = fuse_task(x, y, method, seed)
        for split, p in probs.items():
            rows += [{"exam_id": e, "task": task, "plane": "combined", "prob": float(pi), "label": int(yi),
                      "seed": seed, "split": split, "method": method}
                     for e, pi, yi in zip(x[split].index, p, y[split])]
        if model is not None:
            weights.append({"task": task, "method": method, "intercept": float(model.intercept_[0]),
                            **{plane: float(c) for plane, c in zip(PLANES, model.coef_[0])}})
    return pd.DataFrame(rows), pd.DataFrame(weights)


def check_outside_repo(path):
    """Predictions contain the MRNet labels, which may not be redistributed: keep them out of the repository."""
    path = Path(path).resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError(f"{path} is inside the repository; write predictions to a folder outside it")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    check_outside_repo(args.out)
    fused, weights = fuse(load_predictions(args.predictions), args.method)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fused.to_csv(args.out, index=False, float_format="%.6f")
    if len(weights):
        weights.to_csv(args.out.with_name(args.out.stem + "_weights.csv"), index=False, float_format="%.4f")

    # Only counts are printed here; metrics are computed with src/eval/metrics.py.
    print(fused.groupby(["task", "split"]).size().to_string())
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
