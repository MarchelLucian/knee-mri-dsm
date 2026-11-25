"""AUC with 95% confidence intervals, accuracy, sensitivity and specificity for each task.

- AUC: 95% CI from a stratified percentile bootstrap (positives and negatives are
  resampled separately, 2,000 resamples, seed 42). Resampling is done over exams.
  With about 120 exams, the real coverage of this interval is a little below 95%
  when the AUC is above 0.95.
- Sensitivity, specificity and accuracy at threshold 0.5: 95% Wilson score intervals,
  as in Bien et al. (2018).

Usage:
    python -m src.eval.metrics --predictions results/predictions.csv --out results/tables/metrics.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import PREDICTIONS_CSV, SEED, TABLES_DIR

N_BOOTSTRAP = 2000
THRESHOLD = 0.5
Z_95 = 1.959964  # standard normal quantile for a two-sided 95% interval


def check_inputs(labels, probs):
    """Return labels and probabilities as arrays; stop on values that would silently give wrong metrics."""
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if labels.shape != probs.shape or labels.ndim != 1:
        raise ValueError("labels and probabilities must be 1-D arrays of the same length.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be 0 or 1 (no missing values).")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("probabilities must be numbers between 0 and 1 (no missing values).")
    return labels.astype(int), probs


def auc_bootstrap_ci(labels, probs, n_bootstrap=N_BOOTSTRAP, seed=SEED, alpha=0.05):
    """Return (auc, low, high). Positives and negatives are resampled separately,
    so every resample has the same number of each class and the AUC is always defined."""
    labels, probs = check_inputs(labels, probs)
    pos = probs[labels == 1]
    neg = probs[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUC needs at least one positive and one negative exam.")

    auc = roc_auc_score(labels, probs)
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    aucs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        p = rng.choice(pos, size=len(pos), replace=True)
        n = rng.choice(neg, size=len(neg), replace=True)
        aucs[i] = roc_auc_score(y, np.r_[p, n])
    low, high = np.quantile(aucs, [alpha / 2, 1 - alpha / 2])
    return auc, low, high


def paired_auc_difference(labels, probs_a, probs_b, n_bootstrap=N_BOOTSTRAP, seed=SEED, alpha=0.05):
    """AUC(a) - AUC(b) on the same exams, with a stratified percentile bootstrap interval.
    The same resampled exams are used for both models in every resample. Returns (delta, low, high)."""
    labels, probs_a = check_inputs(labels, probs_a)
    _, probs_b = check_inputs(labels, probs_b)
    pos, neg = np.where(labels == 1)[0], np.where(labels == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUC needs at least one positive and one negative exam.")
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        deltas[i] = roc_auc_score(labels[idx], probs_a[idx]) - roc_auc_score(labels[idx], probs_b[idx])
    delta = roc_auc_score(labels, probs_a) - roc_auc_score(labels, probs_b)
    low, high = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return delta, low, high


def wilson_ci(successes, total, z=Z_95):
    """Return (proportion, low, high) with the Wilson score interval."""
    if total == 0:
        return np.nan, np.nan, np.nan
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def threshold_metrics(labels, probs, threshold=THRESHOLD):
    """Sensitivity, specificity and accuracy at a fixed threshold, each as (value, low, high)."""
    labels, probs = check_inputs(labels, probs)
    pred = (probs >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    return {
        "sensitivity": wilson_ci(tp, n_pos),
        "specificity": wilson_ci(tn, n_neg),
        "accuracy": wilson_ci(tp + tn, n_pos + n_neg),
    }


def evaluate(labels, probs, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """All metrics for one set of predictions, as a flat dict."""
    labels, probs = check_inputs(labels, probs)
    row = {"n": len(labels), "n_pos": int(labels.sum())}
    row["auc"], row["auc_low"], row["auc_high"] = auc_bootstrap_ci(labels, probs, n_bootstrap, seed)
    for name, (value, low, high) in threshold_metrics(labels, probs).items():
        row[name], row[f"{name}_low"], row[f"{name}_high"] = value, low, high
    return row


def evaluate_predictions(df, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """One row of metrics per (task, plane, seed) in a predictions table
    with the columns exam_id, task, plane, prob, label, seed."""
    keys = ["task", "plane", "seed"]
    if df[keys + ["exam_id"]].isna().any().any():
        raise ValueError("exam_id, task, plane and seed must not be missing.")
    if df.duplicated(keys + ["exam_id"]).any():
        raise ValueError("each exam must appear once per task, plane and seed.")
    rows = []
    for (task, plane, run_seed), group in df.groupby(keys, sort=True):
        rows.append({"task": task, "plane": plane, "seed": run_seed,
                     **evaluate(group["label"], group["prob"], n_bootstrap, seed)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_CSV)
    parser.add_argument("--out", type=Path, default=TABLES_DIR / "metrics.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions, dtype={"exam_id": str})
    table = evaluate_predictions(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, float_format="%.4f")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
