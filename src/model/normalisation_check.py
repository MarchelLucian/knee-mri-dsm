"""Compare intensity normalisations with the frozen-feature baseline, on the tuning set only.

For each normalisation, task and plane: logistic regression on the frozen AlexNet features
(standardised; class_weight='balanced'; C chosen by 5-fold stratified cross-validation on the
train set, by AUC), fitted on train and evaluated on tuning. The test set is not used.
Also reports the paired difference A - B (same tuning exams) with a stratified bootstrap.

Usage:
    python -m src.model.normalisation_check --features /mnt/d/knee-mri/features
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MRNET_DIR, PLANES, SEED, SPLITS_CSV, TABLES_DIR, TASKS
from src.eval.metrics import auc_bootstrap_ci, paired_auc_difference
from src.model.dataset import load_table

C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]


def fit_predict(x_train, y_train, x_eval, seed=SEED):
    """Choose C by 5-fold CV AUC on train, refit on all train, return (probabilities on x_eval, C)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = {}
    for c in C_GRID:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000))
        scores[c] = cross_val_score(model, x_train, y_train, cv=cv, scoring="roc_auc").mean()
    best_c = max(scores, key=scores.get)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, class_weight="balanced", max_iter=5000))
    model.fit(x_train, y_train)
    return model.predict_proba(x_eval)[:, 1], best_c


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--norms", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--out", type=Path, default=TABLES_DIR / "normalisation_check_tuning.csv")
    args = parser.parse_args()

    table = load_table(MRNET_DIR, SPLITS_CSV)
    feats = {}
    for n in args.norms:
        z = np.load(args.features / f"alexnet_frozen_{n}.npz", allow_pickle=True)
        if list(z["exam_id"]) != list(table["exam_id"]):
            raise ValueError(f"exam order in features {n} differs from the split table")
        feats[n] = z["features"]
    train = (table["split"] == "train").to_numpy()
    tuning = (table["split"] == "tuning").to_numpy()

    rows, probs = [], {}
    for task in TASKS:
        y_train, y_tune = table.loc[train, task].to_numpy(), table.loc[tuning, task].to_numpy()
        for j, plane in enumerate(PLANES):
            for n in args.norms:
                p, c = fit_predict(feats[n][train, j], y_train, feats[n][tuning, j])
                probs[(task, plane, n)] = p
                auc, low, high = auc_bootstrap_ci(y_tune, p)
                rows.append({"task": task, "plane": plane, "norm": n, "C": c,
                             "tuning_auc": auc, "auc_low": low, "auc_high": high})
            if "A" in args.norms and "B" in args.norms:
                d, lo, hi = paired_auc_difference(y_tune, probs[(task, plane, "A")], probs[(task, plane, "B")])
                rows.append({"task": task, "plane": plane, "norm": "A-B", "C": np.nan,
                             "tuning_auc": d, "auc_low": lo, "auc_high": hi})
            print(f"{task:9s} {plane:9s} " + "  ".join(
                f"{n}={next(r['tuning_auc'] for r in rows[::-1] if r['norm'] == n and r['task'] == task and r['plane'] == plane):.3f}"
                for n in args.norms), flush=True)

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.4f")
    wide = out[out["norm"] != "A-B"].pivot_table(index=["task", "plane"], columns="norm", values="tuning_auc")
    print("\ntuning AUC (frozen AlexNet + logistic regression):\n", wide.round(3).to_string())
    delta = out[out["norm"] == "A-B"]
    print(f"\nA - B: mean {delta['tuning_auc'].mean():+.4f}; "
          f"paired 95% intervals excluding 0: {int(((delta['auc_low'] > 0) | (delta['auc_high'] < 0)).sum())} of {len(delta)}")
    print("mean tuning AUC per normalisation:", wide.mean().round(4).to_dict())


if __name__ == "__main__":
    main()
