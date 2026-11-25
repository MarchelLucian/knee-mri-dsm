"""Safety-net baseline: frozen ImageNet AlexNet features + logistic regression, one model per task and plane.

The features (computed by src/model/frozen_features.py with the same normalisation as MRNet, B) are the
MRNet series features of an AlexNet that is not trained further: per slice convolutional features,
global average pooling, maximum over slices (256 values). For each task and plane, a logistic
regression (standardised features, class_weight='balanced') is fitted on the train set; C is chosen by
5-fold stratified cross-validation on the train set (by AUC). Writes per-plane predictions for the
train set (out-of-fold, from the cross-validation with the chosen C), the tuning set and the test set.
The test predictions are only saved here; metrics on the test set are computed once, at the end.

Usage:
    python -m src.model.safety_net --features /mnt/d/knee-mri/features/alexnet_frozen_B.npz --out results/safety_net_predictions.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MRNET_DIR, PLANES, SEED, SPLITS_CSV, TASKS
from src.model.dataset import load_table

C_GRID = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def model_for(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    table = load_table(MRNET_DIR, SPLITS_CSV)
    z = np.load(args.features, allow_pickle=True)
    if list(z["exam_id"]) != list(table["exam_id"]) or list(z["planes"]) != PLANES:
        raise ValueError("feature file does not match the split table or the plane order")
    feats = z["features"]
    masks = {s: (table["split"] == s).to_numpy() for s in ["train", "tuning", "test"]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows, chosen = [], []
    for task in TASKS:
        y_train = table.loc[masks["train"], task].to_numpy()
        for j, plane in enumerate(PLANES):
            x_train = feats[masks["train"], j]
            scores = {c: cross_val_score(model_for(c), x_train, y_train, cv=cv, scoring="roc_auc").mean()
                      for c in C_GRID}
            best_c = max(scores, key=scores.get)
            chosen.append({"task": task, "plane": plane, "C": best_c, "cv_auc": round(scores[best_c], 4)})
            oof = cross_val_predict(model_for(best_c), x_train, y_train, cv=cv, method="predict_proba")[:, 1]
            final = model_for(best_c).fit(x_train, y_train)
            preds = {"train": oof,
                     "tuning": final.predict_proba(feats[masks["tuning"], j])[:, 1],
                     "test": final.predict_proba(feats[masks["test"], j])[:, 1]}
            for split, p in preds.items():
                part = table.loc[masks[split]]
                rows += [{"exam_id": e, "split": split, "task": task, "plane": plane, "prob": pi,
                          "label": int(yi), "seed": SEED} for e, pi, yi in zip(part["exam_id"], p, part[task])]
            print(f"{task:9s} {plane:9s} C={best_c:g}  CV AUC on train {scores[best_c]:.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False, float_format="%.6f")
    pd.DataFrame(chosen).to_csv(args.out.with_name(args.out.stem + "_choices.csv"), index=False)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
