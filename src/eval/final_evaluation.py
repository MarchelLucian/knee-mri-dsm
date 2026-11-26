"""Evaluate all models on one split, with the same exams and the same code.

Models (prediction files in the common format, outside the repository):
- MRNet: the 3 planes combined by logistic regression fitted on the train predictions (main result),
  the mean of the 3 planes (robustness check) and each plane on its own;
- ELNet: one plane per task (axial for abnormal and ACL, coronal for meniscus);
- frozen ImageNet AlexNet + logistic regression, combined the same way as MRNet;
- trivial baselines: majority class and logistic regression on the number of slices.

Outputs in --out: the SHA-256 of every input file (written before any metric), the table of metrics
(AUC with 95% CI; sensitivity, specificity and accuracy at 0.5; TP, FP, FN, TN and F1 at 0.5), the paired AUC differences and the ROC figure.
The test split is meant to be evaluated once, after every choice has been fixed; try the script on tuning first.

Usage:
    python -m src.eval.final_evaluation --split tuning --final /path/final_eval --runs /path/runs \\
        --elnet /path/elnet_predictions --out /path/final_eval/tuning
    (the same with --split test, once, at the end)
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from src.config import SPLITS_CSV, TASKS
from src.eval.metrics import evaluate_predictions, paired_auc_difference
from src.eval.plots import roc_figure
from src.model.fuse_planes import check_outside_repo



def model_files(final, runs, elnet):
    """Prediction files of each model. final: fused and trivial files; runs: MRNet runs; elnet: ELNet files."""
    combined = {
        "MRNet": [final / "mrnet_fused_lr_train.csv"],
        "MRNet, mean of planes": [final / "mrnet_fused_mean.csv"],
        "ELNet": sorted(elnet.glob("*_predictions.csv")),
        "Frozen AlexNet": [final / "frozen_fused_lr_train.csv"],
        "Frozen AlexNet, mean of planes": [final / "frozen_fused_mean.csv"],
        "Trivial, majority class": [final / "trivial_majority.csv"],
        "Trivial, slice count": [final / "trivial_slices.csv"],
    }
    planes = [runs / f"{task}_{plane}_seed42" / "predictions.csv" for task in TASKS for plane in ["sagittal", "coronal", "axial"]]
    return combined, planes


# Paired AUC differences (first model minus second), per task. The first one is the main comparison.
PAIRS = [("MRNet", "ELNet"), ("MRNet", "Frozen AlexNet"), ("MRNet, same plane as ELNet", "ELNet")]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(paths, split):
    """Rows of one split from one or more prediction files."""
    df = pd.concat([pd.read_csv(p, dtype={"exam_id": str}) for p in paths], ignore_index=True)
    return df[df["split"] == split].reset_index(drop=True)


def check_exams(models, split):
    """Every model must score exactly the exams of the split, once per task, with the same labels."""
    splits = pd.read_csv(SPLITS_CSV, dtype={"exam_id": str})
    expected = set(splits.loc[splits["split"] == split, "exam_id"])
    reference = None
    for name, df in models.items():
        for task, part in df.groupby("task"):
            if set(part["exam_id"]) != expected or part["exam_id"].duplicated().any():
                raise ValueError(f"{name}, {task}: the exams differ from the {split} split")
        labels = df.set_index(["task", "exam_id"])["label"].sort_index()
        if reference is None:
            reference = labels
        elif not labels.equals(reference):
            raise ValueError(f"{name}: the labels differ from the other models")
    return len(expected)


def confusion_counts(df, threshold=0.5):
    """TP, FP, FN, TN at the threshold, plus F1, per task and plane (precision, F1 or MCC can be derived)."""
    rows = []
    for (task, plane), g in df.groupby(["task", "plane"]):
        pred, y = g["prob"] >= threshold, g["label"] == 1
        tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
        fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        rows.append({"task": task, "plane": plane, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": f1})
    return pd.DataFrame(rows)


def paired(models, task, name_a, name_b):
    """AUC(a) - AUC(b) for one task, after aligning the two models on exam_id."""
    a = models[name_a].query("task == @task")[["exam_id", "prob", "label"]]
    b = models[name_b].query("task == @task")[["exam_id", "prob"]]
    both = a.merge(b, on="exam_id", suffixes=("_a", "_b"), validate="one_to_one")
    assert len(both) == len(a) == len(b), "the two models must score the same exams"
    delta, low, high = paired_auc_difference(both["label"], both["prob_a"], both["prob_b"])
    return {"task": task, "model_a": name_a, "model_b": name_b, "n": len(both),
            "delta_auc": delta, "low": low, "high": high, "clear_difference": low > 0 or high < 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["tuning", "test"], required=True)
    parser.add_argument("--final", type=Path, required=True, help="folder with the fused and trivial prediction files")
    parser.add_argument("--runs", type=Path, required=True, help="folder with the MRNet runs (<task>_<plane>_seed42)")
    parser.add_argument("--elnet", type=Path, required=True, help="folder with the ELNet *_predictions.csv files")
    parser.add_argument("--out", type=Path, required=True, help="folder outside the repository")
    args = parser.parse_args()
    COMBINED, MRNET_PLANES = model_files(args.final, args.runs, args.elnet)
    check_outside_repo(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Fingerprints of the inputs, before any metric is computed.
    inputs = [p for paths in COMBINED.values() for p in paths] + MRNET_PLANES
    (args.out / "input_sha256.txt").write_text("".join(f"{sha256(p)}  {p}\n" for p in inputs))

    # 2. Predictions of the split; MRNet per plane is kept apart (one row per plane).
    models = {name: load(paths, args.split) for name, paths in COMBINED.items()}
    planes = load(MRNET_PLANES, args.split)
    elnet_plane = models["ELNet"].groupby("task")["plane"].first()
    models["MRNet, same plane as ELNet"] = pd.concat(
        [planes[(planes["task"] == t) & (planes["plane"] == p)] for t, p in elnet_plane.items()], ignore_index=True)
    each_plane = {f"MRNet, {p}": planes[planes["plane"] == p] for p in ["sagittal", "coronal", "axial"]}
    n = check_exams({**models, **each_plane}, args.split)

    # 3. Metrics: one row per model and task, plus MRNet on each plane.
    tables = []
    for name, df in {**models, "MRNet, one plane": planes}.items():
        table = evaluate_predictions(df).merge(confusion_counts(df), on=["task", "plane"])
        table.insert(0, "model", name)
        tables.append(table)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(args.out / f"results_{args.split}.csv", index=False, float_format="%.4f")

    # 4. Paired differences and the ROC figure.
    pairs = pd.DataFrame([paired(models, task, a, b) for a, b in PAIRS for task in TASKS])
    pairs.to_csv(args.out / f"paired_{args.split}.csv", index=False, float_format="%.4f")
    roc_figure({name: models[name] for name in ["MRNet", "ELNet", "Frozen AlexNet", "Trivial, slice count"]},
               args.out / f"roc_{args.split}.png")

    show = ["model", "task", "plane", "auc", "auc_low", "auc_high", "sensitivity", "specificity", "accuracy", "f1"]
    show = [c for c in show if c in table.columns]
    print(f"{args.split}: {n} exams per task\n")
    print(table[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(pairs.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"\nwritten in {args.out}")


if __name__ == "__main__":
    main()
