"""Split the official MRNet train set into train and tuning sets (120 tuning exams with the label composition of the official valid set, seed 42) and write data/splits.csv.

Each exam has one of a few label combinations (abnormal, ACL, meniscus), e.g. "110" = abnormal
with an ACL tear and no meniscus tear. For every combination, the tuning set gets as many exams
as the official valid set has, drawn at random from the official train set. So the tuning set has
the same label composition as the official valid set, which is our test set, and at least 50
positive exams for each label. The data has no patient IDs, so the split is made by exam.

Train exams with at least one series of severity "high" in data/series_flags.csv (a series that
does not show the knee, or shows another plane than its folder) are marked "excluded": they are
used neither for training nor for tuning. The official valid set (test) is kept unchanged.

Usage:
    python -m src.data.make_splits [--out data/splits.csv]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MRNET_DIR, SEED, SERIES_FLAGS_CSV, SPLITS_CSV, TASKS
from src.data.check_data import read_labels


def label_combination(labels):
    """'abnormal acl meniscus' as a string of 0/1, e.g. '110'."""
    return labels[TASKS].astype(int).astype(str).agg("".join, axis=1)


def draw_tuning(train_labels, quota, seed=SEED):
    """Exam IDs of the tuning set: for each label combination, quota[combination] exams
    drawn without replacement from the train exams with that combination."""
    rng = np.random.default_rng(seed)
    combination = label_combination(train_labels)
    tuning = []
    for key in sorted(quota.index):
        pool = sorted(train_labels.loc[combination == key, "exam_id"])
        if len(pool) < quota[key]:
            raise ValueError(f"only {len(pool)} train exams with labels {key}, {quota[key]} needed")
        tuning += [str(x) for x in rng.choice(pool, size=int(quota[key]), replace=False)]
    return sorted(tuning)


def excluded_exams(flags_csv=SERIES_FLAGS_CSV):
    """Train exams with at least one series of severity 'high'."""
    flags = pd.read_csv(flags_csv, dtype={"exam_id": str})
    high = flags[(flags["split"] == "train") & (flags["severity"] == "high")]
    return sorted(set(high["exam_id"]))


def make_splits(train_labels, valid_labels, excluded=(), seed=SEED):
    """Table with the columns exam_id and split (train, tuning, test or excluded)."""
    quota = label_combination(valid_labels).value_counts()
    pool = train_labels[~train_labels["exam_id"].isin(excluded)]
    tuning = set(draw_tuning(pool, quota, seed))
    train = pd.DataFrame({"exam_id": train_labels["exam_id"]})
    train["split"] = np.where(train["exam_id"].isin(tuning), "tuning", "train")
    train.loc[train["exam_id"].isin(excluded), "split"] = "excluded"
    test = pd.DataFrame({"exam_id": valid_labels["exam_id"], "split": "test"})
    return pd.concat([train, test]).sort_values("exam_id").reset_index(drop=True)


def describe(splits, labels):
    """Number of exams and of positive exams (with %) for each split and task."""
    df = splits.merge(labels, on="exam_id")
    rows = []
    for split in ["train", "tuning", "test", "excluded"]:
        part = df[df["split"] == split]
        if part.empty:
            continue
        row = {"split": split, "exams": len(part)}
        for task in TASKS:
            n = int(part[task].sum())
            row[task] = f"{n} ({100 * n / len(part):.1f}%)"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=MRNET_DIR)
    parser.add_argument("--out", type=Path, default=SPLITS_CSV)
    parser.add_argument("--flags", type=Path, default=SERIES_FLAGS_CSV)
    args = parser.parse_args()

    train_labels = read_labels(args.data, "train")
    valid_labels = read_labels(args.data, "valid")
    excluded = excluded_exams(args.flags)
    splits = make_splits(train_labels, valid_labels, excluded)

    # Checks: sizes, no exam in two sets, same label composition as the test set.
    labels = pd.concat([train_labels, valid_labels])
    assert splits["exam_id"].is_unique and len(splits) == len(labels)
    assert set(splits.loc[splits["split"] == "excluded", "exam_id"]) == set(excluded)
    assert (splits.loc[splits["split"] == "test", "exam_id"] == valid_labels["exam_id"].values).all()
    tuning = labels[labels["exam_id"].isin(splits.loc[splits["split"] == "tuning", "exam_id"])]
    assert (label_combination(tuning).value_counts().sort_index()
            == label_combination(valid_labels).value_counts().sort_index()).all()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(args.out, index=False)
    print(describe(splits, labels).to_string(index=False))
    print("\nlabel combinations (abnormal, acl, meniscus):")
    combo = pd.crosstab(label_combination(labels.set_index("exam_id").loc[splits["exam_id"]].reset_index()).values,
                        splits["split"].values)
    print(combo[[c for c in ["train", "tuning", "test", "excluded"] if c in combo.columns]].to_string())
    print(f"\nexcluded exams ({len(excluded)}): {', '.join(excluded)}")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
