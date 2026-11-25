"""Exploratory analysis of MRNet: label counts per split, label co-occurrence, slices per exam, intensity ranges and example images.

Outputs:
- results/tables/eda_splits.csv: exams and positive exams (with %) for each split and task
- results/tables/eda_label_combinations.csv: exams per label combination and split
- results/tables/eda_slices.csv: number of slices per plane and split
- results/tables/eda_slice_count_auc.csv: AUC of the number of slices alone as a predictor of each
  label, on the train and tuning sets (a check for a possible shortcut: a model could use the
  length of the exam, not the lesion); the test set is not used here
- results/tables/eda_labels_by_slices.csv and results/figures/labels_by_slice_count_train.png:
  share of train exams with each label, by groups of the number of coronal slices
- <figures>/example_exam_planes.png: middle slice of one exam in the three planes
  (this figure shows images from the dataset; by default it is written outside the repository)

Intensities are summarised by src/data/check_data.py (results/tables/data_checks_summary.csv).

Usage:
    python -m src.data.eda [--image-figures /path/outside/the/repo]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from src.config import FIGURES_DIR, MRNET_DIR, PLANES, SPLITS_CSV, TABLES_DIR, TASKS
from src.data.check_data import read_labels
from src.data.make_splits import label_combination

SPLIT_ORDER = ["train", "tuning", "test", "excluded"]
COMBINATION_NAMES = {
    "000": "normal",
    "100": "abnormal, no ACL or meniscus tear",
    "101": "abnormal + meniscus tear",
    "110": "abnormal + ACL tear",
    "111": "abnormal + ACL and meniscus tear",
}
FOLDER = {"train": "train", "tuning": "train", "excluded": "train", "test": "valid"}
# Series used for each plane, as described in Bien et al. (2018), p. 4.
SEQUENCE = {"sagittal": "T2-weighted", "coronal": "T1-weighted", "axial": "PD-weighted"}


def load(data_dir, splits_csv):
    """One row per exam: exam_id, split, abnormal, acl, meniscus."""
    labels = pd.concat([read_labels(data_dir, "train"), read_labels(data_dir, "valid")])
    splits = pd.read_csv(splits_csv, dtype={"exam_id": str})
    df = splits.merge(labels, on="exam_id", how="left", validate="one_to_one")
    if df[TASKS].isna().any().any():
        raise ValueError("some exams in splits.csv have no labels")
    return df


def split_table(df):
    """Exams and positives per split, plus the official train set (train + tuning + excluded)."""
    rows = []
    groups = [(s, df[df["split"] == s]) for s in SPLIT_ORDER]
    groups.append(("official train (all)", df[df["split"] != "test"]))
    for name, part in groups:
        row = {"split": name, "exams": len(part)}
        for task in TASKS:
            n = int(part[task].sum())
            row[f"{task}_n"] = n
            row[f"{task}_pct"] = round(100 * n / len(part), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def combination_table(df):
    """Exams per label combination (abnormal, ACL, meniscus) and split."""
    table = pd.crosstab(label_combination(df), df["split"])[SPLIT_ORDER]
    table.insert(0, "meaning", [COMBINATION_NAMES.get(k, "") for k in table.index])
    table.index.name = "abnormal_acl_meniscus"
    return table


def slice_counts(data_dir, df):
    """Number of slices of every series, read from the .npy headers only."""
    rows = []
    for _, r in df.iterrows():
        for plane in PLANES:
            vol = np.load(data_dir / FOLDER[r["split"]] / plane / f"{r['exam_id']}.npy", mmap_mode="r")
            rows.append({"exam_id": r["exam_id"], "split": r["split"], "plane": plane, "slices": vol.shape[0]})
    return pd.DataFrame(rows)


def slice_table(slices):
    used = slices[slices["split"] != "excluded"]
    table = used.groupby(["plane", "split"])["slices"].agg(["count", "mean", "std", "min", "max"]).round(2)
    overall = used.groupby("plane")["slices"].agg(["count", "mean", "std", "min", "max"]).round(2)
    overall["split"] = "all used"
    return pd.concat([table.reset_index(), overall.reset_index()])


def slice_count_auc(slices, labels):
    """AUC of the number of slices alone, for each split, plane and label, on the train and
    tuning sets only (the test set is kept for the final results).
    0.5 = no relation; above 0.5 = positive exams have more slices; below 0.5 = fewer."""
    wide = slices.pivot_table(index="exam_id", columns="plane", values="slices").reset_index()
    df = labels.merge(wide, on="exam_id")
    rows = []
    for split in ["train", "tuning"]:
        part = df[df["split"] == split]
        for task in TASKS:
            row = {"split": split, "task": task, "exams": len(part), "positives": int(part[task].sum())}
            for plane in PLANES:
                row[f"auc_{plane}"] = round(roc_auc_score(part[task], part[plane]), 3)
            rows.append(row)
    return pd.DataFrame(rows)


# Groups of the number of slices in the coronal series, used in the figure below.
SLICE_GROUPS = [(17, 20), (21, 24), (25, 28), (29, 32), (33, 36), (37, 40), (41, 58)]


def labels_by_slices(slices, labels, split="train", plane="coronal"):
    """Share of exams with each label (%) for groups of the number of slices of one plane."""
    wide = slices[slices["plane"] == plane][["exam_id", "slices"]]
    df = labels[labels["split"] == split].merge(wide, on="exam_id")
    if df["slices"].min() < SLICE_GROUPS[0][0] or df["slices"].max() > SLICE_GROUPS[-1][1]:
        raise ValueError("slice counts outside the groups")
    rows = []
    for lo, hi in SLICE_GROUPS:
        part = df[(df["slices"] >= lo) & (df["slices"] <= hi)]
        row = {"group": f"{lo}-{hi}", "exams": len(part)}
        for task in TASKS:
            row[f"{task}_pct"] = round(100 * part[task].mean(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_labels_by_slices(table, n_exams, out_path):
    """One-column figure (3.5 x 2.0 in): % of exams with each label per group of coronal slice count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Colours: first three categorical slots of a colour-blind-safe palette; the marker shapes
    # and the direct labels keep the series apart in grayscale print.
    series = [("abnormal", "abnormal", "#2a78d6", "o"), ("meniscus", "meniscus tear", "#eb6834", "s"),
              ("acl", "ACL tear", "#1baf7a", "^")]
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    for task, name, color, marker in series:
        y = table[f"{task}_pct"].values
        ax.plot(x, y, color=color, linewidth=2, marker=marker, markersize=4.5, markeredgecolor="white")
        ax.text(x[-1] + 0.15, y[-1], name, va="center", fontsize=6.5)
    ax.set_xticks(x, [f"{g.replace('-', '–')}\nn={n}" for g, n in zip(table["group"], table["exams"])], fontsize=6)
    ax.set_ylabel("exams with the label (%)", fontsize=7.5)
    ax.set_xlabel(f"slices in the coronal series\n(train set, {n_exams} exams; n = number of exams in the group)",
                  fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.4, len(x) + 1.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_example(data_dir, df, out_path):
    """Middle slice of one train exam in the three planes. The title is built from the exam's labels.
    The middle slice shows the plane; it does not necessarily show the lesion."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row = df[(df["split"] == "train") & (label_combination(df) == "110")].sort_values("exam_id").iloc[0]
    exam = row["exam_id"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.3))
    for ax, plane in zip(axes, PLANES):
        vol = np.load(data_dir / FOLDER[row["split"]] / plane / f"{exam}.npy")
        ax.imshow(vol[vol.shape[0] // 2], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{plane}, {SEQUENCE[plane]} (slice {vol.shape[0] // 2 + 1} of {vol.shape[0]})", fontsize=9)
        ax.axis("off")
    yes_no = {1: "yes", 0: "no"}
    fig.suptitle(f"MRNet exam {exam}, middle slices. Labels: abnormal {yes_no[int(row['abnormal'])]}, "
                 f"ACL tear {yes_no[int(row['acl'])]}, meniscus tear {yes_no[int(row['meniscus'])]}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return exam


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=MRNET_DIR)
    parser.add_argument("--splits", type=Path, default=SPLITS_CSV)
    parser.add_argument("--tables", type=Path, default=TABLES_DIR)
    parser.add_argument("--figures", type=Path, default=FIGURES_DIR)
    parser.add_argument("--image-figures", type=Path, default=FIGURES_DIR,
                        help="folder for figures that show images from the dataset")
    args = parser.parse_args()
    for folder in [args.tables, args.figures, args.image_figures]:
        folder.mkdir(parents=True, exist_ok=True)

    df = load(args.data, args.splits)

    splits = split_table(df)
    splits.to_csv(args.tables / "eda_splits.csv", index=False)
    print(splits.to_string(index=False), "\n")

    combos = combination_table(df)
    combos.to_csv(args.tables / "eda_label_combinations.csv")
    print(combos.to_string(), "\n")

    slices = slice_counts(args.data, df)
    stable = slice_table(slices)
    stable.to_csv(args.tables / "eda_slices.csv", index=False)
    print(stable.to_string(index=False), "\n")

    auc = slice_count_auc(slices, df)
    auc.to_csv(args.tables / "eda_slice_count_auc.csv", index=False)
    print(auc.to_string(index=False), "\n")

    by_slices = labels_by_slices(slices, df)
    by_slices.to_csv(args.tables / "eda_labels_by_slices.csv", index=False)
    print(by_slices.to_string(index=False), "\n")
    n_train = int((df["split"] == "train").sum())
    plot_labels_by_slices(by_slices, n_train, args.figures / "labels_by_slice_count_train.png")

    exam = plot_example(args.data, df, args.image_figures / "example_exam_planes.png")
    print(f"figures: {args.figures / 'labels_by_slice_count_train.png'}, "
          f"{args.image_figures / 'example_exam_planes.png'} (exam {exam})")


if __name__ == "__main__":
    main()
