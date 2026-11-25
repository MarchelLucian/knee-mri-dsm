"""Check the MRNet files: every exam has the three planes, the expected image size, no missing values and a label for each task.

For every .npy file: shape, data type, value range, number of slices, constant (empty) slices
and an MD5 hash (to find duplicate files). For the labels: one row per exam in each CSV,
the same exam IDs as the files, values 0/1, and ACL or meniscus tear implies abnormal.
It also saves, for each split and plane, a montage of the middle slice of the first exams,
to check the orientation by eye.

Usage:
    python -m src.data.check_data [--out results/tables] [--figures results/figures]
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MRNET_DIR, PLANES, TABLES_DIR, FIGURES_DIR, TASKS

SPLITS = ["train", "valid"]
EXPECTED = {"train": 1130, "valid": 120}
SLICE_SIZE = (256, 256)
SLICE_RANGE = (17, 61)  # reported in Bien et al. (2018), p. 4


def read_labels(data_dir, split):
    """One row per exam with the columns exam_id, abnormal, acl, meniscus."""
    labels = None
    for task in TASKS:
        df = pd.read_csv(data_dir / f"{split}-{task}.csv", header=None, names=["exam_id", task],
                         dtype={"exam_id": str})
        if df["exam_id"].duplicated().any():
            raise ValueError(f"{split}-{task}.csv has repeated exam IDs")
        labels = df if labels is None else labels.merge(df, on="exam_id", how="outer")
    return labels.sort_values("exam_id").reset_index(drop=True)


def check_file(path):
    """Statistics for one series (one exam, one plane)."""
    raw = path.read_bytes()
    vol = np.load(path)
    flat = vol.reshape(vol.shape[0], -1) if vol.ndim == 3 else None
    return {
        "slices": vol.shape[0] if vol.ndim == 3 else np.nan,
        "height": vol.shape[1] if vol.ndim == 3 else np.nan,
        "width": vol.shape[2] if vol.ndim == 3 else np.nan,
        "ndim": vol.ndim,
        "dtype": str(vol.dtype),
        "min": vol.min(),
        "max": vol.max(),
        "mean": round(float(vol.mean()), 2),
        "constant_slices": int((flat.min(axis=1) == flat.max(axis=1)).sum()) if flat is not None else np.nan,
        "md5": hashlib.md5(raw).hexdigest(),
    }


def montage(data_dir, split, plane, out_path, n=30, cols=10):
    """Middle slice of the first n exams of one split and plane, in one figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted((data_dir / split / plane).glob("*.npy"))[:n]
    rows = int(np.ceil(len(files) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.75))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, f in zip(axes.ravel(), files):
        vol = np.load(f)
        ax.imshow(vol[vol.shape[0] // 2], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{f.stem} ({vol.shape[0]})", fontsize=7)
    fig.suptitle(f"MRNet {split} {plane}: middle slice (number of slices)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=MRNET_DIR)
    parser.add_argument("--out", type=Path, default=TABLES_DIR)
    parser.add_argument("--figures", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    problems = []
    rows = []
    for split in SPLITS:
        labels = read_labels(args.data, split)
        label_ids = set(labels["exam_id"])
        if labels[TASKS].isna().any().any():
            problems.append(f"{split}: an exam is missing from one of the label files")
        if not labels[TASKS].isin([0, 1]).all().all():
            problems.append(f"{split}: labels other than 0/1")
        bad = labels[((labels["acl"] == 1) | (labels["meniscus"] == 1)) & (labels["abnormal"] == 0)]
        if len(bad):
            problems.append(f"{split}: {len(bad)} exams with an ACL or meniscus tear but not abnormal")
        if len(labels) != EXPECTED[split]:
            problems.append(f"{split}: {len(labels)} exams in the labels, expected {EXPECTED[split]}")

        for plane in PLANES:
            files = sorted((args.data / split / plane).glob("*.npy"))
            file_ids = {f.stem for f in files}
            if file_ids != label_ids:
                problems.append(f"{split}/{plane}: file IDs differ from label IDs "
                                f"({len(file_ids - label_ids)} extra, {len(label_ids - file_ids)} missing)")
            for f in files:
                rows.append({"split": split, "plane": plane, "exam_id": f.stem, **check_file(f)})
            montage(args.data, split, plane, args.figures / f"orientation_{split}_{plane}.png")
        print(f"{split}: {len(labels)} exams checked")

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "data_checks.csv", index=False)

    wrong_shape = df[(df["ndim"] != 3) | (df["height"] != SLICE_SIZE[0]) | (df["width"] != SLICE_SIZE[1])]
    if len(wrong_shape):
        problems.append(f"{len(wrong_shape)} files without the shape (slices, 256, 256)")
    if set(df["dtype"]) != {"uint8"}:
        problems.append(f"data types found: {sorted(set(df['dtype']))}")
    out_of_range = df[(df["slices"] < SLICE_RANGE[0]) | (df["slices"] > SLICE_RANGE[1])]
    if len(out_of_range):
        problems.append(f"{len(out_of_range)} series with a number of slices outside {SLICE_RANGE}")
    dup = df[df["md5"].duplicated(keep=False)]
    if len(dup):
        problems.append(f"{len(dup)} files are exact copies of another file")
    if (df["constant_slices"] > 0).any():
        problems.append(f"{int((df['constant_slices'] > 0).sum())} series contain constant (empty) slices")

    summary = df.groupby(["split", "plane"]).agg(
        files=("exam_id", "size"), slices_min=("slices", "min"), slices_max=("slices", "max"),
        slices_mean=("slices", "mean"), slices_sd=("slices", "std"),
        value_min=("min", "min"), value_max=("max", "max"), mean_intensity=("mean", "mean"),
        series_max_below_255=("max", lambda s: int((s < 255).sum())))
    summary.round(2).to_csv(args.out / "data_checks_summary.csv")
    all_slices = df["slices"]
    print(summary.round(2).to_string())
    print(f"\nall series: {len(df)}, slices {all_slices.min()}-{all_slices.max()}, "
          f"mean {all_slices.mean():.2f}, SD {all_slices.std():.2f}")
    print(f"data types: {sorted(set(df['dtype']))}; values {df['min'].min()}-{df['max'].max()}")
    print("\nProblems:" if problems else "\nNo problems found.")
    for p in problems:
        print(" -", p)


if __name__ == "__main__":
    main()
