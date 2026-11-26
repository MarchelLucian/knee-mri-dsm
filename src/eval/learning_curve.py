"""Learning curves: training and tuning loss per epoch, with the selected epoch (lowest tuning loss) marked.

Reads the log.csv written by src/model/train.py or src/model/train_elnet.py. One panel per run.

Usage:
    python -m src.eval.learning_curve D:/CD/runs/elnet_acl_axial_seed42 [more run folders] --out results/figures/learning_curves.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+", help="run folders that contain log.csv")
    parser.add_argument("--out", type=Path, default=Path("results/figures/learning_curves.png"))
    args = parser.parse_args()

    fig, axes = plt.subplots(1, len(args.runs), figsize=(3.6 * len(args.runs), 3.0), squeeze=False)
    for ax, run in zip(axes[0], args.runs):
        log = pd.read_csv(run / "log.csv")
        best = log.loc[log["tuning_loss"].idxmin()]
        ax.plot(log["epoch"], log["train_loss"], lw=1, color="0.55", label="training")
        ax.plot(log["epoch"], log["tuning_loss"], lw=1, color="tab:blue", label="tuning")
        ax.axvline(best["epoch"], color="tab:blue", ls="--", lw=0.8)
        right = best["epoch"] > 0.6 * log["epoch"].max()  # put the label on the side with more room
        ax.text(best["epoch"] + (-3 if right else 3), 0.97,
                f"selected epoch {int(best['epoch'])}\ntuning AUC {best['tuning_auc']:.3f}",
                transform=ax.get_xaxis_transform(), va="top", ha="right" if right else "left",
                fontsize=7, color="tab:blue")
        parts = run.name.split("_")
        if parts[0] != "elnet":  # MRNet runs are named <task>_<plane>_seed<n>
            parts = ["mrnet"] + parts
        model, task, plane = parts[:3]
        ax.set_title(f"{'ELNet' if model == 'elnet' else 'MRNet'}, {task} ({plane} plane)", fontsize=8)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Weighted BCE loss", fontsize=8)
        ax.tick_params(labelsize=7)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
