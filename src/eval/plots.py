"""ROC curves and the table of results, from prediction files in the common format.

Each model is given as NAME=FILE. Only the rows of the chosen split and plane are used (by default the
combined prediction of the three planes). The figure has one panel per task, with the models drawn on top of
each other; the table has one row of metrics (src/eval/metrics.py) per model and task.

Usage:
    python -m src.eval.plots --split tuning --models "MRNet=/path/mrnet_fused.csv" "Frozen AlexNet=/path/frozen_fused.csv" \\
        --figure results/figures/roc_tuning.png --table results/tables/results_tuning.csv
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import TASKS
from src.eval.metrics import evaluate_predictions

# Each model keeps the same colour in every figure (same palette as the data figure).
# Any other model, for example a trivial baseline, is drawn as a grey dashed reference line.
MODEL_COLORS = {"MRNet": "#2a78d6", "Frozen AlexNet": "#eb6834", "ELNet": "#1baf7a"}
REFERENCE_COLOR = "#8c959f"
TASK_TITLES = {"abnormal": "Abnormal", "acl": "ACL tear", "meniscus": "Meniscal tear"}
INK, MUTED, GRID = "#1f2328", "#57606a", "#e6e5e1"


def load(models, split, plane="combined"):
    """{name: predictions of one split and plane}, from strings of the form NAME=FILE."""
    preds = {}
    for item in models:
        name, path = item.split("=", 1)
        df = pd.read_csv(path, dtype={"exam_id": str})
        part = df[(df["split"] == split) & (df["plane"] == plane)]
        if part.empty:
            raise ValueError(f"{path} has no rows for split {split} and plane {plane}")
        preds[name] = part
    return preds


def results_table(preds):
    """Metrics per model and task: AUC with 95% CI, sensitivity, specificity and accuracy at 0.5."""
    tables = []
    for name, df in preds.items():
        table = evaluate_predictions(df)
        table.insert(0, "model", name)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def roc_figure(preds, out_path):
    """One ROC panel per task; the AUC of each model is given in the legend."""
    fig, axes = plt.subplots(1, len(TASKS), figsize=(7.2, 2.6), sharey=True)
    for ax, task in zip(axes, TASKS):
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1, linestyle="--")  # chance level
        for name, df in preds.items():
            part = df[df["task"] == task]
            if part.empty:
                continue
            fpr, tpr, _ = roc_curve(part["label"], part["prob"])
            auc = roc_auc_score(part["label"], part["prob"])
            color = MODEL_COLORS.get(name, REFERENCE_COLOR)
            style = "-" if name in MODEL_COLORS else ":"
            ax.plot(fpr, tpr, color=color, linewidth=1.8, linestyle=style, label=f"{name} ({auc:.3f})")
        ax.set_title(TASK_TITLES.get(task, task), fontsize=9, color=INK)
        ax.set_xlabel("1 - specificity", fontsize=8, color=MUTED)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.grid(color=GRID, linewidth=0.5)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.legend(loc="lower right", fontsize=6.5, frameon=False, labelcolor=INK)
    axes[0].set_ylabel("Sensitivity", fontsize=8, color=MUTED)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", required=True, help="NAME=FILE pairs")
    parser.add_argument("--split", choices=["tuning", "test"], required=True)
    parser.add_argument("--plane", default="combined")
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()

    preds = load(args.models, args.split, args.plane)
    table = results_table(preds)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.table, index=False, float_format="%.4f")
    roc_figure(preds, args.figure)
    print(table[["model", "task", "n", "n_pos", "auc", "auc_low", "auc_high"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"written: {args.table}, {args.figure}")


if __name__ == "__main__":
    main()
