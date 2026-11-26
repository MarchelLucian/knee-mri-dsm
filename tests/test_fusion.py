"""Checks for the plane fusion, the trivial baselines and the plots, on synthetic predictions.

Run from the repository root:
    python -m tests.test_fusion
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import PLANES, ROOT
from src.eval.plots import results_table, roc_figure
from src.eval.trivial_baseline import predict
from src.model.fuse_planes import check_outside_repo, fuse

SIZES = {"train": 200, "tuning": 60, "test": 40}


def plane_predictions(task="acl", noise=0.15, seed=0):
    """Synthetic plane predictions: each plane sees the label plus some noise."""
    rng = np.random.default_rng(seed)
    rows, start = [], 0
    for split, n in SIZES.items():
        labels = np.arange(n) % 2
        for i, y in enumerate(labels):
            exam_id = f"{start + i:04d}"
            for plane in PLANES:
                prob = float(np.clip(0.3 + 0.4 * y + rng.normal(0, noise), 0, 1))
                rows.append({"exam_id": exam_id, "split": split, "task": task, "plane": plane, "prob": prob,
                             "label": int(y)})
        start += n
    return pd.DataFrame(rows)


def test_mean_is_the_average_of_the_three_planes():
    df = plane_predictions()
    fused, weights = fuse(df, "mean")
    expected = df[df["split"] != "train"].groupby("exam_id")["prob"].mean()
    assert np.allclose(fused.set_index("exam_id")["prob"].loc[expected.index], expected)
    assert weights.empty and set(fused["split"]) == {"tuning", "test"} and (fused["plane"] == "combined").all()


def test_lr_train_is_fitted_on_the_train_set_only():
    df = plane_predictions()
    fused, weights = fuse(df, "lr_train")
    tuning = fused[fused["split"] == "tuning"]
    assert roc_auc_score(tuning["label"], tuning["prob"]) > 0.95
    assert list(weights.columns) == ["task", "method", "intercept"] + PLANES
    try:
        fuse(df[df["split"] != "train"], "lr_train")
    except ValueError:
        pass
    else:
        raise AssertionError("lr_train must refuse to run without train predictions")


def test_lr_tuning_scores_each_tuning_exam_out_of_fold():
    df = plane_predictions()
    fused, _ = fuse(df, "lr_tuning")
    tuning = df[df["split"] == "tuning"].pivot(index="exam_id", columns="plane", values="prob")[PLANES]
    labels = df[df["split"] == "tuning"].groupby("exam_id")["label"].first().loc[tuning.index]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    expected = cross_val_predict(LogisticRegression(), tuning.to_numpy(), labels.to_numpy(), cv=cv,
                                 method="predict_proba")[:, 1]
    got = fused[fused["split"] == "tuning"].set_index("exam_id")["prob"].loc[tuning.index]
    assert np.allclose(got, expected)
    test = df[df["split"] == "test"].pivot(index="exam_id", columns="plane", values="prob")[PLANES]
    full = LogisticRegression().fit(tuning.to_numpy(), labels.to_numpy()).predict_proba(test.to_numpy())[:, 1]
    assert np.allclose(fused[fused["split"] == "test"].set_index("exam_id")["prob"].loc[test.index], full)


def test_every_exam_needs_the_three_planes_and_one_label():
    df = plane_predictions()
    for broken in [df.drop(index=df.index[-1]),                                  # a missing plane
                   df.assign(label=np.where(df.index == df.index[-1], 1 - df["label"], df["label"]))]:  # labels differ
        try:
            fuse(broken, "mean")
        except ValueError:
            continue
        raise AssertionError("fuse must reject incomplete or inconsistent predictions")


def test_predictions_cannot_be_written_inside_the_repository():
    check_outside_repo(Path(tempfile.gettempdir()) / "fused.csv")
    try:
        check_outside_repo(ROOT / "results" / "fused.csv")
    except ValueError:
        return
    raise AssertionError("writing predictions inside the repository must be refused")


def exam_table(n=120, seed=1):
    """Synthetic exams with the three labels, split into train, tuning and test."""
    rng = np.random.default_rng(seed)
    split = np.array(["train"] * (n // 2) + ["tuning"] * (n // 4) + ["test"] * (n - n // 2 - n // 4))
    labels = {task: rng.integers(0, 2, n) for task in ["abnormal", "acl", "meniscus"]}
    return pd.DataFrame({"exam_id": [f"{i:04d}" for i in range(n)], "split": split, **labels})


def test_majority_baseline_gives_every_exam_the_train_share():
    table = exam_table()
    out = predict(table, "majority")
    share = table[table["split"] == "train"]["acl"].mean()
    acl = out[out["task"] == "acl"]
    assert np.allclose(acl["prob"], share) and set(acl["split"]) == {"tuning", "test"}
    assert roc_auc_score(acl["label"], acl["prob"]) == 0.5


def test_slice_baseline_learns_from_the_slice_counts():
    table = exam_table()
    # Slice counts that follow the abnormal label: the baseline should find it.
    counts = pd.DataFrame({plane: 20 + 15 * table["abnormal"].to_numpy() for plane in PLANES}, index=table["exam_id"])
    out = predict(table, "slices", counts)
    abnormal = out[(out["task"] == "abnormal") & (out["split"] == "tuning")]
    assert roc_auc_score(abnormal["label"], abnormal["prob"]) == 1.0


def test_results_table_and_roc_figure():
    fused, _ = fuse(plane_predictions(), "mean")
    preds = {"MRNet": fused[fused["split"] == "tuning"].assign(seed=42)}
    table = results_table(preds)
    assert list(table["model"]) == ["MRNet"] and table["auc"].iloc[0] > 0.95
    with tempfile.TemporaryDirectory() as tmp:
        roc_figure(preds, Path(tmp) / "roc.png")
        assert (Path(tmp) / "roc.png").stat().st_size > 0


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
