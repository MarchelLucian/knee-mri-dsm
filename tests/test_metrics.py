"""Checks for src/eval/metrics.py on synthetic predictions.

Run from the repository root:
    python -m pytest tests        (if pytest is installed)
    python -m tests.test_metrics
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.eval.metrics import auc_bootstrap_ci, evaluate_predictions, threshold_metrics, wilson_ci

# Test set of the same size as the official MRNet valid set: 120 exams, 54 ACL positives.
N, N_POS = 120, 54
LABELS = np.r_[np.ones(N_POS, dtype=int), np.zeros(N - N_POS, dtype=int)]


def test_random_predictions_give_auc_near_half():
    probs = np.random.default_rng(0).random(N)
    auc, low, high = auc_bootstrap_ci(LABELS, probs)
    assert 0.35 < auc < 0.65
    assert low < 0.5 < high


def test_perfect_predictions_give_auc_one():
    probs = np.where(LABELS == 1, 0.9, 0.1)
    assert auc_bootstrap_ci(LABELS, probs) == (1.0, 1.0, 1.0)
    m = threshold_metrics(LABELS, probs)
    assert m["sensitivity"][0] == m["specificity"][0] == m["accuracy"][0] == 1.0


def test_inverted_predictions_give_auc_zero():
    probs = np.where(LABELS == 1, 0.1, 0.9)
    assert auc_bootstrap_ci(LABELS, probs)[0] == 0.0


def test_bootstrap_is_reproducible_and_uses_its_own_seed():
    probs = np.random.default_rng(1).random(N)
    first = auc_bootstrap_ci(LABELS, probs)
    np.random.seed(123)  # the global random state must not change the result
    assert auc_bootstrap_ci(LABELS, probs) == first
    assert auc_bootstrap_ci(LABELS, probs, seed=7) != first


def test_auc_matches_mann_whitney():
    # AUC = U / (n_pos * n_neg), computed independently with scipy, ties included.
    probs = np.round(np.random.default_rng(4).random(N), 1)
    u = mannwhitneyu(probs[LABELS == 1], probs[LABELS == 0]).statistic
    assert np.isclose(auc_bootstrap_ci(LABELS, probs, n_bootstrap=10)[0], u / (N_POS * (N - N_POS)))


def test_known_confusion_matrix():
    # TP = 3, FN = 1, TN = 5, FP = 2: sensitivity 3/4, specificity 5/7, accuracy 8/11.
    labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    probs = [0.9, 0.8, 0.7, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.6]
    m = threshold_metrics(labels, probs)
    assert np.isclose(m["sensitivity"][0], 3 / 4)
    assert np.isclose(m["specificity"][0], 5 / 7)
    assert np.isclose(m["accuracy"][0], 8 / 11)


def test_wilson_matches_mrnet_paper():
    # Bien et al. 2018, Table 2, model on ACL: sensitivity 0.759 (0.635-0.850)
    # = 44 of 58 tears, specificity 0.968 (0.890-0.991) = 60 of 62 intact ACLs.
    p, low, high = wilson_ci(44, 58)
    assert (round(p, 3), round(low, 3), round(high, 3)) == (0.759, 0.635, 0.850)
    p, low, high = wilson_ci(60, 62)
    assert (round(p, 3), round(low, 3), round(high, 3)) == (0.968, 0.890, 0.991)


def test_threshold_is_inclusive_at_half():
    m = threshold_metrics([1, 0], [0.5, 0.49])
    assert m["sensitivity"][0] == 1.0 and m["specificity"][0] == 1.0


def test_bad_inputs_are_rejected():
    for labels, probs in [([1, 0, np.nan], [0.9, 0.1, 0.5]),   # missing label
                          ([1, 0, 0.7], [0.9, 0.1, 0.5]),      # label not 0/1
                          ([1, 0, 0], [0.9, np.nan, 0.5]),     # missing probability
                          ([1, 0, 0], [0.9, 0.1, 1.5])]:       # probability above 1
        try:
            threshold_metrics(labels, probs)
        except ValueError:
            continue
        raise AssertionError(f"accepted bad input: {labels}, {probs}")


def test_duplicate_or_missing_rows_are_rejected():
    df = pd.DataFrame({"exam_id": ["0001", "0001", "0002"], "task": "acl", "plane": "axial",
                       "prob": [0.9, 0.9, 0.1], "label": [1, 1, 0], "seed": 42})
    for bad in [df, df.drop(index=0).assign(plane=[np.nan, "axial"])]:
        try:
            evaluate_predictions(bad, n_bootstrap=10)
        except ValueError:
            continue
        raise AssertionError("accepted a duplicate or a missing key")


def test_evaluate_predictions_gives_one_row_per_task_plane_seed():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "exam_id": [f"{i:04d}" for i in range(N)] * 2,
        "task": ["acl"] * N + ["meniscus"] * N,
        "plane": "combined",
        "prob": rng.random(2 * N),
        "label": np.r_[LABELS, LABELS],
        "seed": 42,
    })
    table = evaluate_predictions(df, n_bootstrap=200)
    assert list(table["task"]) == ["acl", "meniscus"]
    assert (table["n"] == N).all() and (table["n_pos"] == N_POS).all()
    assert ((table["auc_low"] <= table["auc"]) & (table["auc"] <= table["auc_high"])).all()


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
