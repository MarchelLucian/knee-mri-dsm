# Conventions

Rules that all code and results in this repository follow.

## 1. Platform

- All experiments run on **Kaggle** notebooks with a GPU: <https://www.kaggle.com/code>
- The GPU is available only after phone verification: <https://www.kaggle.com/settings>
- Kaggle gives about 30 GPU hours per week and sessions of at most 12 hours, so long training runs should be planned.

## 2. Data

| Dataset | Source | Notes |
|---|---|---|
| MRNet | Kaggle: <https://www.kaggle.com/datasets/cjinny/mrnet-v1> | Added to a notebook with **Add Input**. Official page: <https://stanfordmlgroup.github.io/competitions/mrnet/>. Paper: <https://doi.org/10.1371/journal.pmed.1002699> |
| KneeMRI | Zenodo: <https://zenodo.org/records/14789903> | Licence CC BY-NC-ND 4.0. Paper: <https://doi.org/10.1016/j.cmpb.2016.12.006> |

- The path to the MRNet data is set in `src/config.py` (`MRNET_DIR`).
- Image data is never committed to the repository.

## 3. Data split

| Split | Exams | Use |
|---|---|---|
| `train` | official train set minus the tuning set | training |
| `tuning` | taken from the official train set (size and sampling: to be decided) | model selection and early stopping |
| `test` | 120, the official `valid` set | final results only, used once |

- The official `valid` set is the set called *tuning set* in the MRNet paper; the original test set is hidden. The official train and valid sets have no patient in common. Source: <https://stanfordmlgroup.github.io/competitions/mrnet/>
- In the MRNet paper, the tuning set has 120 exams, sampled so that it has at least 50 positive examples of each label.
- The split is stored in `data/splits.csv` and is the same for every experiment.
- If the data includes patient IDs, the split is made by patient, so that all exams of one patient are in the same set. If it does not, the split is made by exam, and this is stated in the report.

## 4. Random seed

`SEED = 42` (in `src/config.py`), used for the data split, PyTorch, NumPy and Python `random`.

## 5. File formats

### `data/splits.csv`

| Column | Values |
|---|---|
| `exam_id` | exam ID as in the MRNet file names, e.g. `0000` |
| `split` | `train`, `tuning` or `test` |

### `results/predictions.csv`

| Column | Values |
|---|---|
| `exam_id` | exam ID |
| `task` | `abnormal`, `acl` or `meniscus` |
| `plane` | `sagittal`, `coronal`, `axial` or `combined` |
| `prob` | predicted probability of the positive class, from 0 to 1 |
| `label` | true label, 0 or 1 |
| `seed` | random seed of the run |

## 6. Code layout

| Folder | Content |
|---|---|
| `src/data/` | data checks, data split, exploratory analysis, KneeMRI |
| `src/model/` | preprocessing, models, training, plane fusion |
| `src/eval/` | metrics, confidence intervals, plots |
| `src/config.py` | shared settings; change it only when both parts agree |

Shared names: tasks `abnormal`, `acl`, `meniscus`; planes `sagittal`, `coronal`, `axial` (as in `src/config.py`).

## 7. Git

- Pull before you start: `git pull`
- Small commits with a clear message, e.g. `Add data checks`
- Never commit: image data (`.npy`, `.pck`), model weights (`.pt`), paper PDFs, `kaggle.json`. These are already in `.gitignore`.

## 8. Language

Code, comments, results and reports are written in English.
