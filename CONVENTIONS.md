# Conventions

Rules that all code and results in this repository follow.

## 1. Platform

- **Main:** a local machine with an NVIDIA GPU (Ubuntu / WSL, Python virtual environment, PyTorch with CUDA).
- **Alternative:** Kaggle notebooks with a GPU: <https://www.kaggle.com/code>. Kaggle gives about 30 GPU hours per week and sessions of at most 12 hours.
- The code is the same on both. Only the path to the data changes, and `src/config.py` finds it: the folder set in the `MRNET_DIR` environment variable, `/kaggle/input/mrnet-v1` on Kaggle, or `~/data/mrnet` if `MRNET_DIR` is not set.
- Data on a local machine is kept outside the repository and outside synced folders such as OneDrive, on a disk with enough free space (MRNet is about 7.7 GB, and about twice that while the archive is extracted). Example: `export MRNET_DIR=/mnt/d/knee-mri/data/MRNet-v1.0` (the folder that holds `train/`, `valid/` and the label CSV files; if it has a subfolder `MRNet-v1.0/`, that subfolder is used).

## 2. Data

| Dataset | Source | Notes |
|---|---|---|
| MRNet | Kaggle: <https://www.kaggle.com/datasets/cjinny/mrnet-v1> | Local: downloaded with the Kaggle CLI to the folder set in `MRNET_DIR`. On Kaggle: added to a notebook with **Add Input**. Official page: <https://stanfordmlgroup.github.io/competitions/mrnet/>. Paper: <https://doi.org/10.1371/journal.pmed.1002699> |
| KneeMRI | Zenodo: <https://zenodo.org/records/14789903> | Licence CC BY-NC-ND 4.0. Paper: <https://doi.org/10.1016/j.cmpb.2016.12.006> |

- Image data is never committed to the repository.

## 3. Data split

| Split | Exams | Use |
|---|---|---|
| `train` | official train set minus the tuning set | training |
| `tuning` | 120, taken from the official train set, with the same label composition as the official `valid` set (at least 50 positive exams for each label) | model selection (choice of the epoch) |
| `test` | 120, the official `valid` set | final results only, used once |
| `excluded` | 8 train exams with a series that does not show the knee or shows another plane than its folder (listed in `data/series_flags.csv`) | not used |

- The official `valid` set is the set called *tuning set* in the MRNet paper; the original test set is hidden. The official train and valid sets have no patient in common. Source: <https://stanfordmlgroup.github.io/competitions/mrnet/>
- The tuning set follows the rule of the MRNet paper (120 exams, at least 50 positive examples of each label), but it is drawn from the official train set and split by exam. For each of the label combinations (abnormal, ACL, meniscus) it has as many exams as the official `valid` set, drawn at random with seed 42.
- The split is stored in `data/splits.csv` and is the same for every experiment.
- If the data includes patient IDs, the split is made by patient, so that all exams of one patient are in the same set. If it does not, the split is made by exam, and this is stated in the report.

## 4. Random seed

`SEED = 42` (in `src/config.py`), used for the data split, PyTorch, NumPy and Python `random`.

## 5. File formats

### `data/splits.csv`

| Column | Values |
|---|---|
| `exam_id` | exam ID as in the MRNet file names, e.g. `0000` |
| `split` | `train`, `tuning`, `test` or `excluded` |

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
