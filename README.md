# Deep Learning for Knee MRI on the MRNet Dataset

Robert Dieaconu and Marchel Lucian, POLITEHNICA Bucharest.

We train models that read knee MRI exams and detect three things: an abnormal knee, an anterior cruciate ligament (ACL) tear and a meniscus tear. We re-implement two known models, MRNet [1] and ELNet [3], and compare them with the same data split, the same model selection, the same test set and the same metrics. A second dataset from another hospital, KneeMRI [2], is used later to test how well the models generalise.

## Plan

| Step | What | Files | Status |
|---|---|---|---|
| 1 | Check every series of the data | `src/data/check_data.py`, `data/series_flags.csv` | done |
| 2 | Split the data into train, tuning and test | `src/data/make_splits.py`, `data/splits.csv` | done |
| 3 | Describe the data (MRNet and KneeMRI) | `src/data/eda.py`, `src/data/kneemri.py`, `results/` | done |
| 4 | Evaluation code and its tests | `src/eval/metrics.py`, `tests/test_metrics.py` | done |
| 5 | Choose the intensity normalisation (tuning set only) | `src/model/normalisation_check.py` | done |
| 6 | Baseline: frozen ImageNet AlexNet + logistic regression | `src/model/frozen_features.py`, `src/model/safety_net.py` | done |
| 7 | MRNet: 9 networks (3 tasks × 3 planes) | `src/model/dataset.py`, `src/model/mrnet.py`, `src/model/train.py` | in progress |
| 8 | Combine the three planes of each task | `src/model/fuse_planes.py` | planned |
| 9 | ELNet | `src/model/elnet.py` | planned |
| 10 | ROC curves and a majority-class baseline | `src/eval/plots.py`, `src/eval/trivial_baseline.py` | planned |
| 11 | Final evaluation on the test set, once, for all models together | `src/eval/metrics.py` | planned |

Files of planned steps contain only a one-line description for now.

## Rules shared by all models

- **Same data files.** `src/data/check_data.py` writes an MD5 checksum for every file in `results/tables/data_checks.csv`; after running it, `git diff` on that file shows whether a local copy differs.
- **Same split:** `data/splits.csv`. Models are trained on `train` only; `excluded` exams are not used anywhere.
- **Model selection on `tuning` only:** the epoch and every setting are chosen on the tuning set, never on the test set.
- **The `test` set is used once,** for all models together, at the end. Until then, test predictions are saved, but no metric is reported on them.
- **Seed 42** (`src/config.py`).
- **Same metrics:** `src/eval/metrics.py` (AUC with a 95% bootstrap interval; sensitivity, specificity and accuracy at threshold 0.5 with Wilson intervals; paired AUC difference between two models on the same exams).
- **Same prediction format** (see [CONVENTIONS.md](CONVENTIONS.md)). Predictions contain the labels, so they are never committed (they are listed in `.gitignore`).

Training settings follow the MRNet paper [1] where it gives them, and the authors' code and the reproduction in [4] where it does not.

## Data

- **MRNet:** 1,370 knee MRI exams from Stanford, each with a sagittal, a coronal and an axial series, and three labels (abnormal, ACL tear, meniscus tear). 1,250 exams are public (train 1,130, valid 120).
  - Copy used here (Kaggle): <https://www.kaggle.com/datasets/cjinny/mrnet-v1>
  - Official page: <https://stanfordmlgroup.github.io/competitions/mrnet/>
  - MRNet is released for non-commercial research use and may not be redistributed, so neither images nor labels are stored in this repository.
- **KneeMRI:** 917 knee MRI volumes (909 exams) from Rijeka, labelled for ACL injury; licence CC BY-NC-ND 4.0.
  - Zenodo: <https://zenodo.org/records/14789903>

## Data split

| Split | Exams | Use |
|---|---|---|
| `train` | 1,002: the official train set minus the tuning and excluded exams | training |
| `tuning` | 120, taken from the official train set | model selection |
| `test` | 120: the official valid set | final results, used once |
| `excluded` | 8, from the official train set | not used: a series does not show the knee or shows the wrong plane (`data/series_flags.csv`) |

Eleven series do not show the knee or show the wrong plane: 9 in the 8 excluded exams and 2 in the test set, which is kept unchanged. Nine of these eleven were also excluded in [5].

Following the MRNet paper, the tuning set has at least 50 positive exams for each label; it has the same label composition as the official valid set and is drawn at random from the official train set (seed 42). The exam IDs of each split are in `data/splits.csv` (`exam_id, split`).

## How to run

Set up once:

```bash
python -m venv venv
source venv/bin/activate                 # Windows: venv\Scripts\activate
pip install -r requirements.txt          # install PyTorch with CUDA to use a GPU
export MRNET_DIR=/path/to/MRNet-v1.0     # Windows PowerShell: $env:MRNET_DIR="C:\path\to\MRNet-v1.0"
```

`MRNET_DIR` is the folder that holds `train/`, `valid/` and the label CSV files. The ImageNet AlexNet weights are downloaded by torchvision on first use (`TORCH_HOME` sets where they are stored).

Then, from the repository root:

```bash
# 1. Check the data; compare the MD5 column with the committed file
python -m src.data.check_data --figures /path/outside/the/repo
git diff --stat results/tables/data_checks.csv

# 2. Recreate the split (the result should be identical to data/splits.csv)
python -m src.data.make_splits

# 3. Describe the data
python -m src.data.eda --image-figures /path/outside/the/repo
python -m src.data.kneemri --metadata /path/to/KneeMRI/metadata.csv

# 4. Test the metrics
python -m tests.test_metrics

# 5. Choose the intensity normalisation (tuning set only)
python -m src.model.frozen_features --norms A B C D --out /path/to/features
python -m src.model.normalisation_check --features /path/to/features

# 6. Baseline: logistic regression on the frozen AlexNet features
python -m src.model.safety_net --features /path/to/features/alexnet_frozen_B.npz --out /path/to/runs/safety_net/predictions.csv

# 7. MRNet: one network per task (abnormal, acl, meniscus) and plane (sagittal, coronal, axial)
python -m src.model.train --task acl --plane sagittal --epochs 50 --out /path/to/runs

# 8. ELNet: one network per task, on the plane used in the ELNet paper
python -m src.model.train_elnet --task acl --epochs 200 --out /path/to/runs

# 9. Combine the three planes of each task (logistic regression fitted on the train predictions, as in MRNet;
#    "mean" is the plain average, used as a check). Run it for MRNet and for the frozen-feature baseline.
python -m src.model.fuse_planes --predictions /path/to/runs/*_seed42/predictions.csv --method lr_train --out /path/to/final_eval/mrnet_fused_lr_train.csv
python -m src.model.fuse_planes --predictions /path/to/runs/*_seed42/predictions.csv --method mean --out /path/to/final_eval/mrnet_fused_mean.csv

# 10. Trivial baselines: majority class, and logistic regression on the number of slices
python -m src.eval.trivial_baseline --kind majority --out /path/to/final_eval/trivial_majority.csv
python -m src.eval.trivial_baseline --kind slices --out /path/to/final_eval/trivial_slices.csv

# 11. Evaluate all models on the same exams: first on tuning; on test only once, at the end
python -m src.eval.final_evaluation --split tuning --final /path/to/final_eval --runs /path/to/runs --elnet /path/to/elnet_predictions --out /path/to/final_eval/tuning

# Tests of the metrics and of the plane combination
python -m tests.test_metrics
python -m tests.test_fusion

# Metrics from a single predictions file
python -m src.eval.metrics --predictions /path/to/predictions.csv --out /path/to/metrics.csv
```

Each MRNet run writes, in its own folder: the settings (`args.json`), one line per epoch (`log.csv`), the tuning predictions after every epoch, the weights of the selected epoch (`best.pt`) and the predictions of the selected epoch (`predictions.csv`). One network takes about 40 minutes on a laptop GPU (RTX 3060).

## Structure

```
data/
  splits.csv               exam IDs of the train, tuning, test and excluded sets
  series_flags.csv         series that do not show the knee, show the wrong plane or have artefacts
src/
  config.py                seed, tasks, planes, paths and intensity normalisation
  data/
    check_data.py          checks every series (planes, image size, values, labels, MD5)
    make_splits.py         creates data/splits.csv
    eda.py                 statistics and figures about the data
    kneemri.py             summarises the KneeMRI metadata
    fix_redivis_labels.py  restores the label files of the Redivis download to the original format
  model/
    dataset.py             loads, augments and normalises one series
    mrnet.py               MRNet model
    train.py               trains one MRNet network (one task, one plane)
    frozen_features.py     frozen ImageNet AlexNet features (baseline)
    safety_net.py          logistic regression on the frozen features (baseline)
    normalisation_check.py compares intensity normalisations on the tuning set
    fuse_planes.py         combines the three planes of each task
    elnet.py               ELNet model
    train_elnet.py         trains one ELNet network (one task)
  eval/
    metrics.py             AUC, confidence intervals, sensitivity, specificity, paired comparison
    plots.py               ROC curves and tables
    trivial_baseline.py    trivial baselines (majority class, number of slices)
    final_evaluation.py    evaluates all models on the same exams and compares them in pairs
    learning_curve.py      training and tuning loss per epoch
tests/
  test_metrics.py          checks of the metrics on synthetic predictions
  test_fusion.py           checks of the plane combination and the trivial baselines
results/
  tables/                  data checks, data description, normalisation check
  figures/                 figures for the report
notebooks/                 Kaggle notebooks
report/                    final reports (PDF)
```

## References

1. N. Bien et al., "Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet," PLOS Medicine, 15(11), e1002699, 2018.
2. I. Štajduhar, M. Mamula, D. Miletić, G. Ünal, "Semi-automated detection of anterior cruciate ligament injury from MRI," Computer Methods and Programs in Biomedicine, 140, 151–164, 2017.
3. C.-H. Tsai, N. Kiryati, E. Konen, I. Eshed, A. Mayer, "Knee injury detection using MRI with efficiently-layered network (ELNet)," MIDL, 2020.
4. M. Dunnhofer, N. Martinel, C. Micheloni, "Improving MRI-based knee disorder diagnosis with pyramidal feature details," MIDL, PMLR 143, 131–147, 2021.
5. A. C. Kara, F. Hardalaç, "Detection and classification of knee injuries from MR images using the MRNet dataset with progressively operating deep learning methods," Machine Learning and Knowledge Extraction, 3(4), 1009–1029, 2021.
