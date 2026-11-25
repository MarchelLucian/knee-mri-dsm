# Deep Learning for Knee MRI on the MRNet Dataset

Robert Dieaconu and Marchel Lucian, POLITEHNICA Bucharest.

We train deep learning models that read knee MRI exams and detect three things: an abnormal knee, an ACL tear and a meniscus tear. We reproduce known models (MRNet, ELNet) and compare them under the same training and test setup.

## Data

- **MRNet**: 1,370 knee MRI exams from Stanford, each with sagittal, coronal and axial series. 1,250 of them are public (train 1,130, valid 120).
  - Kaggle: https://www.kaggle.com/datasets/cjinny/mrnet-v1
  - Official page: https://stanfordmlgroup.github.io/competitions/mrnet/
- **KneeMRI**: 917 knee MRI volumes from Rijeka, labelled for ACL injury.
  - Zenodo: https://zenodo.org/records/14789903

The image data is not stored in this repository.

## Data split

| Split | Exams | Use |
|---|---|---|
| train | official train set minus tuning | training |
| tuning | 120, taken from the official train set | model selection |
| test | 120, the official validation set | final results |
| excluded | 8, from the official train set | not used: a series does not show the knee or shows the wrong plane (`data/series_flags.csv`) |

Following the MRNet paper, the tuning set has at least 50 positive exams for each label; it has the same label composition as the official validation set and is drawn at random from the official train set. Seed: 42. The exam IDs of each split are in `data/splits.csv` (`exam_id, split`).

## Structure

```
data/
  splits.csv              exam IDs of the train, tuning, test and excluded sets
  series_flags.csv        series that do not show the knee, show the wrong plane or have artefacts
src/
  config.py               seed, tasks, planes and paths
  data/
    check_data.py         checks that every exam is complete
    make_splits.py        creates data/splits.csv
    eda.py                statistics and figures about the data
    kneemri.py            reads the KneeMRI data
  model/
    dataset.py            loads and preprocesses one exam
    mrnet.py              MRNet model
    elnet.py              ELNet model
    train.py              trains one model (one task, one plane)
    fuse_planes.py        combines the three planes, writes predictions.csv
  eval/
    metrics.py            AUC, confidence intervals, sensitivity, specificity
    plots.py              ROC curves and tables
    trivial_baseline.py   majority-class baseline
tests/
  test_metrics.py         checks of the metrics on synthetic predictions
notebooks/                Kaggle notebooks
results/
  predictions.csv
  figures/
  tables/
report/                   final reports (PDF)
```

Predictions are saved in `results/predictions.csv` with the columns `exam_id, task, plane, prob, label, seed`.

## How to run

The code runs on a local machine with an NVIDIA GPU, or on Kaggle.

- **Local:** create a virtual environment, install the packages in `requirements.txt` (PyTorch with CUDA), download MRNet to a folder outside the repository and set `MRNET_DIR` to that folder.
- **Kaggle:** add the MRNet dataset to the notebook as input.

The path to the data is found in `src/config.py` (or set with `MRNET_DIR`). See [CONVENTIONS.md](CONVENTIONS.md) for the platform, data split, seed and file formats.

## References

1. N. Bien et al., "Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet," PLOS Medicine, 15(11), e1002699, 2018.
2. I. Štajduhar, M. Mamula, D. Miletić, G. Unal, "Semi-automated detection of anterior cruciate ligament injury from MRI," Computer Methods and Programs in Biomedicine, 140, 151–164, 2017.
3. C.-H. Tsai, N. Kiryati, E. Konen, I. Eshed, A. Mayer, "Knee injury detection using MRI with efficiently-layered network (ELNet)," MIDL, 2020.
