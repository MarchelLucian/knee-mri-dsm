"""Settings shared by all scripts."""

import os
from pathlib import Path

SEED = 42

TASKS = ["abnormal", "acl", "meniscus"]
PLANES = ["sagittal", "coronal", "axial"]

# Folder with the MRNet data: the folder that holds train/, valid/ and the label CSV files.
# Set MRNET_DIR to choose the folder. Otherwise: Kaggle input folder on Kaggle, ~/data/mrnet on a local machine.
# The Kaggle copy keeps everything in a subfolder MRNet-v1.0/, which is used if present.
ON_KAGGLE = Path("/kaggle/input").exists()
_default_mrnet = Path("/kaggle/input/mrnet-v1") if ON_KAGGLE else Path.home() / "data" / "mrnet"


def _data_folder(path):
    path = Path(path)
    if not (path / "train").is_dir() and (path / "MRNet-v1.0" / "train").is_dir():
        return path / "MRNet-v1.0"
    return path


MRNET_DIR = _data_folder(os.environ.get("MRNET_DIR", _default_mrnet))

# Folder with the KneeMRI data (metadata.csv and the volumes). Set KNEEMRI_DIR to choose the folder.
KNEEMRI_DIR = Path(os.environ.get("KNEEMRI_DIR", Path.home() / "data" / "kneemri"))

ROOT = Path(__file__).resolve().parents[1]
SPLITS_CSV = ROOT / "data" / "splits.csv"
SERIES_FLAGS_CSV = ROOT / "data" / "series_flags.csv"
RESULTS_DIR = ROOT / "results"
PREDICTIONS_CSV = RESULTS_DIR / "predictions.csv"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
