"""Settings shared by all scripts."""

import os
from pathlib import Path

SEED = 42

TASKS = ["abnormal", "acl", "meniscus"]
PLANES = ["sagittal", "coronal", "axial"]

# Folder with the MRNet data (the folder that holds train/ and valid/).
# Set MRNET_DIR to choose the folder. Otherwise: Kaggle input folder on Kaggle, ~/data/mrnet on a local machine.
ON_KAGGLE = Path("/kaggle/input").exists()
_default_mrnet = Path("/kaggle/input/mrnet-v1") if ON_KAGGLE else Path.home() / "data" / "mrnet"
MRNET_DIR = Path(os.environ.get("MRNET_DIR", _default_mrnet))

ROOT = Path(__file__).resolve().parents[1]
SPLITS_CSV = ROOT / "data" / "splits.csv"
RESULTS_DIR = ROOT / "results"
PREDICTIONS_CSV = RESULTS_DIR / "predictions.csv"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
