"""Settings shared by all scripts."""

import os
from pathlib import Path

SEED = 42

TASKS = ["abnormal", "acl", "meniscus"]
PLANES = ["sagittal", "coronal", "axial"]

# Folder with the MRNet data (the folder that holds train/ and valid/).
# On Kaggle it is under /kaggle/input/; set MRNET_DIR if it is somewhere else.
MRNET_DIR = Path(os.environ.get("MRNET_DIR", "/kaggle/input/mrnet-v1"))

ROOT = Path(__file__).resolve().parents[1]
SPLITS_CSV = ROOT / "data" / "splits.csv"
RESULTS_DIR = ROOT / "results"
PREDICTIONS_CSV = RESULTS_DIR / "predictions.csv"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
