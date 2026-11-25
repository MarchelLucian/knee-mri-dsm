"""Summarise the KneeMRI metadata: label counts per volume and per exam, and the regions of interest.

KneeMRI (Štajduhar et al., 2017; Zenodo record 14789903) has one sagittal volume per row of
metadata.csv. Some exams have more than one volume. aclDiagnosis: 0 = healthy, 1 = partially
injured, 2 = completely ruptured. This script uses only metadata.csv (no image volumes) and
writes results/tables/kneemri_summary.csv.

Usage:
    python -m src.data.kneemri --metadata /path/to/KneeMRI/metadata.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from src.config import KNEEMRI_DIR, TABLES_DIR

DIAGNOSIS = {0: "healthy", 1: "partially injured", 2: "completely ruptured"}


def summary(meta):
    """Table with the columns quantity and value."""
    per_exam = meta.groupby("examId").agg(volumes=("volumeFilename", "size"),
                                          diagnoses=("aclDiagnosis", "nunique"),
                                          diagnosis=("aclDiagnosis", "first"),
                                          knee_sides=("kneeLR", "nunique"))
    if (per_exam["diagnoses"] > 1).any():
        raise ValueError("an exam has volumes with different diagnoses")
    rows = [("volumes", len(meta)), ("exams", len(per_exam))]
    for k, n in per_exam["volumes"].value_counts().sort_index().items():
        rows.append((f"exams with {k} volume(s)", int(n)))
    for code, name in DIAGNOSIS.items():
        rows.append((f"volumes: {name}", int((meta["aclDiagnosis"] == code).sum())))
    for code, name in DIAGNOSIS.items():
        rows.append((f"exams: {name}", int((per_exam["diagnosis"] == code).sum())))
    injured = int((per_exam["diagnosis"] > 0).sum())
    rows.append(("exams: injured (partial or complete)", injured))
    rows.append(("exams: injured (%)", round(100 * injured / len(per_exam), 1)))
    rows.append(("exams with volumes of both knee sides", int((per_exam["knee_sides"] > 1).sum())))
    for code in sorted(meta["kneeLR"].unique()):
        rows.append((f"volumes with kneeLR = {code} (meaning not documented)", int((meta["kneeLR"] == code).sum())))
    for col, unit in [("roiHeight", "pixels"), ("roiWidth", "pixels"), ("roiDepth", "slices")]:
        rows.append((f"{col} ({unit}): min / median / max",
                     f"{meta[col].min()} / {meta[col].median():g} / {meta[col].max()}"))
    return pd.DataFrame(rows, columns=["quantity", "value"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metadata", type=Path, default=KNEEMRI_DIR / "metadata.csv")
    parser.add_argument("--out", type=Path, default=TABLES_DIR / "kneemri_summary.csv")
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata)
    table = summary(meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
