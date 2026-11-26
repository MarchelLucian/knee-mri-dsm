"""Rewrite the label CSV files exported from Redivis (Stanford AIMI) in the original MRNet format.

The Redivis export turns the first exam into a header row, e.g. "_0000","_0" instead of 0000,0.
This script writes the six files back as "exam_id,label" lines without a header (as in the Kaggle copy
and the original release) and keeps a copy of each original file as <name>.redivis.csv.

Usage:
    python -m src.data.fix_redivis_labels            (uses MRNET_DIR)
"""

import shutil

from src.config import MRNET_DIR, TASKS


def fix(path):
    lines = path.read_text().strip().splitlines()
    if not lines[0].startswith('"_'):
        return False  # already in the original format
    shutil.copy(path, path.with_suffix(".redivis.csv"))
    rows = [lines[0].replace('"', "").replace("_", "")] + lines[1:]
    rows = sorted(r.strip() for r in rows)
    path.write_text("\n".join(rows) + "\n")
    return True


if __name__ == "__main__":
    for split in ["train", "valid"]:
        for task in TASKS:
            p = MRNET_DIR / f"{split}-{task}.csv"
            n = len(p.read_text().strip().splitlines())
            changed = fix(p)
            n_new = len(p.read_text().strip().splitlines())
            print(f"{p.name}: {'fixed' if changed else 'already OK'}, {n_new} exams" + ("" if changed else f" ({n})"))
