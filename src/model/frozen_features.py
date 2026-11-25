"""Frozen AlexNet features for the safety-net baseline.

For every exam and plane: the ImageNet AlexNet convolutional features of each slice, global average
pooling (256 values per slice) and the maximum over slices (256 values per series), as in MRNet
but without training the network. Several intensity normalisations are computed from one read of
the data, so they can be compared on the same exams.

Normalisations (pixel values are 0-255):
- A: (x - mean) / sd of each plane, computed on our train set
- B: (x - 58.09) / 49.73 for every plane (the authors' code [1] and the reproduction in [8])
- C: ImageNet: x / 255, then per channel (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
- D: none (raw 0-255)

Usage:
    python -m src.model.frozen_features --norms A B C D --out /mnt/d/knee-mri/features
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.config import MRNET_DIR, PLANES, SEED, SPLITS_CSV
from src.model.dataset import FOLDER, load_table
from src.model.mrnet import MRNet

TRAIN_STATS = {"sagittal": (58.89, 48.56), "coronal": (59.71, 64.15), "axial": (63.74, 60.98)}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_SD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalise(x, plane, norm):
    """x: (slices, H, W) float, values 0-255. Returns (slices, 3, H, W)."""
    if norm == "A":
        mean, sd = TRAIN_STATS[plane]
        return ((x - mean) / sd).unsqueeze(1).expand(-1, 3, -1, -1)
    if norm == "B":
        return ((x - 58.09) / 49.73).unsqueeze(1).expand(-1, 3, -1, -1)
    if norm == "C":
        x3 = (x / 255.0).unsqueeze(1).expand(-1, 3, -1, -1)
        return (x3 - IMAGENET_MEAN.to(x.device)) / IMAGENET_SD.to(x.device)
    if norm == "D":
        return x.unsqueeze(1).expand(-1, 3, -1, -1)
    raise ValueError(f"unknown normalisation {norm}")


@torch.no_grad()
def extract(data_dir, table, norms, device="cuda"):
    """Returns {norm: array (exams, planes, 256)} in the order of table rows and PLANES."""
    model = MRNet(pretrained=True).to(device).eval()
    out = {n: np.zeros((len(table), len(PLANES), 256), dtype=np.float32) for n in norms}
    start = time.time()
    for i, row in enumerate(table.itertuples()):
        for j, plane in enumerate(PLANES):
            vol = np.load(data_dir / FOLDER[row.split] / plane / f"{row.exam_id}.npy")
            x = torch.from_numpy(vol.astype(np.float32)).to(device)
            for n in norms:
                out[n][i, j] = model.series_features(normalise(x, plane, n)).squeeze(0).cpu().numpy()
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(table)} exams, {time.time() - start:.0f} s", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--norms", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--out", type=Path, required=True, help="folder outside the repository")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    table = load_table(MRNET_DIR, SPLITS_CSV)
    feats = extract(MRNET_DIR, table, args.norms)
    args.out.mkdir(parents=True, exist_ok=True)
    for n, f in feats.items():
        np.savez_compressed(args.out / f"alexnet_frozen_{n}.npz", features=f, exam_id=table["exam_id"].to_numpy(),
                            split=table["split"].to_numpy(), planes=np.array(PLANES))
        print(f"saved {args.out / f'alexnet_frozen_{n}.npz'}: {f.shape}")


if __name__ == "__main__":
    main()
