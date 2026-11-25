"""MRNet dataset: loads one exam and one plane, normalises intensities, repeats the slice over 3 channels and applies augmentation (training only).

Augmentation follows Bien et al. (2018): each time an exam is loaded, a random rotation between
-25 and 25 degrees, a random shift between -25 and 25 pixels and a horizontal flip with
probability 0.5. The same transform is applied to every slice of the series.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as TF

from src.data.check_data import read_labels

FOLDER = {"train": "train", "tuning": "train", "test": "valid"}


def load_table(data_dir, splits_csv):
    """One row per exam: exam_id, split, abnormal, acl, meniscus (excluded exams removed)."""
    labels = pd.concat([read_labels(data_dir, "train"), read_labels(data_dir, "valid")])
    splits = pd.read_csv(splits_csv, dtype={"exam_id": str})
    df = splits.merge(labels, on="exam_id", how="left", validate="one_to_one")
    return df[df["split"] != "excluded"].reset_index(drop=True)


def augment(x, generator=None):
    """Random rotation, shift and horizontal flip, the same for all slices. x: (slices, H, W) float."""
    angle = float(torch.empty(1).uniform_(-25, 25, generator=generator))
    shift = [int(torch.randint(-25, 26, (1,), generator=generator)) for _ in range(2)]
    x = TF.affine(x, angle=angle, translate=shift, scale=1.0, shear=[0.0, 0.0],
                  interpolation=InterpolationMode.BILINEAR, fill=0.0)
    if float(torch.rand(1, generator=generator)) < 0.5:
        x = TF.horizontal_flip(x)
    return x


class MRNetDataset(Dataset):
    """Exams of one split for one plane and one task. Each item: (series (slices, 3, H, W), label (1,), exam_id)."""

    def __init__(self, data_dir, table, split, plane, task, mean, std, train=False):
        part = table[table["split"] == split]
        self.paths = [data_dir / FOLDER[split] / plane / f"{e}.npy" for e in part["exam_id"]]
        self.labels = part[task].astype(np.float32).to_numpy()
        self.exam_ids = part["exam_id"].tolist()
        self.mean, self.std, self.train = float(mean), float(std), train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = torch.from_numpy(np.load(self.paths[i]).astype(np.float32))  # (slices, H, W), values 0-255
        if self.train:
            x = augment(x)
        x = (x - self.mean) / self.std
        x = x.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()  # grey copied to 3 channels
        return x, torch.tensor([self.labels[i]]), self.exam_ids[i]

    def prevalence(self):
        """Share of positive exams in this set."""
        return float(self.labels.mean())
