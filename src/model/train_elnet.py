"""Train one ELNet for one task, and keep the epoch with the lowest loss on the tuning set.

Same rules as src/model/train.py (MRNet): data/splits.csv, training on "train", choice of the epoch on
"tuning" (lowest weighted BCE, weights p for negatives and 1 - p for positives), the test set only
predicted once with the selected weights, seed 42, the same output files.

Follows Tsai et al. (2020) where the paper is explicit: one network per task, K = 4, one plane per task
(axial with layer normalisation for ACL and abnormal, coronal with contrast normalisation for meniscus),
trained from scratch, Adam with a learning rate between 1e-5 and 3e-5, 200 epochs, minority-class
oversampling, augmentation with translation, horizontal flip, scaling, rotation up to 10 degrees and, for
axial and coronal series, a random rotation by a multiple of 90 degrees.

Our choices where the paper is silent or where we differ (to state in the report):
- intensities: (x - 58.09) / 49.73, the same as MRNet (the paper uses histogram standardisation);
- shift up to 25 pixels and scale 0.9-1.1 (amounts not given in the paper); weight decay 0;
- one output with binary cross-entropy (the authors' code: two outputs with cross-entropy);
- one seed (42) by default; the paper keeps the best of seeds 0-4 (possible with --seed, chosen on tuning).

Outputs (in --out, outside the repository): args.json, log.csv, tuning_probs.csv, best.pt and
predictions.csv (train, tuning and test predictions of the selected epoch), as in src/model/train.py.

Usage:
    python -m src.model.train_elnet --task acl --epochs 200 --out D:/knee-mri/runs
    python -m src.model.train_elnet --task acl --epochs 1 --max-exams 50 --out D:/knee-mri/runs   (timing test)
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for deterministic cuBLAS

import argparse
import functools
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as TF

from src.config import MRNET_DIR, NORM_MEAN, NORM_SD, SEED, SPLITS_CSV, TASKS
from src.model.dataset import MRNetDataset, load_table
from src.model.elnet import ELNet, count_parameters
from src.model.train import git_commit, weighted_loss

PAPER_SETUP = {"abnormal": ("axial", "layer"), "acl": ("axial", "layer"), "meniscus": ("coronal", "contrast")}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # warn_only: a few CUDA kernels used by ELNet have no deterministic version
    torch.use_deterministic_algorithms(True, warn_only=True)


class ELNetDataset(MRNetDataset):
    """Same exams, labels and normalisation as MRNetDataset; one grey channel and the ELNet augmentation.
    Each item: (series (slices, 1, H, W), label (1,), exam_id)."""

    def __init__(self, *args, plane, gpu_aug=False, **kwargs):
        super().__init__(*args, plane=plane, **kwargs)
        self.plane = plane
        self.gpu_aug = gpu_aug  # if True, augmentation is done on the GPU in run_epoch (faster)

    def augment(self, x):
        angle = random.uniform(-10, 10)
        shift = [random.randint(-25, 25), random.randint(-25, 25)]
        scale = random.uniform(0.9, 1.1)
        x = TF.affine(x, angle=angle, translate=shift, scale=scale, shear=[0.0, 0.0],
                      interpolation=InterpolationMode.BILINEAR, fill=0.0)
        if random.random() < 0.5:
            x = TF.horizontal_flip(x)
        if self.plane in ("axial", "coronal"):
            x = torch.rot90(x, k=random.randint(0, 3), dims=(1, 2))
        return x

    def __getitem__(self, i):
        x = torch.from_numpy(np.load(self.paths[i]).astype(np.float32))  # (slices, H, W), values 0-255
        if self.train and not self.gpu_aug:
            x = self.augment(x)
            x = (x - self.mean) / self.std
        elif not self.train:
            x = (x - self.mean) / self.std
        # with gpu_aug the raw 0-255 series is returned; run_epoch augments and normalises it on the GPU
        return x.unsqueeze(1), torch.tensor([self.labels[i]]), self.exam_ids[i]


def run_epoch(model, loader, prevalence, device, optimizer=None, amp=False, gpu_augment=None):
    """One pass over a loader. Trains if an optimizer is given (plain BCE: the classes are balanced by
    oversampling). The reported loss is always the weighted BCE used to choose the epoch."""
    training = optimizer is not None
    model.train(training)
    losses, probs, labels, ids = [], [], [], []
    with torch.set_grad_enabled(training):
        for x, y, exam_id in loader:
            x, y = x.to(device, non_blocking=True), y.to(device)
            if gpu_augment is not None:  # x: (1, slices, 1, H, W), raw 0-255
                x = ((gpu_augment(x[0, :, 0]) - NORM_MEAN) / NORM_SD)[None, :, None]
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logit = model(x)
            logit = logit.float()
            if training:
                loss = F.binary_cross_entropy_with_logits(logit, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(weighted_loss(logit, y, prevalence).item())
            probs.append(torch.sigmoid(logit).item())
            labels.append(y.item())
            ids.extend(exam_id)
    auc = roc_auc_score(labels, probs) if 0 < sum(labels) < len(labels) else float("nan")
    return float(np.mean(losses)), auc, probs, labels, ids


def seed_worker(worker_id, seed):
    """Seeds NumPy and random in each loader worker (a top-level function, so it also works on Windows)."""
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def make_loader(dataset, seed, workers, sampler=None):
    return DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0,
                      worker_init_fn=functools.partial(seed_worker, seed=seed))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--plane", choices=["sagittal", "coronal", "axial"], default=None,
                        help="default: the plane used in the paper for this task")
    parser.add_argument("--norm", choices=["layer", "contrast"], default=None,
                        help="default: the normalisation used in the paper for this task")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--stop-after", type=int, default=0,
                        help="stop after N epochs without a lower tuning loss (0 = run all epochs)")
    parser.add_argument("--cpu-aug", action="store_true", help="augment on the CPU (slower; default: on the GPU)")
    parser.add_argument("--amp", action="store_true", help="mixed precision (faster, not bit-for-bit reproducible)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-exams", type=int, default=0, help="only for a quick test: first N train and tuning exams")
    parser.add_argument("--out", type=Path, required=True, help="folder outside the repository")
    args = parser.parse_args()
    plane = args.plane or PAPER_SETUP[args.task][0]
    norm = args.norm or PAPER_SETUP[args.task][1]

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = args.out / (f"elnet_{args.task}_{plane}_seed{args.seed}" + ("_test" if args.max_exams else ""))
    run_dir.mkdir(parents=True, exist_ok=True)

    table = load_table(MRNET_DIR, SPLITS_CSV)
    gpu_aug = device == "cuda" and not args.cpu_aug
    sets = {s: ELNetDataset(MRNET_DIR, table, s, plane=plane, task=args.task, mean=NORM_MEAN, std=NORM_SD,
                            train=(s == "train"), gpu_aug=gpu_aug) for s in ["train", "tuning", "test"]}
    prevalence = {s: d.prevalence() for s, d in sets.items()}
    train_set, tuning_set = sets["train"], sets["tuning"]
    train_labels = sets["train"].labels
    if args.max_exams:
        n_train, n_tuning = min(args.max_exams, len(train_set)), min(args.max_exams, len(tuning_set))
        train_set, tuning_set = Subset(train_set, range(n_train)), Subset(tuning_set, range(n_tuning))
        train_labels = train_labels[:n_train]

    # minority-class oversampling: each class drawn with the same probability
    n_pos = float(train_labels.sum())
    weights = np.where(train_labels > 0.5, 1.0 / max(n_pos, 1), 1.0 / max(len(train_labels) - n_pos, 1))
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(train_labels),
                                    replacement=True, generator=torch.Generator().manual_seed(args.seed))

    model = ELNet(args.K, norm, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    info = {**{k: str(v) for k, v in vars(args).items()}, "model": "elnet", "plane": plane, "norm": norm,
            "prevalence": prevalence, "parameters": count_parameters(model), "torch": torch.__version__,
            "git_commit": git_commit(), "splits_md5": hashlib.md5(SPLITS_CSV.read_bytes()).hexdigest(),
            "normalisation": [NORM_MEAN, NORM_SD], "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    (run_dir / "args.json").write_text(json.dumps(info, indent=2))
    print(f"ELNet K={args.K} {norm}, {plane} plane, {args.task}: {count_parameters(model):,} parameters, "
          f"{len(train_set)} train / {len(tuning_set)} tuning exams", flush=True)

    train_loader = make_loader(train_set, args.seed, args.workers, sampler)
    tuning_loader = make_loader(tuning_set, args.seed, args.workers)
    log, tuning_probs, best = [], {}, (float("inf"), -1)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr_loss, tr_auc, *_ = run_epoch(model, train_loader, prevalence["train"], device, optimizer, args.amp,
                                        gpu_augment=sets["train"].augment if gpu_aug else None)
        tu_loss, tu_auc, tu_p, _, tu_ids = run_epoch(model, tuning_loader, prevalence["tuning"], device, amp=args.amp)
        tuning_probs[f"epoch_{epoch}"] = tu_p
        if tu_loss < best[0]:
            best = (tu_loss, epoch)
            torch.save(model.state_dict(), run_dir / "best.pt")
        seconds = time.time() - start
        log.append({"epoch": epoch, "lr": args.lr, "train_loss": tr_loss, "train_auc": tr_auc,
                    "tuning_loss": tu_loss, "tuning_auc": tu_auc, "seconds": round(seconds, 1)})
        pd.DataFrame(log).to_csv(run_dir / "log.csv", index=False, float_format="%.6g")
        pd.DataFrame({"exam_id": tu_ids, **tuning_probs}).to_csv(run_dir / "tuning_probs.csv", index=False,
                                                               float_format="%.5f")
        print(f"epoch {epoch:3d} | train loss {tr_loss:.4f} auc {tr_auc:.3f} | "
              f"tuning loss {tu_loss:.4f} auc {tu_auc:.3f} | {seconds:.0f} s", flush=True)
        if args.stop_after and epoch - best[1] >= args.stop_after:
            print(f"no lower tuning loss for {args.stop_after} epochs: stop")
            break

    # Predictions of the selected epoch, without augmentation.
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    rows = []
    # In a quick test (--max-exams) the test set is not touched at all.
    for split in ["train", "tuning"] + ([] if args.max_exams else ["test"]):
        plain = ELNetDataset(MRNET_DIR, table, split, plane=plane, task=args.task, mean=NORM_MEAN, std=NORM_SD)
        if args.max_exams:
            plain = Subset(plain, range(min(args.max_exams, len(plain))))
        _, _, p, y, ids = run_epoch(model, make_loader(plain, args.seed, args.workers), prevalence[split],
                                    device, amp=args.amp)
        rows += [{"exam_id": i, "split": split, "task": args.task, "plane": plane, "prob": pi,
                  "label": int(yi), "seed": args.seed, "epoch": best[1]} for i, pi, yi in zip(ids, p, y)]
    pd.DataFrame(rows).to_csv(run_dir / "predictions.csv", index=False, float_format="%.6f")
    print(f"selected epoch {best[1]} (tuning loss {best[0]:.4f}); outputs in {run_dir}")


if __name__ == "__main__":
    main()
