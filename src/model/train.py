"""Train one model for one task and one plane, and keep the epoch with the lowest loss on the tuning set.

Follows Bien et al. (2018) [1] where the paper is explicit (256x256 input, ImageNet AlexNet fine-tuned,
weighted binary cross-entropy, augmentation, model with the lowest tuning loss) and the authors' code
(S1) for the rest: Adam (lr 1e-5, weight decay 0.01), ReduceLROnPlateau on the tuning loss (patience 5,
factor 0.3), loss weights p for negatives and 1 - p for positives with p the prevalence of each set,
one exam per step. Intensities: (x - 58.09) / 49.73, as in S1 and in Dunnhofer et al. (2021).

Outputs (in --out, outside the repository): args.json, log.csv (one row per epoch), tuning_probs.csv
(tuning predictions after every epoch), best.pt (weights of the selected epoch) and predictions.csv
(train, tuning and test predictions of the selected epoch, without augmentation).

Usage:
    python -m src.model.train --task acl --plane sagittal --epochs 50 --out /mnt/d/knee-mri/runs
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for deterministic cuBLAS

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from src.config import MRNET_DIR, NORM_MEAN, NORM_SD, PLANES, SEED, SPLITS_CSV, TASKS
from src.model.dataset import MRNetDataset, load_table
from src.model.mrnet import MRNet, count_parameters


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def weighted_loss(logit, label, prevalence):
    """BCE with weight p for negatives and 1 - p for positives (authors' code, S1)."""
    weight = torch.where(label > 0.5, 1.0 - prevalence, prevalence)
    return F.binary_cross_entropy_with_logits(logit, label, weight=weight)


def run_epoch(model, loader, prevalence, device, optimizer=None):
    """One pass over a loader. Trains if an optimizer is given. Returns mean loss, AUC, probs, labels, ids."""
    training = optimizer is not None
    model.train(training)
    losses, probs, labels, ids = [], [], [], []
    with torch.set_grad_enabled(training):
        for x, y, exam_id in loader:
            x, y = x.to(device, non_blocking=True), y.to(device)
            logit = model(x)
            loss = weighted_loss(logit, y, prevalence)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            probs.append(torch.sigmoid(logit).item())
            labels.append(y.item())
            ids.extend(exam_id)
    auc = roc_auc_score(labels, probs) if 0 < sum(labels) < len(labels) else float("nan")
    return float(np.mean(losses)), auc, probs, labels, ids


def make_loader(dataset, shuffle, seed, workers):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=workers, pin_memory=True,
                      generator=generator, persistent_workers=workers > 0,
                      worker_init_fn=lambda w: np.random.seed(seed + w))


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    except OSError:
        return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--plane", choices=PLANES, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5, help="ReduceLROnPlateau patience (epochs)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-exams", type=int, default=0, help="only for a quick test: first N train and tuning exams")
    parser.add_argument("--out", type=Path, required=True, help="folder outside the repository")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda"
    run_dir = args.out / (f"{args.task}_{args.plane}_seed{args.seed}" + ("_test" if args.max_exams else ""))
    run_dir.mkdir(parents=True, exist_ok=True)

    table = load_table(MRNET_DIR, SPLITS_CSV)
    sets = {s: MRNetDataset(MRNET_DIR, table, s, args.plane, args.task, NORM_MEAN, NORM_SD, train=(s == "train"))
            for s in ["train", "tuning", "test"]}
    prevalence = {s: d.prevalence() for s, d in sets.items()}
    train_set, tuning_set = sets["train"], sets["tuning"]
    if args.max_exams:
        train_set, tuning_set = Subset(train_set, range(args.max_exams)), Subset(tuning_set, range(args.max_exams))

    model = MRNet(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, factor=0.3,
                                                           threshold=1e-4)
    info = {**{k: str(v) for k, v in vars(args).items()}, "prevalence": prevalence,
            "parameters": count_parameters(model), "torch": torch.__version__, "git_commit": git_commit(),
            "splits_md5": hashlib.md5(SPLITS_CSV.read_bytes()).hexdigest(),
            "normalisation": [NORM_MEAN, NORM_SD], "gpu": torch.cuda.get_device_name(0)}
    (run_dir / "args.json").write_text(json.dumps(info, indent=2))

    train_loader = make_loader(train_set, True, args.seed, args.workers)
    tuning_loader = make_loader(tuning_set, False, args.seed, args.workers)
    log, tuning_probs, best = [], {}, (float("inf"), -1)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_auc, *_ = run_epoch(model, train_loader, prevalence["train"], device, optimizer)
        tu_loss, tu_auc, tu_p, _, tu_ids = run_epoch(model, tuning_loader, prevalence["tuning"], device)
        scheduler.step(tu_loss)
        tuning_probs[f"epoch_{epoch}"] = tu_p
        if tu_loss < best[0]:
            best = (tu_loss, epoch)
            torch.save(model.state_dict(), run_dir / "best.pt")
        seconds = time.time() - start
        log.append({"epoch": epoch, "lr": lr, "train_loss": tr_loss, "train_auc": tr_auc,
                    "tuning_loss": tu_loss, "tuning_auc": tu_auc, "seconds": round(seconds, 1)})
        pd.DataFrame(log).to_csv(run_dir / "log.csv", index=False, float_format="%.6g")
        pd.DataFrame({"exam_id": tu_ids, **tuning_probs}).to_csv(run_dir / "tuning_probs.csv", index=False,
                                                               float_format="%.5f")
        print(f"epoch {epoch:2d} | lr {lr:.1e} | train loss {tr_loss:.4f} auc {tr_auc:.3f} | "
              f"tuning loss {tu_loss:.4f} auc {tu_auc:.3f} | {seconds:.0f} s", flush=True)

    # Predictions of the selected epoch, without augmentation.
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    rows = []
    # In a quick test (--max-exams) the test set is not touched at all.
    for split in ["train", "tuning"] + ([] if args.max_exams else ["test"]):
        plain = MRNetDataset(MRNET_DIR, table, split, args.plane, args.task, NORM_MEAN, NORM_SD, train=False)
        if args.max_exams:
            plain = Subset(plain, range(args.max_exams))
        _, _, p, y, ids = run_epoch(model, make_loader(plain, False, args.seed, args.workers),
                                    prevalence[split], device)
        rows += [{"exam_id": i, "split": split, "task": args.task, "plane": args.plane, "prob": pi,
                  "label": int(yi), "seed": args.seed, "epoch": best[1]} for i, pi, yi in zip(ids, p, y)]
    pd.DataFrame(rows).to_csv(run_dir / "predictions.csv", index=False, float_format="%.6f")
    print(f"selected epoch {best[1]} (tuning loss {best[0]:.4f}); outputs in {run_dir}")


if __name__ == "__main__":
    main()
