# -*- coding: utf-8 -*-
"""Stage 23 — the two-stage analytic formulation of the zone task: a learned
counterpart of the oracle measurement made during dataset construction.

Rationale. End-to-end multiple-instance learning does not learn zone (0.55 axial,
0.58 with slice position supplied, 0.60 coronal, against 0.50 chance), yet the
oracle — perfect stone localization from the images in the kidney-relative frame —
agrees with the reported zone 77% of the time. The gap is not missing signal; it
is the difficulty of learning the composition of "find the stone" and "name the
zone" from a single loss. This stage separates the two:

  stage 1   train the has_stone multiple-instance detector on AXIAL input. Because
            the slice index IS the z axis, the normalized position of the
            highest-scoring slice is directly the kidney-relative craniocaudal
            position of the stone. (On coronal input the slice index is y, which
            makes this construction invalid — the choice of plane is load-bearing.)
  stage 2   z_hat -> zone: two thresholds are found by grid search ON THE TRAINING
            FOLD and applied to the test fold. There is no gradient training for
            zone at all.

The zone scores use exactly the oracle definition (upper = z, lower = -z,
mid = -|z - 0.5|), so the result is directly comparable with the 0.77 ceiling.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from common.ml import data as D
from common.ml import metrics as MET
from common.ml import models as M
from common.ml import runio as R
from common.ml.seed import set_seed
from common.ml.transforms import eval_transform, train_transform

_s = importlib.util.spec_from_file_location(
    "t19", str(Path(__file__).resolve().parent / "19_train_kidney.py"))
T19 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(T19)

ZONES = ("LOWER", "MID", "UPPER")


def train_detector(tr, va, model_name, seed, epochs, bag_k, dev, eval_tfm):
    """Stage 1: the has_stone multiple-instance detector, selected at best
    validation AUC."""
    set_seed(seed)
    ds = D.KidneyBagDataset(tr, "has_stone", "stone", bag_k, True, True, train_transform())
    dl = DataLoader(ds, batch_size=T19.BS_BAG, shuffle=True, num_workers=0)
    enc, feat = M.build_encoder(model_name)
    model = M.MILWrapper(enc, feat, 1).to(dev)
    crit = T19.make_loss("has_stone", D.targets_of(tr, "has_stone"), dev)
    opt = optim.AdamW(model.parameters(), lr=T19.LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    y_va = D.targets_of(va, "has_stone")

    best, state = -1.0, None
    for _ in range(epochs):
        model.train()
        for xs, pos, y in dl:
            opt.zero_grad(set_to_none=True)
            B, K = xs.shape[:2]
            zi = model.instance_logits(xs.view(B * K, *xs.shape[2:]).to(dev))
            crit(model.pool(zi.view(B, K, -1)), T19.bag_targets("has_stone", y, dev)).backward()
            opt.step()
        pv = T19.full_stack_infer(model, va, "has_stone", "stone", True, dev, eval_tfm)
        auc = MET.binary_metrics(y_va, pv)["auc"]
        sched.step(auc)
        if auc > best:
            best = auc
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, best


def z_hat(model, df, dev, tfm, chunk=128):
    """Normalized z position of the highest-scoring slice, per kidney.
    0 is caudal, 1 is cranial.

    The axial stack is ordered by filename in the direction of increasing z (see
    stage 04), so index/(n-1) is directly the kidney-relative craniocaudal
    position. Mirroring acts on the horizontal axis and so leaves z intact.
    """
    model.eval()
    out = []
    with torch.no_grad():
        for _, row in df.iterrows():
            xs, _ = D.full_stack(row, "stone", True)
            zs = []
            for i in range(0, len(xs), chunk):
                zs.append(model.instance_logits(tfm(xs[i:i + chunk]).to(dev))[:, 0]
                          .float().cpu().numpy())
            s = np.concatenate(zs)
            out.append(float(np.argmax(s)) / max(len(s) - 1, 1))
    return np.array(out)


def zone_scores(z: np.ndarray) -> np.ndarray:
    """(N,) z -> (N,3) scores for [upper, mid, lower], using the oracle definition."""
    return np.stack([z, -np.abs(z - 0.5), -z], axis=1)


def fit_thresholds(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Choose the two thresholds by grid search on the training fold, maximizing
    accuracy. y: 0=LOWER, 1=MID, 2=UPPER. The geometric reference is 1/3 and 2/3;
    how far the fitted thresholds land from it is itself a result."""
    grid = np.arange(0.05, 0.96, 0.025)
    best, bt = -1.0, (1 / 3, 2 / 3)
    for t1 in grid:
        for t2 in grid[grid > t1]:
            pred = (z >= t1).astype(int) + (z >= t2).astype(int)
            acc = float((pred == y).mean())
            if acc > best:
                best, bt = acc, (float(t1), float(t2))
    return bt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "sst"])
    ap.add_argument("--seeds", nargs="*", type=int, default=list(D.SEEDS))
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=T19.EPOCHS)
    ap.add_argument("--bag-k", type=int, default=16)
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.folds, args.epochs = [args.seeds[0]], [args.folds[0]], 3

    R.ensure_ml_dirs()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = D.load_table()
    eval_tfm = eval_transform()
    done, rows = R.load_done()
    print(f"config={args.config} two-stage zone, model={args.model} dev={dev}", flush=True)

    for seed in args.seeds:
        for fold in args.folds:
            key = (args.config, "zone", args.model, "stone", "two_stage", "on",
                   str(seed), str(fold))
            if key in done:
                print(f"  skipped: seed {seed} fold {fold}", flush=True)
                continue
            t0 = time.time()
            tr_all, va_all, te_all = D.fold_split(df, seed, fold)
            if args.smoke:
                tr_all, va_all, te_all = tr_all.head(16), va_all.head(8), te_all.head(24)
            model, det_auc = train_detector(tr_all, va_all, args.model, seed,
                                            args.epochs, args.bag_k, dev, eval_tfm)

            # stage 2 — thresholds come only from single-zone training-fold kidneys
            tr_z = D.task_subset(tr_all, "zone")
            tr_single = tr_z[tr_z["dominant_zone"].isin(ZONES)]
            t1, t2 = fit_thresholds(z_hat(model, tr_single, dev, eval_tfm),
                                    tr_single["dominant_zone"].map(
                                        {z: i for i, z in enumerate(ZONES)}).to_numpy())

            te_z = D.task_subset(te_all, "zone")
            z_te = z_hat(model, te_z, dev, eval_tfm)
            y_ml = D.targets_of(te_z, "zone")                       # (N,3) upper/mid/lower
            m = MET.multilabel_metrics(y_ml, zone_scores(z_te))   # AUC on the scores
            te_single = te_z["dominant_zone"].isin(ZONES).to_numpy()
            y_s = te_z.loc[te_single, "dominant_zone"].map(
                {z: i for i, z in enumerate(ZONES)}).to_numpy()
            pred = (z_te[te_single] >= t1).astype(int) + (z_te[te_single] >= t2).astype(int)
            acc3 = float((pred == y_s).mean()) if len(y_s) else float("nan")
            maj = float(np.bincount(y_s, minlength=3).max() / len(y_s)) if len(y_s) else float("nan")

            del model
            torch.cuda.empty_cache()
            row = {"config": args.config, "task": "zone", "model": args.model,
                   "input": "stone", "form": "two_stage", "mirror": "on", "pos": "off",
                   "bag_k": args.bag_k, "seed": seed, "fold": fold, "epochs": args.epochs,
                   "n_test": int(len(te_z)), "best_val": round(det_auc, 4),
                   "sec": round(time.time() - t0),
                   "kidney_auc_macro": round(m["auc_macro"], 4),
                   "kidney_f1": round(m["f1"], 4),
                   "kidney_acc": round(acc3, 4), "kidney_majority": round(maj, 4),
                   "kidney_degenerate": False,
                   "thr_lo": round(t1, 3), "thr_hi": round(t2, 3),
                   "n_single": int(te_single.sum()), "det_val_auc": round(det_auc, 4)}
            rows.append(row)
            R.save_rows(rows)
            R.save_rows([{"kidney_id": k, "patient_id": p, "side": s,
                          "z_hat": round(float(z), 4),
                          "y": json.dumps(np.asarray(yy).tolist()),
                          "dominant_zone": dz}
                         for k, p, s, z, yy, dz in zip(
                             te_z["kidney_id"], te_z["patient_id"], te_z["side"],
                             z_te, y_ml, te_z["dominant_zone"])],
                        R.preds_path(args.config, "zone", seed, fold))
            print(f"  -> tohum {seed} kat {fold}: makro AUC {m['auc_macro']:.4f}  "
                  f"3-class acc {acc3:.4f} (baseline {maj:.3f}, thr {t1:.2f}/{t2:.2f}, "
                  f"detector val AUC {det_auc:.3f}, {row['sec']} s)", flush=True)
    print(f"\nwrote: {R.RUNS_CSV} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
