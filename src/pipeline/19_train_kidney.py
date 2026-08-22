# -*- coding: utf-8 -*-
"""Stage 19 — kidney-level training (has_stone / size3 / zone).

Protocol: the unit is the kidney and the split is by patient, using the cv5_seed*
fold columns of the kidney table. Hyperparameters are carried over unchanged from
the earlier pipeline, so that the difference measured here is attributable to the
data and the formulation rather than to retuning: 40 epochs, AdamW with lr 1e-4
and weight decay 1e-4, ReduceLROnPlateau(max, 0.5, patience 4).

Training uses bags of K slices sampled stratified-uniformly; evaluation always
uses ALL slices of the kidney with LSE pooling. The stage is interruption
tolerant: the run row and the prediction dump are written after every
(seed, fold) cell, and a completed cell is skipped on restart.

  --form mil       : loss on the bag logit, after pooling — the primary form
  --form instance  : loss on the slice logits, each slice inheriting the kidney
                     label; evaluation is still full-stack LSE. This is the
                     comparison arm and rests on a deliberately weaker assumption,
                     since a stone appears on only a few slices of the stack and
                     most slices of a positive kidney do not show it.
"""
from __future__ import annotations

import argparse
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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from common.ml import data as D
from common.ml import metrics as MET
from common.ml import models as M
from common.ml import runio as R
from common.ml.seed import set_seed
from common.ml.transforms import eval_transform, train_transform

EPOCHS, LR, BS_BAG = 40, 1e-4, 4
N_OUT = {"has_stone": 1, "size3": 2, "zone": 3}
VAL_METRIC = {"has_stone": "auc", "size3": "f1", "zone": "auc_macro"}


def full_stack_infer(model, df, task, input_kind, mirror, dev, tfm, chunk=128):
    """Every slice of each kidney -> slice logits -> numpy LSE -> bag probabilities."""
    model.eval()
    bag_logits = []
    with torch.no_grad():
        for _, row in df.iterrows():
            xs, pos = D.full_stack(row, input_kind, mirror)
            zs = []
            for i in range(0, len(xs), chunk):
                x = tfm(xs[i:i + chunk]).to(dev)
                zs.append(model.instance_logits(x, pos[i:i + chunk].to(dev))
                          .float().cpu().numpy())
            bag_logits.append(M.numpy_lse(np.concatenate(zs)))
    bl = np.stack(bag_logits)                                    # (N, C)
    if task == "has_stone":
        return 1.0 / (1.0 + np.exp(-bl[:, 0]))                   # (N,)
    if task == "size3":
        return M.ordinal_to_class(torch.from_numpy(bl))[1].numpy()   # (N,3)
    return 1.0 / (1.0 + np.exp(-bl))                             # zone (N,3)


def eval_metrics(task, y, probs):
    if task == "has_stone":
        return MET.binary_metrics(y, probs)
    if task == "size3":
        return MET.ordinal_metrics(y, probs)
    return MET.multilabel_metrics(y, probs)


def make_loss(task, y_tr, dev):
    if task == "has_stone":
        pos = max(float(y_tr.mean()), 1e-6)
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=dev))
    if task == "size3":
        return nn.BCEWithLogitsLoss()
    pos = np.clip(y_tr.mean(axis=0), 1e-6, 1 - 1e-6)             # zone (3,)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos,
                                                        dtype=torch.float32, device=dev))


def bag_targets(task, y, dev):
    if task == "has_stone":
        return y.float().to(dev).unsqueeze(1)                    # (B,1)
    if task == "size3":
        return M.ordinal_targets(y.long()).to(dev)               # (B,2)
    return y.float().to(dev)                                     # zone (B,3)


def run_one(df, args, task, seed, fold, dev, eval_tfm):
    tr, va, te = (D.task_subset(s, task) for s in D.fold_split(df, seed, fold))
    if args.smoke:
        tr, va, te = tr.head(16), va.head(8), te.head(8)
    set_seed(seed)
    mirror = args.mirror == "on"
    ds_tr = D.KidneyBagDataset(tr, task, args.input, args.bag_k, True, mirror, train_transform())
    dl_tr = DataLoader(ds_tr, batch_size=BS_BAG, shuffle=True, num_workers=0)

    enc, feat = M.build_encoder(args.model)
    model = M.MILWrapper(enc, feat, N_OUT[task], use_pos=args.pos == "on").to(dev)
    crit = make_loss(task, D.targets_of(tr, task), dev)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)

    y_va, y_te = D.targets_of(va, task), D.targets_of(te, task)
    best, state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        for xs, pos, y in dl_tr:                                  # xs (B,K,3,224,224)
            opt.zero_grad(set_to_none=True)
            B, K = xs.shape[:2]
            zi = model.instance_logits(xs.view(B * K, *xs.shape[2:]).to(dev),
                                       pos.view(B * K).to(dev))
            tgt = bag_targets(task, y, dev)
            if args.form == "mil":
                loss = crit(model.pool(zi.view(B, K, -1)), tgt)
            else:                                    # instance: slice inherits the label
                loss = crit(zi, tgt.repeat_interleave(K, dim=0))
            loss.backward()
            opt.step()
        pv = full_stack_infer(model, va, task, args.input, mirror, dev, eval_tfm)
        vm = eval_metrics(task, y_va, pv)[VAL_METRIC[task]]
        sched.step(vm)
        if vm > best:
            best = vm
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"    epoch {ep + 1:02d}  val {VAL_METRIC[task]} {vm:.4f}"
              f"{'  *' if vm == best else ''}", flush=True)

    model.load_state_dict(state)
    pt = full_stack_infer(model, te, task, args.input, mirror, dev, eval_tfm)
    m = eval_metrics(task, y_te, pt)
    if args.save_ckpt:
        torch.save(state, R.CKPT / f"{args.config}_{task}_s{seed}_f{fold}.pt")
    del model
    torch.cuda.empty_cache()
    return te, y_te, pt, m, best


def dump_preds(path, te, y, probs, task):
    rows = []
    for i, (_, r) in enumerate(te.iterrows()):
        rows.append({
            "kidney_id": r["kidney_id"], "patient_id": r["patient_id"], "side": r["side"],
            "mask_plausible": str(r["mask_plausible"]),
            "y": json.dumps(np.asarray(y[i]).tolist()),
            "prob": json.dumps(np.round(np.asarray(probs[i], dtype=float), 6).tolist()),
        })
    R.save_rows(rows, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="has_stone", choices=list(N_OUT))
    ap.add_argument("--input", default="mw3", choices=["stone", "mw3", "cor"])
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "sst"])
    ap.add_argument("--form", default="mil", choices=["mil", "instance"])
    ap.add_argument("--mirror", default="on", choices=["on", "off"])
    ap.add_argument("--pos", default="off", choices=["on", "off"],
                    help="append the normalized slice z position to the features")
    ap.add_argument("--bag-k", type=int, default=16)
    # The default is the FULL protocol (5 seeds); use --seeds or --smoke for a
    # quick single-seed probe. Stage 23 must carry the same default: the two
    # diverged once, and has_stone and size3 silently received 5 runs, not 25.
    ap.add_argument("--seeds", nargs="*", type=int, default=list(D.SEEDS))
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--config", required=True)
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.folds, args.epochs = [args.seeds[0]], [args.folds[0]], 3

    R.ensure_ml_dirs()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = D.load_table()
    eval_tfm = eval_transform()
    done, rows = R.load_done()
    if done:
        print(f"resuming: {len(done)} runs already complete", flush=True)
    print(f"config={args.config} task={args.task} model={args.model} input={args.input} "
          f"form={args.form} mirror={args.mirror} pos={args.pos} K={args.bag_k} dev={dev}",
          flush=True)

    for seed in args.seeds:
        for fold in args.folds:
            key = (args.config, args.task, args.model, args.input, args.form,
                   args.mirror, str(seed), str(fold))
            if key in done:
                print(f"  skipped: seed {seed} fold {fold}", flush=True)
                continue
            t0 = time.time()
            print(f"  tohum {seed} kat {fold}", flush=True)
            te, y_te, pt, m, best_val = run_one(df, args, args.task, seed, fold, dev, eval_tfm)
            pm = (MET.patient_binary(te["patient_id"].to_numpy(), y_te, pt)
                  if args.task == "has_stone" else None)
            row = {"config": args.config, "task": args.task, "model": args.model,
                   "input": args.input, "form": args.form, "mirror": args.mirror,
                   "pos": args.pos,
                   "bag_k": args.bag_k, "seed": seed, "fold": fold, "epochs": args.epochs,
                   "n_test": m["n"], "best_val": round(best_val, 4),
                   "sec": round(time.time() - t0)}
            row.update({f"kidney_{k}": (round(v, 4) if isinstance(v, float) else v)
                        for k, v in m.items() if not isinstance(v, list)})
            if pm:
                row.update({f"patient_{k}": (round(v, 4) if isinstance(v, float) else v)
                            for k, v in pm.items() if not isinstance(v, list)})
            rows.append(row)
            R.save_rows(rows)
            dump_preds(R.preds_path(args.config, args.task, seed, fold), te, y_te, pt, args.task)
            main_metric = m.get("auc", m.get("auc_macro", m.get("f1")))
            print(f"  -> tohum {seed} kat {fold}: test {main_metric:.4f} "
                  f"(val {best_val:.4f}, {row['sec']} sn)", flush=True)
    print(f"\nwrote: {R.RUNS_CSV} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
