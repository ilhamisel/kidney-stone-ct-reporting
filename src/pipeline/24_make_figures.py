# -*- coding: utf-8 -*-
"""Stage 24 — result and data figures.

Produces seven figures:
  fig1  half-crop atlas: slices of stone-positive against stone-negative kidneys
  fig2  coronal crop with the kidney-relative zone bands
  fig3  Grad-CAM: what the model based its decision on
  fig4  result summary: mean with 95% CI per configuration
  fig5  zone mechanism: the two-stage z_hat distribution and learned thresholds
  fig6  size3 confusion matrix, row-normalized
  fig7  the end-to-end pipeline for one patient: axial -> crop -> coronal ->
        per-slice evidence -> structured facts -> generated report

Drawing conventions: an ordinal single-hue ramp for zone (it is an ordered axis,
not a categorical one), one axis per panel, thin marks, the grid behind the data,
and labels placed directly rather than in a legend where that is possible.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec
from scipy import stats

from common.ml import data as D
from common.ml import models as M
from common.ml import viz as V
from common.ml.runio import CKPT, PREDS, RESULTS
from common.ml.transforms import eval_transform
from common.paths import ARCHIVE, LABELS, ROOT

OUT = ROOT / "10_examples" / "figures"
ZONES = ("LOWER", "MID", "UPPER")
SIZE_EN = {"MIKROLITIYAZIS": "MICROLITHIASIS", "KUCUK": "SMALL", "ORTA": "MEDIUM",
           "BUYUK": "LARGE", "COK_BUYUK": "VERY LARGE"}
SIZE3_EN = ("SMALL\n(≤5 mm)", "MEDIUM\n(6–10 mm)", "LARGE\n(≥11 mm)")


def imshow(ax, img, title=None):
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True), s.set_color(V.AXIS), s.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=8, pad=3, color=V.INK)


def tag(ax, txt, x=0.04, y=0.035, va="bottom", size=7.5):
    ax.text(x, y, txt, transform=ax.transAxes, ha="left", va=va, fontsize=size,
            color="#ffffff", bbox=dict(boxstyle="round,pad=0.25", fc="#000000b0", ec="none"))


def head(fig, title, sub=None, note=None, x=0.012, top=0.88):
    fig.suptitle(title, x=x, ha="left", fontsize=11, color=V.INK, weight="semibold")
    if sub:
        fig.text(x, top + 0.045, sub, ha="left", fontsize=8, color=V.INK2)
    if note:
        fig.text(x, top + 0.008, note, ha="left", fontsize=7.5, color=V.MUTED)


def matched_cand(row):
    """The candidate matched to a reported stone for this kidney, or None."""
    doc = json.loads((LABELS / "reports" / f"{row['case_id']}.json").read_text(encoding="utf-8"))
    cs = [c for c in doc.get("stone_candidates", [])
          if c.get("side") == row["side"] and c.get("matched_stone_index")]
    return cs[0] if cs else None


def peak_slice(row, kind="stone"):
    """Index within the stack of the slice holding the stone: from the matched
    candidate where one exists, otherwise the brightest slice."""
    files = sorted(Path(row["halfcrop_stone_dir"]).glob("*.png"))
    cand = matched_cand(row)
    if cand:
        want = int(round(cand["centroid_voxel"][2]))            # eksen 2 = kesit
        return min(range(len(files)), key=lambda i: abs(int(files[i].stem) - want))
    st = D.get_stack(row, kind)[:, 0].numpy()
    return int(np.argmax((st > 200).sum(axis=(1, 2))))


def load_detector(dev):
    ckpt = CKPT / "fig_ckpt_has_stone_s1337_f0.pt"
    if not ckpt.exists():
        return None
    enc, feat = M.build_encoder("resnet18")
    model = M.MILWrapper(enc, feat, 1).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()
    return model


def slice_scores(model, row, dev, tfm, chunk=128):
    """(position 0..1, slice probabilities, bag probability), in the mirrored
    space. Feeding the model unmirrored input here is a silent error: left
    kidneys then score near zero and the figure looks plausible but is wrong."""
    st, pos = D.full_stack(row, "stone", mirror=True)
    with torch.no_grad():
        z = np.concatenate([
            model.instance_logits(tfm(st[i:i + chunk]).to(dev))[:, 0].float().cpu().numpy()
            for i in range(0, len(st), chunk)])
    p_bag = float(1 / (1 + np.exp(-M.numpy_lse(z[:, None])[0])))
    return pos.numpy(), 1 / (1 + np.exp(-z)), p_bag, st


# ------------------------------------------------------------------ fig 1
def fig1_atlas(df):
    pos = df[df["has_stone"] & df["max_class"].notna()].sort_values("max_mm", ascending=False)
    # Per class, choose a kidney whose stone can actually be LOCALIZED in the image.
    picks = []
    for c in ("COK_BUYUK", "BUYUK", "ORTA", "KUCUK", "MIKROLITIYAZIS"):
        sub = pos[pos["max_class"] == c]
        loc = [i for i, (_, r) in enumerate(sub.iterrows()) if matched_cand(r) is not None]
        picks.append(sub.iloc[[loc[0]]] if loc else sub.head(1))
    pick_pos = pd.concat(picks)
    neg = df[~df["has_stone"]].head(5)

    fig, axes = plt.subplots(2, 5, figsize=(9.5, 4.6))
    for j, (_, r) in enumerate(pick_pos.iterrows()):
        st = D.get_stack(r, "stone")
        imshow(axes[0, j], st[peak_slice(r)].permute(1, 2, 0).numpy(),
               f"{SIZE_EN[r['max_class']]} · {r['max_mm']:.0f} mm")
        tag(axes[0, j], r["kidney_id"])
    for j, (_, r) in enumerate(neg.iterrows()):
        st = D.get_stack(r, "stone")
        imshow(axes[1, j], st[len(st) // 2].permute(1, 2, 0).numpy())
        tag(axes[1, j], r["kidney_id"])
    axes[0, 0].set_ylabel("stone", color=V.INK, fontsize=9, labelpad=8)
    axes[1, 0].set_ylabel("no stone", color=V.INK, fontsize=9, labelpad=8)
    head(fig, "Per-kidney half-crop inputs — stone window (WL 400 / WW 1500)",
         "top: largest stone of each size class · bottom: kidneys with no reported "
         "stone (mid slice)", top=0.875)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "fig1_halfcrop_atlas.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 2
def fig2_coronal(df):
    """Examples are chosen using the stage 08b measurement: the reported zone and
    the measured focus must fall in the same third."""
    zf = pd.read_csv(ROOT / "logs" / "08b_zone_frame.csv")
    zf["kidney_id"] = zf["case_id"] + "_" + zf["side"].str[0]
    zf["third"] = np.where(zf["z_kidney"] >= 2 / 3, "UPPER",
                           np.where(zf["z_kidney"] >= 1 / 3, "MID", "LOWER"))
    ok = zf[zf["third"] == zf["zone"]]
    tbl = df.set_index("kidney_id")

    picks = []
    for z in ZONES:
        c = ok[ok["zone"] == z].sort_values("size_mm", ascending=False)
        kid = c.iloc[0]["kidney_id"]
        picks.append((tbl.loc[kid].rename(kid), float(c.iloc[0]["z_kidney"]),
                      float(c.iloc[0]["size_mm"])))

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 4.0))
    for ax, (r, zk, mm) in zip(axes, picks):
        draw_coronal(ax, r, focus=zk)
        ax.set_title(f"reported zone: {r['dominant_zone']}", fontsize=9)
        ax.text(0.5, -0.045, f"{r.name} · {mm:.0f} mm · measured position {zk:.2f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=V.INK2)
    head(fig, "Coronal kidney crops with kidney-relative zone bands",
         "bands are thirds of the kidney's OWN craniocaudal extent (not of the body) "
         "— the frame correction of §3",
         "◄ orange marker: stone focus measured from the image (08b)", top=0.855)
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.savefig(OUT / "fig2_coronal_zones.png", bbox_inches="tight")
    plt.close(fig)


def draw_coronal(ax, row, focus=None, band_labels=True):
    """Coronal crop with the zone bands. `focus` is the within-kidney position in
    0..1, marked with an arrow."""
    cand = matched_cand(row)
    files = sorted(Path(row["corcrop_dir"]).glob("*.png"))
    if cand:
        want = int(round(cand["centroid_voxel"][0]))            # axis 0 is the row axis
        best = min(files, key=lambda f: abs(int(f.stem) - want))
    else:
        best = max(files, key=lambda f: (cv2.imread(str(f), 0) > 200).sum())
    img = cv2.imread(str(best), 0)
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="auto")
    h = img.shape[0]
    for k, zn in enumerate(("UPPER", "MID", "LOWER")):          # top of image is cranial
        ax.add_patch(plt.Rectangle((-0.075, 1 - (k + 1) / 3), 0.045, 1 / 3,
                                   transform=ax.transAxes, clip_on=False,
                                   color=V.ZONE_RAMP[zn], lw=0))
        if band_labels:
            ax.text(-0.095, 1 - (2 * k + 1) / 6, zn, transform=ax.transAxes,
                    ha="right", va="center", fontsize=8, color=V.ZONE_RAMP[zn], weight="bold")
        if k:
            ax.axhline(k * h / 3, color=V.SURFACE, lw=1.3, alpha=0.85)
    if focus is not None:
        ax.plot([img.shape[1] * 0.985], [(1 - focus) * h], marker="<", ms=7,
                color=V.SERIES[1], clip_on=False, zorder=5)
    ax.set_xticks([]), ax.set_yticks([])
    return img


# ------------------------------------------------------------------ fig 3
def fig3_gradcam(df, model, dev, tfm):
    if model is None:
        print("  ATLANDI fig3: checkpoint yok")
        return
    acts, grads = {}, {}
    layer = model.encoder.layer4
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))

    _, _, te = D.fold_split(df, 1337, 0)          # the fold the model did not see
    pos = te[te["has_stone"] & te["max_mm"].notna()].sort_values("max_mm", ascending=False)
    picks = list(pos.head(3).iterrows()) + list(te[~te["has_stone"]].head(1).iterrows())

    fig, axes = plt.subplots(2, 4, figsize=(9.5, 5.2))
    for j, (_, r) in enumerate(picks):
        # Which slice drives the decision? LSE converges on the highest slice
        # logit, so the MODEL'S OWN arg-max slice is shown rather than the slice
        # where the stone is known to be — otherwise the figure would illustrate
        # the label, not the model. Input is in the mirrored space.
        _, p_slices, p_bag, st = slice_scores(model, r, dev, tfm)
        k = int(np.argmax(p_slices))
        x = tfm(st[k:k + 1]).to(dev).requires_grad_(True)
        model.zero_grad()
        logit = model.instance_logits(x)[0, 0]
        logit.backward()
        w = grads["g"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * acts["a"]).sum(1))[0].detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = cam / cam.max() if cam.max() > 0 else cam
        base = st[k].permute(1, 2, 0).numpy()
        lbl = "stone" if r["has_stone"] else "no stone"
        imshow(axes[0, j], base, f"{r['kidney_id']} · {lbl}")
        tag(axes[0, j], f"{r['max_mm']:.0f} mm" if r["has_stone"] else "none reported")
        axes[1, j].imshow(base.mean(2), cmap="gray", vmin=0, vmax=255)
        axes[1, j].imshow(cam, cmap="inferno", alpha=0.45)
        axes[1, j].set_xticks([]), axes[1, j].set_yticks([])
        axes[1, j].text(0.04, 0.955, f"kidney P = {p_bag:.3f}\nthis slice p = {p_slices[k]:.3f}",
                        transform=axes[1, j].transAxes, ha="left", va="top",
                        fontsize=8.5, weight="semibold", linespacing=1.5,
                        color=V.GOOD if (p_bag > 0.5) == bool(r["has_stone"]) else V.CRITICAL,
                        bbox=dict(boxstyle="round,pad=0.28", fc="#000000cc", ec="none"))
    axes[0, 0].set_ylabel("input", color=V.INK, fontsize=9, labelpad=8)
    axes[1, 0].set_ylabel("Grad-CAM", color=V.INK, fontsize=9, labelpad=8)
    h1.remove(), h2.remove()
    head(fig, "Grad-CAM — what drives the model's decision?",
         "held-out fold 0 (the model never saw these kidneys) · ResNet-18 layer4 · "
         "the model's own highest-scoring slice",
         "images are shown in the model's input space: LEFT kidneys are mirrored to "
         "the canonical side", top=0.865)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(OUT / "fig3_gradcam.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 4
def fig4_results():
    runs = pd.read_csv(RESULTS / "runs.csv")
    panels = [
        ("has_stone — kidney-level AUC", "kidney_auc",
         [("e9_hs_sst", "SST"), ("e9_hs_resnet18", "ResNet-18")], 0.5, "chance"),
        ("size3 — 3-class accuracy", "kidney_acc",
         [("e9_size_sst", "SST"), ("e9_size_resnet18", "ResNet-18")], 0.419, "majority"),
        ("zone — macro AUC", "kidney_auc_macro",
         [("e9_2stage_sst", "two-stage, SST"),
          ("e9_2stage_resnet18", "two-stage, ResNet-18"),
          ("e6_zonecor_resnet18", "end-to-end, coronal*"),
          ("e5_zonepos_resnet18", "end-to-end + z-position"),
          ("e4_zone_resnet18", "end-to-end, axial")], 0.5, "chance"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5),
                             gridspec_kw={"width_ratios": [1, 1, 1.5]})
    for ax, (title, col, cfgs, base, base_lbl) in zip(axes, panels):
        ys, ms, los, his, labels = [], [], [], [], []
        for i, (cfg, lbl) in enumerate(cfgs):
            g = runs[(runs["config"] == cfg) & runs[col].notna()][col].astype(float)
            if not len(g):
                continue
            m = g.mean()
            h = (stats.t.ppf(0.975, len(g) - 1) * g.std(ddof=1) / np.sqrt(len(g))
                 if len(g) > 1 else 0)
            ys.append(-i), ms.append(m), los.append(m - h), his.append(m + h)
            labels.append(f"{lbl}  (n={len(g)})")
        ax.axvline(base, color=V.MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(base, 0.6, f" {base_lbl}", fontsize=7.5, color=V.MUTED, va="top")
        ax.hlines(ys, los, his, color=V.SERIES[0], lw=2, zorder=2)
        ax.scatter(ms, ys, s=42, color=V.SERIES[0], zorder=3, edgecolor=V.SURFACE, lw=1.2)
        for y, m in zip(ys, ms):
            ax.annotate(f"{m:.3f}", (m, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8, color=V.INK, weight="semibold")
        ax.set_yticks(ys, labels, fontsize=8)
        ax.set_ylim(min(ys) - 0.7, 0.9)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("mean ± 95% CI", fontsize=8)
    head(fig, "Full-protocol results for the three tasks (5 seeds × 5 folds = 25 runs)",
         None, "* single seed (5 runs) · the end-to-end zone arms are ablations run on the full 394-kidney cohort", x=0.008, top=0.885)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT / "fig4_results.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 5
def fig5_zone_mechanism():
    runs = pd.read_csv(RESULTS / "runs.csv")
    g = runs[runs["config"] == "e9_2stage_sst"]
    t1, t2 = g["thr_lo"].mean(), g["thr_hi"].mean()
    parts = [pd.read_csv(f) for f in sorted(PREDS.glob("e9_2stage_sst_zone_s1337_f*.csv"))]
    p = pd.concat(parts)
    p = p[p["dominant_zone"].isin(ZONES)]

    fig, ax = plt.subplots(figsize=(8.0, 3.9))
    rng = np.random.default_rng(0)
    for i, z in enumerate(ZONES):
        v = p.loc[p["dominant_zone"] == z, "z_hat"].to_numpy()
        ax.scatter(v, np.full(len(v), i) + rng.uniform(-0.15, 0.15, len(v)),
                   s=22, color=V.ZONE_RAMP[z], alpha=0.75, lw=0.6, edgecolor=V.SURFACE)
        ax.scatter([v.mean()], [i], marker="|", s=460, color=V.INK, lw=1.8, zorder=4)
        ax.annotate(f"mean {v.mean():.2f}", (v.mean(), i), textcoords="offset points",
                    xytext=(0, -23), ha="center", fontsize=8, color=V.INK, weight="semibold")
    for t in (1 / 3, 2 / 3):
        ax.axvline(t, color=V.MUTED, lw=0.9, ls=(0, (1, 2)), zorder=1)
    for t, ha in ((t1, "right"), (t2, "left")):
        ax.axvline(t, color=V.INK, lw=1.1, ls=(0, (3, 2)), zorder=2)
        ax.text(t + (-0.012 if ha == "right" else 0.012), 2.9, f"{t:.2f}",
                fontsize=8, color=V.INK, va="top", ha=ha, weight="semibold")
    ax.set_yticks(range(3), list(ZONES), fontsize=9)
    ax.set_ylabel("reported zone", fontsize=9)
    ax.set_xlabel("position of the detector's highest-scoring slice within the kidney  "
                  "(0 = caudal, 1 = cranial)", fontsize=8.5)
    ax.set_xlim(-0.03, 1.03), ax.set_ylim(-0.62, 2.95)
    ax.grid(axis="y", visible=False)
    head(fig, "Two-stage zone estimation: the mechanism",
         "seed 1337, five folds pooled (every kidney is in the test set exactly once) · "
         "one point per kidney",
         "dashed: learned thresholds  ·  dotted: geometric 1/3 and 2/3", x=0.008, top=0.845)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.savefig(OUT / "fig5_zone_mechanism.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 6
def fig6_confusion():
    cm = np.zeros((3, 3))
    for f in PREDS.glob("e9_size_sst_size3_s*_f*.csv"):
        d = pd.read_csv(f)
        for _, r in d.iterrows():
            y = int(r["y"] if not isinstance(r["y"], str) else json.loads(r["y"]))
            cm[y, int(np.argmax(json.loads(r["prob"])))] += 1
    row = cm / cm.sum(1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.9, 4.2))
    ax.imshow(row, cmap=seq_cmap(), vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{row[i, j]:.2f}\n{int(cm[i, j])}", ha="center", va="center",
                    fontsize=9, weight="semibold" if i == j else "normal",
                    color="#ffffff" if row[i, j] > 0.55 else V.INK)
    ax.set_xticks(range(3), SIZE3_EN, fontsize=8)
    ax.set_yticks(range(3), SIZE3_EN, fontsize=8)
    ax.set_xlabel("predicted", fontsize=9), ax.set_ylabel("reported", fontsize=9)
    ax.grid(False)
    head(fig, "Stone size class — row-normalized confusion matrix",
         "SST, pooled over 25 runs (5 seeds × 5 folds) · cell: rate and kidney count",
         "errors fall into adjacent classes — the ordinal structure is preserved", top=0.845)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.savefig(OUT / "fig6_size_confusion.png", bbox_inches="tight")
    plt.close(fig)


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", [V.SURFACE] + V.SEQ)


# ------------------------------------------------------------------ fig 7
def pick_case(df, model, dev, tfm):
    """Choose a case for the end-to-end figure: from the fold 0 test set, so the
    model has not seen it; both kidneys segmented; at least one characterized
    stone; a single zone; and the model correct on both sides."""
    _, _, te = D.fold_split(df, 1337, 0)
    best = None
    for case_id, g in te.groupby("case_id"):
        if len(g) != 2:
            continue
        pos = g[g["has_stone"] & g["max_mm"].notna() & g["dominant_zone"].isin(ZONES)]
        if not len(pos):
            continue
        ok, score = True, 0.0
        for _, r in g.iterrows():
            _, _, p_bag, _ = slice_scores(model, r, dev, tfm)
            if (p_bag > 0.5) != bool(r["has_stone"]):
                ok = False
                break
            score += p_bag if r["has_stone"] else 1 - p_bag
        if ok and (best is None or score > best[1]):
            best = (case_id, score)
    return best[0] if best else te["case_id"].iloc[0]


def fig7_pipeline(df, model, dev, tfm, case_id=None, out_name="fig7_pipeline.png",
                  title=None):
    """The end-to-end pipeline figure. With no case_id one is chosen
    automatically; with a case_id that case is drawn, which is how stage 28 calls
    this for the appendix cards."""
    if model is None:
        print("  ATLANDI fig7: checkpoint yok")
        return
    case_id = case_id or pick_case(df, model, dev, tfm)
    g = df[df["case_id"] == case_id].set_index("side", drop=False)  # keep the 'side' column
    doc = json.loads((LABELS / "reports" / f"{case_id}.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(16.0, 10.0))
    gs = GridSpec(2, 6, figure=fig, height_ratios=[0.95, 1.15],
                  hspace=0.26, wspace=0.30, left=0.042, right=0.988, top=0.865, bottom=0.055)

    # --- 1. axial volume: the slice showing the stone, plus the kidney boxes
    ax = fig.add_subplot(gs[0, 0:2])
    ref = g.loc["RIGHT"] if g.loc["RIGHT"]["has_stone"] else g.loc["LEFT"]
    cand = matched_cand(ref)
    crops = {s: doc["targets"]["kidneys"][s.lower()]["crop"] for s in ("RIGHT", "LEFT")}
    s_idx = (int(round(cand["centroid_voxel"][2])) if cand
             else (crops["RIGHT"]["bbox_voxel_padded"][4] + crops["RIGHT"]["bbox_voxel_padded"][5]) // 2)
    meta = json.loads((ARCHIVE / case_id / "axial_std" / "meta.json").read_text(encoding="utf-8"))
    png = cv2.imread(str(ARCHIVE / case_id / "axial_std" / f"{s_idx:04d}.png"), cv2.IMREAD_UNCHANGED)
    from common.hu import WINDOWS, window8
    img = window8(png.astype(np.int32) - meta["hu_offset"], *WINDOWS["stone"])
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    for side, col in (("RIGHT", V.SERIES[0]), ("LEFT", V.SERIES[2])):
        r0, r1, c0, c1, *_ = crops[side]["bbox_voxel_padded"]
        ax.add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec=col, lw=1.6))
        ax.text(c0, r0 - 6, side, color=col, fontsize=8, weight="bold")
    if cand:
        ax.plot([cand["centroid_voxel"][1]], [cand["centroid_voxel"][0]], marker="o",
                ms=13, mfc="none", mec=V.SERIES[1], mew=1.8)
    # Crop the field-of-view padding; the square image left wide black margins
    ys, xs = np.where(img > 12)
    if len(xs):
        m = 8
        ax.set_xlim(max(xs.min() - m, 0), min(xs.max() + m, img.shape[1]))
        ax.set_ylim(min(ys.max() + m, img.shape[0]), max(ys.min() - m, 0))
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title("1.  Axial CT volume — lossless 16-bit HU archive", fontsize=9)
    ax.text(0.5, -0.035, f"{case_id} · slice {s_idx} of {meta['n_slices']} · "
            f"stone window applied · TotalSegmentator kidney boxes",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=V.INK2)

    # --- 2. axial half-crops: one independent problem per kidney
    for k, (side, col) in enumerate((("RIGHT", V.SERIES[0]), ("LEFT", V.SERIES[2]))):
        a = fig.add_subplot(gs[0, 2 + k])
        r = g.loc[side]
        st = D.get_stack(r, "stone")
        imshow(a, st[peak_slice(r)].permute(1, 2, 0).numpy())
        a.set_title(f"2.  {side} half-crop" if k == 0 else f"2.  {side} half-crop",
                    fontsize=9, color=col)
        tag(a, f"{r['n_crop_slices']} slices · {r['crop_h']}×{r['crop_w']} px")

    # --- 3. coronal reformat with the zone bands
    for k, (side, col) in enumerate((("RIGHT", V.SERIES[0]), ("LEFT", V.SERIES[2]))):
        a = fig.add_subplot(gs[0, 4 + k])
        r = g.loc[side]
        draw_coronal(a, r, band_labels=(k == 0))
        a.set_title(f"3.  {side} coronal + zones", fontsize=9, color=col)

    # --- 4. per-slice stone evidence; both has_stone and zone are read from it
    axc = fig.add_subplot(gs[1, 0:2])
    for n, (side, col) in enumerate((("RIGHT", V.SERIES[0]), ("LEFT", V.SERIES[2]))):
        r = g.loc[side]
        pos_n, p_sl, p_bag, _ = slice_scores(model, r, dev, tfm)
        axc.plot(pos_n, p_sl, color=col, lw=1.8, label=f"{side}  (kidney P = {p_bag:.3f})")
        i = int(np.argmax(p_sl))
        axc.scatter([pos_n[i]], [p_sl[i]], s=52, color=col, zorder=4,
                    edgecolor=V.SURFACE, lw=1.4)
        if r["has_stone"]:
            # both sides can peak, so the labels are offset in opposite directions
            axc.annotate(f"argmax ẑ = {pos_n[i]:.2f}", (pos_n[i], p_sl[i]),
                         textcoords="offset points",
                         xytext=((8, 9) if n == 0 else (-8, -17)),
                         ha="left" if n == 0 else "right",
                         fontsize=8, color=col, weight="semibold")
    for k, zn in enumerate(ZONES):
        axc.axvspan(k / 3, (k + 1) / 3, color=V.ZONE_RAMP[zn], alpha=0.10, lw=0, zorder=0)
        # zone names go BELOW the axis: curves running near zero overlapped them
        axc.text((2 * k + 1) / 6, -0.085, zn, ha="center", va="center", fontsize=7.5,
                 color=V.ZONE_RAMP[zn], weight="bold")
    axc.axhline(0.5, color=V.MUTED, lw=0.9, ls=(0, (4, 3)))
    axc.set_xlim(0, 1), axc.set_ylim(-0.14, 1.16)
    axc.set_yticks([0, 0.25, 0.5, 0.75, 1.0])          # keep the zone strip off-axis
    axc.set_xlabel("position within the kidney  (0 = caudal, 1 = cranial)", fontsize=8.5)
    axc.set_ylabel("per-slice P(stone)", fontsize=8.5)
    axc.set_title("4.  Per-slice stone evidence  →  presence (max) and zone (its position)",
                  fontsize=9)
    axc.legend(loc="center left", fontsize=8)
    axc.grid(axis="x", visible=False)
    axc.text(1.0, -0.145, "each separate peak is a distinct stone along the kidney axis",
             transform=axc.transAxes, ha="right", va="top", fontsize=7.5, color=V.MUTED)

    # --- 5. reference facts as a structured table | 6. the model's end-to-end output
    W = 52
    per_side = {}
    for side in ("RIGHT", "LEFT"):
        r = g.loc[side]
        _, p_sl, p_bag, _ = slice_scores(model, r, dev, tfm)
        zhat = float(np.argmax(p_sl)) / max(len(p_sl) - 1, 1)
        s3 = predicted_size3(r["kidney_id"])
        present = p_bag > 0.5
        per_side[side] = {"present": present, "p": p_bag, "zhat": zhat,
                          "zone": ZONES[(zhat >= 0.344) + (zhat >= 0.688)] if present else None,
                          "cls": SIZE3_TO_CLS.get(s3) if present else None}

    lab = doc["labels"]
    gt = ["REPORTED STONES", "─" * W,
          f"{'KIDNEY':<8}{'SIZE':<9}{'CLASS':<13}{'ZONE':<7}"]
    for s in sorted(lab["stones"], key=lambda s: (s["side"], -(s["size_mm"] or 0))):
        size = f"{s['size_mm']:.0f} mm" if s["size_known"] else "unknown"
        cls = SIZE_EN.get(s["size_class"], "—")
        gt.append(f"{s['side']:<8}{size:<9}{cls:<13}{s['zone'] or '—':<7}")
    for side in ("RIGHT", "LEFT"):
        kd = doc["targets"]["kidneys"][side.lower()]
        if kd["has_stone"] and not kd["n_characterized"]:
            gt.append(f"{side:<8}{'unknown':<9}{'—':<13}{'—':<7}")
    gt += ["", "CASE-LEVEL FACTS", "─" * W]
    sides_with = [s.upper() for s in ("right", "left")
                  if doc["targets"]["kidneys"][s]["has_stone"]]
    lat = "NONE" if not sides_with else ("BILATERAL" if len(sides_with) == 2 else sides_with[0])
    for k, v in (("laterality", lat),
                 ("stones declared", f"{lab['stone_count_declared']}   "
                                     f"(individually described: {lab['stone_count_listed']})"),
                 ("count qualifier", lab["count_qualifier"])):
        gt.append(f"{k:<18}{v}")
    if lab.get("anomalies"):
        gt.append(f"{'other findings':<18}" + ", ".join(
            a.replace("_", " ").lower() for a in lab["anomalies"]))
    gt += ["", f"facts_hash {doc['reports']['facts_hash']}",
           f"template round-trip  {'PASSED' if doc['reports']['roundtrip_parse_ok'] else 'FAILED'}"]

    axg = fig.add_subplot(gs[1, 2:4])
    axg.axis("off")
    axg.set_title("5.  Ground truth — structured facts", fontsize=9)
    axg.text(0.0, 1.0, "\n".join(gt), transform=axg.transAxes, va="top", ha="left",
             fontsize=7.1, family="monospace", color=V.INK, linespacing=1.5)

    from common.templates import choose_variants, render_en
    fs_pred = predicted_factset(case_id, per_side)
    rep_pred = render_en(fs_pred, choose_variants(case_id, doc["reports"]["generator_seed"]))
    # same column layout as panel 5, so the two tables compare directly
    pr = ["PREDICTED STONES (model, held-out)", "─" * W,
          f"{'KIDNEY':<8}{'P(stone)':<10}{'CLASS':<13}{'ZONE':<7}{'ẑ':<5}"]
    for side in ("LEFT", "RIGHT"):
        p = per_side[side]
        pr.append(f"{side:<8}{p['p']:<10.3f}"
                  + (f"{SIZE_EN.get(p['cls'], '—'):<13}{p['zone']:<7}{p['zhat']:<5.2f}"
                     if p["present"] else f"{'—':<13}{'—':<7}{'—':<5}"))
    pr += ["", "REPORT GENERATED FROM THE MODEL'S OWN PREDICTIONS", "─" * W]
    for para in rep_pred["full"].split("\n"):
        pr += wrap(para, W) if para.strip() else [""]
    pr += ["", "same template as the reference; the model predicts a size",
           "CLASS, not a millimetre, so the generator writes",
           "\"indeterminate size (CLASS)\" instead of inventing a number,",
           "and omits the total-count sentence it cannot support."]

    axp = fig.add_subplot(gs[1, 4:6])
    axp.axis("off")
    axp.set_title("6.  Model output  →  generated report", fontsize=9)
    axp.text(0.0, 1.0, "\n".join(pr), transform=axp.transAxes, va="top", ha="left",
             fontsize=6.9, family="monospace", color=V.INK, linespacing=1.42)

    # explicit placement rather than head(): the GridSpec top is 0.855 and the
    # titles must clear it
    fig.suptitle(title or "End-to-end pipeline for one patient: "
                 "CT volume → structured facts → report",
                 x=0.008, y=0.985, ha="left", fontsize=12, color=V.INK, weight="semibold")
    fig.text(0.008, 0.955, "the model has never seen this patient (held-out fold 0); every "
             "panel is produced by the pipeline, nothing is hand-drawn",
             ha="left", fontsize=8.5, color=V.INK2)
    fig.text(0.008, 0.930, "the unit of modelling is the KIDNEY — right and left are two "
             "independent problems, and the report is assembled from both",
             ha="left", fontsize=8, color=V.MUTED)
    fig.savefig(OUT / out_name, bbox_inches="tight")
    plt.close(fig)
    print(f"    ({out_name} vaka: {case_id})")
    return case_id


SIZE3_TO_CLS = {0: "KUCUK", 1: "ORTA", 2: "BUYUK"}


def predicted_size3(kidney_id: str):
    """The stored size3 prediction for this kidney, averaged over seeds. Every
    kidney is in a test fold exactly once per seed, so all of them are held out."""
    ps = []
    for f in PREDS.glob("e9_size_sst_size3_s*_f*.csv"):
        d = pd.read_csv(f)
        m = d[d["kidney_id"] == kidney_id]
        if len(m):
            ps.append(json.loads(m.iloc[0]["prob"]))
    return int(np.argmax(np.mean(ps, axis=0))) if ps else None


def predicted_factset(case_id, per_side):
    """The canonical fact set built from the model's predictions.

    The model does NOT predict millimetres, it predicts a size CLASS, so the
    template writes "indeterminate size (CLASS)" rather than inventing a
    measurement. No count is predicted either, so total_n is None and the
    template omits the sentence that would state a total.
    """
    kid = {}
    for key in ("right", "left"):
        p = per_side[key.upper()]
        stones = ([{"zone": p["zone"], "mm": None, "cls": p["cls"]}]
                  if p["present"] and p["zone"] and p["cls"] else [])
        kid[key] = {"present": p["present"], "n_characterized": len(stones),
                    "stones": stones, "max_mm": None,
                    "max_cls": stones[0]["cls"] if stones else None,
                    "zones": sorted({s["zone"] for s in stones})}
    pres = [s for s in ("RIGHT", "LEFT") if per_side[s]["present"]]
    lat = "NONE" if not pres else ("BILATERAL" if len(pres) == 2 else pres[0])
    return {"laterality": lat, "total_n": None, "total_qualifier": "EXACT",
            "n_characterized": sum(k["n_characterized"] for k in kid.values()),
            "kidneys": kid,
            # Without a diameter the "largest stone ... mm" sentence cannot be
            # formed. The field is left empty rather than fabricated, and the
            # template simply skips that sentence.
            "largest": None, "anomalies": []}


def wrap(text, width):
    out, line = [], ""
    for w in text.split():
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    out.append(line)
    return out


def main() -> int:
    V.apply_theme()
    OUT.mkdir(parents=True, exist_ok=True)
    df = D.load_table()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = eval_transform()
    model = load_detector(dev)

    for name, fn in (("fig1_atlas", lambda: fig1_atlas(df)),
                     ("fig2_coronal", lambda: fig2_coronal(df)),
                     ("fig3_gradcam", lambda: fig3_gradcam(df, model, dev, tfm)),
                     ("fig4_results", fig4_results),
                     ("fig5_zone_mechanism", fig5_zone_mechanism),
                     ("fig6_confusion", fig6_confusion),
                     ("fig7_pipeline", lambda: fig7_pipeline(df, model, dev, tfm))):
        print(f"  {name} ...", flush=True)
        fn()
    print(f"\nwrote -> {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
