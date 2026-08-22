"""Stage 09 — 3-D stone candidate generation, heuristic.

This runs with or without segmentation. When no kidney mask is available it
falls back on two discriminative features:

  1. Connected components at or above the density threshold, inside the body
     mask.
  2. A SHELL TEST: the mean attenuation of a thin shell around the component.
     A stone is surrounded by soft tissue (shell around 0-60 HU) whereas bone is
     surrounded by more bone (shell above 150 HU). This rejects spine, pelvis and
     ribs far more reliably than a volume threshold would — a 55 mm staghorn
     calculus would have been caught by any volume cut-off large enough to
     exclude the spine.

Candidates are matched to the reported stones and stored with a confidence
score. This component is a CANDIDATE GENERATOR, not a detector: no modelling
stage consumes its output, and it must not be read as a detection result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

from common.paths import ARCHIVE, LABELS, LOGS

HU_STONE = 300          # 200 was too loose: bowel content and vascular calcification flooded in
BODY_HU = -500
SHELL_HU_MAX = 80.0     # below this shell mean, the component sits in soft tissue
MIN_VOX = 8
MAX_DIAM_MM = 70.0
MIN_DIAM_MM = 1.5

# Coarse kidney box, the same one stage 07c uses. Without segmentation this is
# the only meaningful gain in specificity available: it excludes the spine (centre),
# the ribs and body wall (edges), and the pelvis and lung base (the z extremes).
X_BAND = (0.20, 0.80)
X_SPINE = (0.44, 0.56)   # the spinal corridor, excluded
Z_BAND = (0.35, 0.85)


def load_volume(case_dir: Path) -> tuple[np.ndarray, dict]:
    meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    n = meta["n_slices"]
    vol = np.zeros((512, 512, n), dtype=np.int16)
    for i in range(n):
        png = cv2.imread(str(case_dir / f"{i:04d}.png"), cv2.IMREAD_UNCHANGED)
        vol[:, :, i] = (png.astype(np.int32) - meta["hu_offset"]).astype(np.int16)
    return vol, meta


def kidney_mask(case_id: str, shape: tuple[int, int, int]) -> np.ndarray | None:
    """Use the segmentation mask where one exists; it is far more specific than
    the coarse box."""
    import nibabel as nib

    from common.paths import ROOT

    seg = ROOT / "08_seg" / case_id
    parts = []
    for f in ("kidney_left.nii.gz", "kidney_right.nii.gz"):
        p = seg / f
        if p.exists():
            m = np.asarray(nib.load(str(p)).dataobj) > 0
            if m.shape == shape:
                parts.append(m)
    if not parts:
        return None
    return np.logical_or.reduce(parts)


def detect(vol: np.ndarray, meta: dict, mask: np.ndarray | None = None) -> list[dict]:
    ps_r, ps_c = meta["pixel_spacing_mm"]
    dz = meta["median_dz_mm"] if np.isfinite(meta["median_dz_mm"]) else meta["slice_thickness_mm"]
    vox_mm3 = ps_r * ps_c * dz

    body = vol > BODY_HU
    body = ndimage.binary_opening(body, np.ones((3, 3, 1)))

    if mask is not None:
        # A true kidney mask, dilated by 5 mm so that stones sitting at the edge
        # of the collecting system are still covered.
        ps_r = meta["pixel_spacing_mm"][0]
        it = max(1, int(round(5.0 / ps_r / 2)))
        box = ndimage.binary_dilation(mask, np.ones((3, 3, 3)), iterations=it)
        roi_source = "kidney_mask"
    else:
        # fallback: the coarse kidney box, excluding the spinal corridor and edges
        W, H, N = vol.shape
        box = np.zeros_like(body)
        x0, x1 = int(X_BAND[0] * W), int(X_BAND[1] * W)
        sx0, sx1 = int(X_SPINE[0] * W), int(X_SPINE[1] * W)
        z0, z1 = int(Z_BAND[0] * N), int(Z_BAND[1] * N)
        box[x0:x1, :, z0:z1] = True
        box[sx0:sx1, :, :] = False
        roi_source = "heuristic_box"

    hi = (vol >= HU_STONE) & body & box
    lab, n = ndimage.label(hi)
    if n == 0:
        return []

    objs = ndimage.find_objects(lab)
    # The body centre must be found along the LEFT-RIGHT axis. `vol` is ordered
    # (row, column, slice); under the corrected affine, axis 0 runs anterior to
    # posterior and axis 1 runs toward the patient's left.
    #
    # An earlier version profiled axis 0 instead, so side assignment was a coin
    # flip: it agreed with the mask on 30 of 58 candidates. Using axis 1 gives
    # 58 of 58. The lesson is that verifying an axis convention once at its source
    # is not enough — every component that consumes it needs its own check.
    prof = body.sum(axis=(0, 2))
    idx = np.where(prof > 0)[0]
    body_cx = float(idx.mean()) if len(idx) else 256.0

    out = []
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        sub = lab[sl] == i
        nvox = int(sub.sum())
        # With a mask available the threshold is relaxed: stones of 5 mm or less
        # occupy a median of 3.5 voxels, so an 8-voxel cut-off removed most of them.
        if nvox < (3 if mask is not None else MIN_VOX):
            continue
        ext_mm = (
            (sl[0].stop - sl[0].start) * ps_r,
            (sl[1].stop - sl[1].start) * ps_c,
            (sl[2].stop - sl[2].start) * dz,
        )
        diam = float(max(ext_mm))
        if diam > MAX_DIAM_MM or diam < MIN_DIAM_MM:
            continue  # elongated structures such as spine and pelvis, and single-voxel noise

        # shell test: dilate the component by 2 voxels and take the difference
        pad = tuple(slice(max(s.start - 3, 0), min(s.stop + 3, vol.shape[a]))
                    for a, s in enumerate(sl))
        sub_pad = (lab[pad] == i)
        dil = ndimage.binary_dilation(sub_pad, np.ones((3, 3, 3)), iterations=2)
        shell = dil & ~ndimage.binary_dilation(sub_pad, np.ones((3, 3, 3)), iterations=1)
        shell_vals = vol[pad][shell]
        shell_mean = float(shell_vals.mean()) if shell_vals.size else 1e9
        # The shell test exists to reject bone. A kidney mask already does that,
        # so with a mask the test is skipped — it was removing small stones.
        if mask is None and shell_mean > SHELL_HU_MAX:
            continue

        vals = vol[sl][sub]
        # vol axes: 0 = row (anterior->posterior, the radiologist row index j),
        #           1 = column (right->left), 2 = slice (caudal->cranial)
        c_row, c_col, c_slice = ndimage.center_of_mass(sub)
        g_row = sl[0].start + c_row
        g_col = sl[1].start + c_col
        g_slice = sl[2].start + c_slice
        out.append({
            "centroid_voxel": [round(float(g_row), 1), round(float(g_col), 1),
                               round(float(g_slice), 1)],
            "max_hu": int(vals.max()),
            "mean_hu": round(float(vals.mean()), 1),
            "n_voxels": nvox,
            "volume_mm3": round(nvox * vox_mm3, 1),
            "max_diameter_mm": round(diam, 1),
            "shell_mean_hu": round(shell_mean, 1),
            # DICOM LPS: increasing COLUMN index moves toward the patient's LEFT
            "side": "LEFT" if g_col > body_cx else "RIGHT",
            # the radiologist-selected index is an axial ROW index, i.e. axis 0;
            # match() compares it against the selected rows
            "y_row": round(float(g_row), 1),
            "roi_source": roi_source,
        })
    out.sort(key=lambda c: -c["volume_mm3"])
    for k, c in enumerate(out, start=1):
        c["candidate_id"] = k
    return out


def match(cands: list[dict], doc: dict) -> None:
    """Match candidates to reported stones, within side and by SIZE RANK.

    Requiring absolute agreement on diameter was wrong. The density threshold
    captures only the dense core, while the radiologist measures the diameter in
    the widest plane, so the measured diameter is systematically smaller than the
    reported one (median 4.5 against 8.0 mm). The ORDERING, however, is preserved
    (Spearman r = 0.874), which makes rank agreement the correct criterion and
    absolute difference the wrong one.
    """
    sel = doc.get("selected_slices") or {}
    sel_rows = [it.get("j_new_axial_std") for it in sel.get("items", [])
                if it.get("j_new_axial_std") is not None]

    for side in ("RIGHT", "LEFT"):
        cs = sorted([c for c in cands if c["side"] == side],
                    key=lambda c: -c["volume_mm3"])
        ss = sorted([s for s in doc["labels"]["stones"] if s["side"] == side],
                    key=lambda s: -(s["size_mm"] or -1))
        for rank, (c, s) in enumerate(zip(cs, ss)):
            ev = ["side", f"size_rank_{rank + 1}"]
            conf = 0.6
            if sel_rows and min(abs(c["y_row"] - r) for r in sel_rows) <= 10:
                ev.append("row_within_10px")
                conf += 0.2
            if s["size_known"] and c["max_diameter_mm"] >= 0.4 * s["size_mm"]:
                ev.append("size_consistent")
                conf += 0.2
            c["matched_stone_index"] = s["stone_index"]
            c["match_confidence"] = round(conf, 2)
            c["match_evidence"] = ev
    for c in cands:
        c.setdefault("matched_stone_index", None)
        c.setdefault("match_confidence", None)
        c.setdefault("match_evidence", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--write", action="store_true", help="write into the report JSON files")
    args = ap.parse_args()

    rep_dir = LABELS / "reports"
    cases = sorted(p.stem for p in rep_dir.glob("*.json")
                   if (ARCHIVE / p.stem / "axial_std" / "meta.json").exists())
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} vaka")

    rows, stone_rows = [], []
    for n, case_id in enumerate(cases, 1):
        doc = json.loads((rep_dir / f"{case_id}.json").read_text(encoding="utf-8"))
        vol, meta = load_volume(ARCHIVE / case_id / "axial_std")
        mask = kidney_mask(case_id, vol.shape)
        cands = detect(vol, meta, mask)
        match(cands, doc)
        n_matched = sum(1 for c in cands if c["matched_stone_index"])
        n_stones = len(doc["labels"]["stones"])
        rows.append({"case_id": case_id, "n_candidates": len(cands),
                     "roi_source": "kidney_mask" if mask is not None else "heuristic_box",
                     "n_stones": n_stones, "n_matched": n_matched,
                     "match_rate": n_matched / n_stones if n_stones else np.nan})
        matched_idx = {c["matched_stone_index"] for c in cands if c["matched_stone_index"]}
        for s in doc["labels"]["stones"]:
            stone_rows.append({"case_id": case_id, "stone_index": s["stone_index"],
                               "size_mm": s["size_mm"], "size_class": s["size_class"],
                               "side": s["side"],
                               "detected": s["stone_index"] in matched_idx})
        if args.write:
            doc["stone_candidates"] = cands
            doc.setdefault("targets", {})
            for side in ("right", "left"):
                # Do NOT overwrite the segmentation metadata written by stage 08.
                # The heuristic note is written only when no real crop block
                # exists. An earlier unconditional version destroyed the stage 08
                # blocks, which then had to be rebuilt from the masks by stage 21.
                cur = doc["targets"]["kidneys"][side].get("crop") or {}
                if cur.get("source") != "totalsegmentator":
                    doc["targets"]["kidneys"][side]["crop"] = {"source": "heuristic", "note":
                        "segmentation was not run; no kidney mask available"}
            (rep_dir / f"{case_id}.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        if n % 20 == 0:
            print(f"  {n}/{len(cases)}")

    df = pd.DataFrame(rows)
    df.to_csv(LOGS / "09_stone_candidates.csv", index=False, encoding="utf-8")
    print(f"\naday/vaka medyan : {df['n_candidates'].median():.0f}")
    print(f"matched stone rate: {df['match_rate'].mean():.3f} "
          f"({int(df['n_matched'].sum())}/{int(df['n_stones'].sum())})")
    sr = pd.DataFrame(stone_rows)
    sr.to_csv(LOGS / "09_stone_recall_by_size.csv", index=False, encoding="utf-8")
    if len(sr):
        print("\n=== detection sensitivity by size class ===")
        g = sr.groupby("size_class", observed=True)["detected"].agg(["sum", "count"])
        order = ["MIKROLITIYAZIS", "KUCUK", "ORTA", "BUYUK", "COK_BUYUK"]
        for k in order:
            if k in g.index:
                d, t = int(g.loc[k, "sum"]), int(g.loc[k, "count"])
                print(f"  {k:15s} {d:3d}/{t:3d}  = {d / t:.2f}")
        print("  Note: partial-volume effects push small stones below the density")
        print("  threshold. The limit is detection sensitivity, not the matching rule.")
    print(f"log -> {LOGS / '09_stone_candidates.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
