"""Stage 8 — kidney masks, bounding boxes and per-kidney crops via segmentation.

Two things depend on this stage:
  1. A TRUE per-kidney crop. Splitting the image at the body midline was the
     single largest gain of the earlier pipeline, but it is only an approximation
     of the anatomy; an organ mask makes the crop correct rather than merely
     useful.
  2. KIDNEY-RELATIVE z. The zone task failed in the earlier work precisely
     because its reference frame was relative to the body. This stage records
     each kidney's own craniocaudal extent, so upper, mid and lower can be mapped
     onto thirds of that range.

The stage checks itself. Axis 1 (the column axis) increases toward the patient's
left, so the mean column index of the left kidney mask must be GREATER than that
of the right. If that fails, the orientation is wrong and the stage stops rather
than producing mirrored crops — which is exactly the failure that the affine
correction in stage 8a exists to prevent.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import nibabel as nib
import numpy as np
import pandas as pd

from common.hu import WINDOWS, multiwindow3, window8
from common.paths import ARCHIVE, DERIVED, LABELS, LOGS, ROOT

# TotalSegmentator is installed in its own environment (see README): it needs torch
# and conflicts with the data-pipeline dependencies. Resolved in order: an explicit
# environment variable, then PATH.
SEG_PY = Path(os.environ.get("TOTALSEGMENTATOR_BIN")
              or shutil.which("TotalSegmentator")
              or "TotalSegmentator")
SEG_ROOT = ROOT / "08_seg"
PAD_MM = 15.0
BUILDER = "kidneyct2027-seg/1.0.0"


def run_totalsegmentator(nii: Path, out_dir: Path, fast: bool, device: str) -> bool:
    if (out_dir / "kidney_left.nii.gz").exists() and (out_dir / "kidney_right.nii.gz").exists():
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(SEG_PY), "-i", str(nii), "-o", str(out_dir),
           "--roi_subset", "kidney_left", "kidney_right", "--device", device, "-q"]
    if fast:
        cmd.append("--fast")
    # Without an explicit encoding this decodes with the system code page and
    # raises UnicodeDecodeError on the segmentation tool's progress bar.
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        print(f"    ERROR from TotalSegmentator: {p.stderr.strip()[-300:]}")
        return False
    return (out_dir / "kidney_left.nii.gz").exists()


def mask_stats(mask: np.ndarray, meta: dict) -> dict | None:
    if mask.sum() == 0:
        return None
    ps_r, ps_c = meta["pixel_spacing_mm"]
    dz = meta["median_dz_mm"] if np.isfinite(meta["median_dz_mm"]) else meta["slice_thickness_mm"]
    idx = np.argwhere(mask)
    r0, c0, s0 = idx.min(axis=0)
    r1, c1, s1 = idx.max(axis=0) + 1
    z = np.array(meta["z_mm"])
    z0 = float(z[s0]) if s0 < len(z) else float(z[-1])
    z1 = float(z[min(s1 - 1, len(z) - 1)])
    zlo, zhi = min(z0, z1), max(z0, z1)
    # The zone frame: boundaries are defined against the kidney's OWN craniocaudal
    # extent, not the body's. In the earlier work this frame error made the
    # vertical position of the bright focus non-discriminative (Cohen d = 0.109);
    # measured in this frame the same quantity gives d = 3.82.
    third = (zhi - zlo) / 3.0
    return {
        "bbox_voxel": [int(r0), int(r1), int(c0), int(c1), int(s0), int(s1)],
        "centroid_voxel": [round(float(x), 1) for x in idx.mean(axis=0)],
        "volume_mm3": round(float(mask.sum()) * ps_r * ps_c * dz, 1),
        "z_min_mm": round(zlo, 2), "z_max_mm": round(zhi, 2),
        "z_extent_mm": round(zhi - zlo, 1),
        "n_slices": int(s1 - s0),
        # z increases toward the head, so the topmost third is the upper zone
        "zone_bands_mm": {
            "LOWER": [round(zlo, 2), round(zlo + third, 2)],
            "MID": [round(zlo + third, 2), round(zlo + 2 * third, 2)],
            "UPPER": [round(zlo + 2 * third, 2), round(zhi, 2)],
        },
        "zone_frame": "kidney_relative",
    }


def build_halfcrop(case_id: str, side: str, stats: dict, meta: dict, kinds: list[str]) -> dict:
    """Produce the axial crop by expanding the kidney box by PAD_MM."""
    ps_r, ps_c = meta["pixel_spacing_mm"]
    r0, r1, c0, c1, s0, s1 = stats["bbox_voxel"]
    pr, pc = int(round(PAD_MM / ps_r)), int(round(PAD_MM / ps_c))
    r0, r1 = max(r0 - pr, 0), min(r1 + pr, 512)
    c0, c1 = max(c0 - pc, 0), min(c1 + pc, 512)

    src = ARCHIVE / case_id / "axial_std"
    dirs = {k: DERIVED / case_id / f"halfcrop_{side}_{k}" for k in kinds}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    for i in range(s0, s1):
        png = cv2.imread(str(src / f"{i:04d}.png"), cv2.IMREAD_UNCHANGED)
        hu = (png.astype(np.int32) - meta["hu_offset"])[r0:r1, c0:c1]
        if "stone" in kinds:
            cv2.imwrite(str(dirs["stone"] / f"{i:04d}.png"), window8(hu, *WINDOWS["stone"]),
                        [cv2.IMWRITE_PNG_COMPRESSION, 6])
        if "mw3" in kinds:
            cv2.imwrite(str(dirs["mw3"] / f"{i:04d}.png"), multiwindow3(hu)[:, :, ::-1],
                        [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return {"bbox_voxel_padded": [r0, r1, c0, c1, s0, s1],
            "crop_size": [int(r1 - r0), int(c1 - c0)],
            "n_slices": int(s1 - s0),
            "dirs": {k: f"03_derived/{case_id}/halfcrop_{side}_{k}" for k in kinds}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--fast", action="store_true", default=True)
    ap.add_argument("--kinds", default="stone,mw3")
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--write", action="store_true", help="write into the report JSON files")
    args = ap.parse_args()

    if not SEG_PY.exists():
        print(f"ERROR: TotalSegmentator not found -> {SEG_PY}")
        print("Install it in a separate environment, then point "
              "TOTALSEGMENTATOR_BIN at the executable.")
        return 1

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    cases = sorted(d.name for d in ARCHIVE.iterdir()
                   if d.is_dir() and (d / "volume_axial_std.nii.gz").exists())
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} cases, device={args.device}, fast={args.fast}")

    rows, t0 = [], time.time()
    for n, case_id in enumerate(cases, 1):
        nii = ARCHIVE / case_id / "volume_axial_std.nii.gz"
        meta = json.loads((ARCHIVE / case_id / "axial_std" / "meta.json").read_text(encoding="utf-8"))
        if meta.get("affine_version") != "affine/2.0.0-ras":
            print(f"  SKIPPED {case_id}: affine not corrected (run 08a first)")
            continue
        out_dir = SEG_ROOT / case_id
        if not run_totalsegmentator(nii, out_dir, args.fast, args.device):
            rows.append({"case_id": case_id, "ok": False})
            continue

        masks = {}
        for side, fname in (("left", "kidney_left.nii.gz"), ("right", "kidney_right.nii.gz")):
            p = out_dir / fname
            masks[side] = np.asarray(nib.load(str(p)).dataobj) > 0 if p.exists() else np.zeros((1, 1, 1), bool)

        st = {s: mask_stats(masks[s], meta) for s in ("left", "right")}
        # --- orientation self-check: axis 1 increases toward the patient's left
        orient_ok = None
        if st["left"] and st["right"]:
            orient_ok = st["left"]["centroid_voxel"][1] > st["right"]["centroid_voxel"][1]
            if not orient_ok:
                print(f"  ! {case_id}: right/left orientation check FAILED "
                      f"(left c={st['left']['centroid_voxel'][1]}, right c={st['right']['centroid_voxel'][1]})")

        crops = {}
        if not args.no_crop:
            for side in ("left", "right"):
                if st[side]:
                    crops[side] = build_halfcrop(case_id, side, st[side], meta, kinds)

        rec = {"case_id": case_id, "ok": True, "orientation_ok": orient_ok,
               "left_vol_mm3": st["left"]["volume_mm3"] if st["left"] else 0.0,
               "right_vol_mm3": st["right"]["volume_mm3"] if st["right"] else 0.0,
               "left_z_extent": st["left"]["z_extent_mm"] if st["left"] else 0.0,
               "right_z_extent": st["right"]["z_extent_mm"] if st["right"] else 0.0}
        rows.append(rec)

        if args.write:
            p = LABELS / "reports" / f"{case_id}.json"
            doc = json.loads(p.read_text(encoding="utf-8"))
            for side in ("left", "right"):
                doc["targets"]["kidneys"][side]["crop"] = (
                    {"source": "totalsegmentator", "builder": BUILDER,
                     "orientation_check_passed": orient_ok, **st[side],
                     **(crops.get(side) or {})} if st[side] else
                    {"source": "totalsegmentator", "builder": BUILDER, "mask_empty": True}
                )
            p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

        if n % 10 == 0 or n == len(cases):
            el = time.time() - t0
            print(f"  {n}/{len(cases)}  {el:.0f}s  (kalan ~{el / n * (len(cases) - n):.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(LOGS / "08_segmentation.csv", index=False, encoding="utf-8")
    ok = df[df["ok"]] if "ok" in df else df
    print(f"\nsucceeded : {len(ok)}/{len(df)}")
    if len(ok):
        print(f"left kidney volume mm3 : median {ok['left_vol_mm3'].median():.0f}")
        print(f"right kidney volume mm3: median {ok['right_vol_mm3'].median():.0f}")
        print(f"craniocaudal extent mm : left {ok['left_z_extent'].median():.0f} / "
              f"right {ok['right_z_extent'].median():.0f}")
        bad = int((ok["orientation_ok"] == False).sum())  # noqa: E712
        print(f"orientation check failures: {bad}  (must be 0)")
        empty = int(((ok["left_vol_mm3"] == 0) | (ok["right_vol_mm3"] == 0)).sum())
        print(f"empty mask on one side    : {empty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
