"""Stage 8b — testing whether the zone signal was measured in the wrong frame.

The earlier pipeline recorded a negative result: the normalized vertical
position of the brightest focus in a cropped window did not separate the zones
(LOWER 0.610±0.082, MID 0.595±0.067, UPPER 0.602±0.059, Cohen d = 0.109).
But that measurement was taken in a frame relative to the BODY.

This stage repeats the same measurement in two frames side by side:
  A) body-relative     normalized to the z range of the volume (the earlier frame)
  B) kidney-relative   normalized to that kidney's OWN craniocaudal extent

The stone position is measured FROM THE IMAGE, independently of the reported
zone: the centroid of the largest high-attenuation component inside the kidney
mask. The test is therefore not circular.

If the earlier result was a frame error, the zones separate in B. If they do
not, then the negative result is a correctly measured one and should be reported
as such — which is the point of running both frames rather than only the new one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import kruskal

from common.paths import ARCHIVE, LABELS, LOGS, ROOT

SEG_ROOT = ROOT / "08_seg"
HU_STONE = 300
DILATE_ITER = 2


def load_volume(case_dir: Path, meta: dict) -> np.ndarray:
    n = meta["n_slices"]
    vol = np.zeros((512, 512, n), dtype=np.int16)
    for i in range(n):
        png = cv2.imread(str(case_dir / f"{i:04d}.png"), cv2.IMREAD_UNCHANGED)
        vol[:, :, i] = (png.astype(np.int32) - meta["hu_offset"]).astype(np.int16)
    return vol


def largest_focus(vol: np.ndarray, mask: np.ndarray) -> tuple[float, int, int] | None:
    """The largest high-attenuation component inside the mask, returned as
    (slice index, voxel count, maximum HU)."""
    m = ndimage.binary_dilation(mask, np.ones((3, 3, 3)), iterations=DILATE_ITER)
    hi = (vol >= HU_STONE) & m
    if hi.sum() == 0:
        return None
    lab, n = ndimage.label(hi)
    if n == 0:
        return None
    sizes = ndimage.sum(hi, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    idx = np.argwhere(lab == k)
    return float(idx[:, 2].mean()), int(sizes[k - 1]), int(vol[lab == k].max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rep_dir = LABELS / "reports"
    cases = sorted(p.stem for p in rep_dir.glob("*.json")
                   if (SEG_ROOT / p.stem / "kidney_left.nii.gz").exists())
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} vakada segmentasyon mevcut")

    rows = []
    for n, case_id in enumerate(cases, 1):
        doc = json.loads((rep_dir / f"{case_id}.json").read_text(encoding="utf-8"))
        meta = json.loads((ARCHIVE / case_id / "axial_std" / "meta.json").read_text(encoding="utf-8"))
        z = np.array(meta["z_mm"])
        vol = None
        for side_key, side in (("right", "RIGHT"), ("left", "LEFT")):
            k = doc["targets"]["kidneys"][side_key]
            stones = [s for s in doc["labels"]["stones"] if s["side"] == side]
            # only single-stone, single-zone kidneys, so that the correspondence
            # between the label and the measurement is unambiguous
            zones = {s["zone"] for s in stones}
            if len(stones) != 1 or len(zones) != 1:
                continue
            zone = stones[0]["zone"]
            if zone not in ("UPPER", "MID", "LOWER"):
                continue
            crop = k.get("crop") or {}
            if crop.get("source") != "totalsegmentator" or "z_min_mm" not in crop:
                continue
            p = SEG_ROOT / case_id / f"kidney_{side_key}.nii.gz"
            if not p.exists():
                continue
            mask = np.asarray(nib.load(str(p)).dataobj) > 0
            if mask.sum() == 0:
                continue
            if vol is None:
                vol = load_volume(ARCHIVE / case_id / "axial_std", meta)
            f = largest_focus(vol, mask)
            if f is None:
                continue
            sl, nvox, maxhu = f
            z_focus = float(np.interp(sl, np.arange(len(z)), z))

            # A) body-relative
            z_body = (z_focus - z.min()) / (z.max() - z.min())
            # B) kidney-relative
            kz0, kz1 = crop["z_min_mm"], crop["z_max_mm"]
            z_kidney = (z_focus - kz0) / (kz1 - kz0) if kz1 > kz0 else np.nan

            rows.append({"case_id": case_id, "side": side, "zone": zone,
                         "z_focus_mm": round(z_focus, 2), "z_body": round(z_body, 4),
                         "z_kidney": round(float(z_kidney), 4),
                         "n_voxels": nvox, "max_hu": maxhu,
                         "size_mm": stones[0]["size_mm"]})
        if n % 40 == 0:
            print(f"  {n}/{len(cases)}")

    df = pd.DataFrame(rows)
    df.to_csv(LOGS / "08b_zone_frame.csv", index=False, encoding="utf-8")
    if df.empty:
        print("no measurable kidney found")
        return 1

    print(f"\nkidneys measured: {len(df)}  (single stone, single zone)")
    print(f"zone distribution: {df['zone'].value_counts().to_dict()}")

    def cohen_d(a, b):
        a, b = np.asarray(a), np.asarray(b)
        s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                    / (len(a) + len(b) - 2))
        return float((a.mean() - b.mean()) / s) if s > 0 else 0.0

    for frame, col in (("A) body-relative (the earlier frame)", "z_body"),
                       ("B) kidney-relative (the new frame)", "z_kidney")):
        print(f"\n=== {frame} ===")
        g = {zz: df[df.zone == zz][col].dropna() for zz in ("LOWER", "MID", "UPPER")}
        for zz in ("LOWER", "MID", "UPPER"):
            if len(g[zz]):
                print(f"  {zz:6s} n={len(g[zz]):3d}  ortalama={g[zz].mean():.3f} ± {g[zz].std():.3f}")
        valid = [v for v in g.values() if len(v) >= 3]
        if len(valid) >= 2:
            st = kruskal(*valid)
            print(f"  Kruskal-Wallis : H={st.statistic:.2f}  p={st.pvalue:.3e}")
        if len(g["UPPER"]) >= 3 and len(g["LOWER"]) >= 3:
            d = cohen_d(g["UPPER"], g["LOWER"])
            print(f"  Cohen d (upper vs lower) = {d:+.3f}   "
                  f"(the earlier work found +0.109 in the body frame)")
    # --- Is the measured focus REALLY the stone? Test it against reported size.
    # If the component were bone or vascular calcification it would be unrelated
    # to the reported diameter. This is what establishes that the zone test above
    # measured the intended object rather than something incidental.
    from scipy.stats import pearsonr, spearmanr

    v = df.dropna(subset=["size_mm", "n_voxels"])
    if len(v) >= 8:
        ps_r, ps_c = 0.0, 0.0
        # sphere-equivalent diameter from the voxel count; voxel volume varies
        # between cases, but the median geometry is a good enough approximation
        vox_mm3 = 0.977 * 0.977 * 2.5
        est = 2 * ((3 * v["n_voxels"] * vox_mm3 / (4 * np.pi)) ** (1 / 3))
        rs, pvs = spearmanr(v["size_mm"], v["n_voxels"])
        rp, pvp = pearsonr(v["size_mm"], est)
        print("\n=== measurement validity: focus size against reported stone size ===")
        print(f"  Spearman (mm vs voksel)      : r={rs:.3f}  p={pvs:.2e}  n={len(v)}")
        print(f"  Pearson  (mm vs sphere-equivalent): r={rp:.3f}  p={pvp:.2e}")
        print(f"  median reported {v['size_mm'].median():.1f} mm against measured "
              f"{est.median():.1f} mm (the density threshold captures only the dense core)")
        band = v.groupby(pd.cut(v["size_mm"], [0, 5, 10, 20, 60]), observed=True)[
            ["n_voxels", "max_hu"]].median()
        print(band.to_string())

    print("\nNote: the stone position was measured from the image independently of "
          "the reported zone; the test is not circular.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
