"""Aşama 8a — NIfTI affine'ini düzelt (segmentasyon öncesi zorunlu).

04_build_archive_png.py ilk sürümünde affine şöyle kuruluyordu:
    affine[0,0] = pixel_spacing[1];  affine[1,1] = pixel_spacing[0]
Bu iki hata içeriyordu:
  1. Dizi ekseni 0 SATIR indeksidir ve IOP'nin SATIR yön kosinüsüne (IOP[3:6],
     tipik olarak +y) eşlenir; sütun yön kosinüsüne (IOP[0:3], +x) değil.
     Yani x ve y eksenleri yer değiştirmişti.
  2. DICOM hasta koordinatları LPS'tir, NIfTI ise RAS varsayar. Dönüşüm
     yapılmadığı için sağ/sol AYNALANMIŞ olurdu.

İkisi birlikte segmentasyonun sağ/sol böbreği ters atamasına yol açardı —
lateralite bu çalışmanın ana hedefi olduğu için kabul edilemez.

Bu script hacmi arşiv PNG'lerinden (kayıpsız) yeniden okuyup doğru affine ile
yazar ve meta.json'a image_orientation_patient alanını ekler.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import nibabel as nib
import numpy as np
import pandas as pd

from common.paths import ARCHIVE, PRIVATE

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
AFFINE_VERSION = "affine/2.0.0-ras"


def build_affine(iop: np.ndarray, ps: list[float], dz: float, ipp: list[float]) -> np.ndarray:
    """LPS -> RAS affine. Dizi ekseni sırası: (satır, sütun, kesit)."""
    col_dir = np.asarray(iop[0:3], dtype=float)   # sütun indeksi artışı
    row_dir = np.asarray(iop[3:6], dtype=float)   # satır indeksi artışı
    normal = np.cross(col_dir, row_dir)
    n = np.linalg.norm(normal)
    normal = normal / n if n > 0 else np.array([0.0, 0.0, 1.0])

    a = np.eye(4)
    a[:3, 0] = row_dir * ps[0]     # eksen 0 = satır, aralık = PixelSpacing[0]
    a[:3, 1] = col_dir * ps[1]     # eksen 1 = sütun, aralık = PixelSpacing[1]
    a[:3, 2] = normal * dz
    a[:3, 3] = np.asarray(ipp, dtype=float)
    return LPS_TO_RAS @ a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sel = pd.read_csv(PRIVATE / "series_selection.csv")
    sel = sel[sel["role"] == "axial_std"]
    idx = pd.read_parquet(PRIVATE / "dicom_index.parquet",
                          columns=["series_instance_uid", "iop_0", "iop_1", "iop_2",
                                   "iop_3", "iop_4", "iop_5", "pixel_spacing_r",
                                   "pixel_spacing_c"])
    iop_by_series = (idx.groupby("series_instance_uid")[["iop_0", "iop_1", "iop_2",
                                                          "iop_3", "iop_4", "iop_5"]]
                     .median())

    cases = sorted(d.name for d in ARCHIVE.iterdir()
                   if d.is_dir() and (d / "axial_std" / "meta.json").exists())
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} vaka")

    t0 = time.time()
    n_done = 0
    for n, case_id in enumerate(cases, 1):
        d = ARCHIVE / case_id / "axial_std"
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta.get("affine_version") == AFFINE_VERSION and not args.force:
            continue
        row = sel[sel["case_id"] == case_id]
        if row.empty:
            print(f"  UYARI: seri seçimi yok -> {case_id}")
            continue
        suid = row.iloc[0]["series_instance_uid"]
        if suid not in iop_by_series.index:
            print(f"  UYARI: IOP bulunamadı -> {case_id}")
            continue
        iop = iop_by_series.loc[suid].to_numpy(dtype=float)

        n_sl = meta["n_slices"]
        vol = np.zeros((512, 512, n_sl), dtype=np.int16)
        for i in range(n_sl):
            png = cv2.imread(str(d / f"{i:04d}.png"), cv2.IMREAD_UNCHANGED)
            vol[:, :, i] = (png.astype(np.int32) - meta["hu_offset"]).astype(np.int16)

        dz = meta["median_dz_mm"] if np.isfinite(meta["median_dz_mm"]) else meta["slice_thickness_mm"]
        affine = build_affine(iop, meta["pixel_spacing_mm"], dz, meta["first_ipp"])
        img = nib.Nifti1Image(vol, affine)
        img.header.set_xyzt_units("mm")
        out = ARCHIVE / case_id / "volume_axial_std.nii.gz"
        with gzip.GzipFile(str(out), "wb", compresslevel=1) as fh:
            fh.write(img.to_bytes())

        meta["image_orientation_patient"] = [float(x) for x in iop]
        meta["affine_version"] = AFFINE_VERSION
        meta["affine_ras"] = [[float(x) for x in r] for r in affine]
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        n_done += 1
        if n % 20 == 0 or n == len(cases):
            el = time.time() - t0
            print(f"  {n}/{len(cases)}  yenilenen={n_done}  {el:.0f}s")

    # doğrulama: bir vakada RAS yönelimi
    if cases:
        d = ARCHIVE / cases[0] / "axial_std"
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        img = nib.load(str(ARCHIVE / cases[0] / "volume_axial_std.nii.gz"))
        orient = nib.aff2axcodes(img.affine)
        print(f"\n{cases[0]} yönelim kodları: {orient}  (RAS bekleniyor: ('R','A','S') benzeri)")
        print(f"  zoom (mm): {tuple(round(float(z), 3) for z in img.header.get_zooms())}")
        print(f"  shape    : {img.shape}")
    print(f"\nyenilenen NIfTI: {n_done}   süre: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
