"""Aşama 4 — 16-bit kayıpsız HU arşivi (02_archive) + NIfTI.

Kesitler ImagePositionPatient'ın seri normaline izdüşümüyle sıralanır
(SliceLocation DEĞİL: opsiyonel, işareti tutarsız ve GE reformat serilerinde tekrar ediyor).
PNG'lere metin chunk'ı yazılmaz; tüm geometri meta.json'da tutulur.
Hasta bazında idempotent: meta.json varsa ve builder_version eşleşiyorsa atlanır.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixels import apply_modality_lut

from common.hu import HU_OFFSET, hu_to_png16
from common.paths import ARCHIVE, HOLDOUT, INDEX, LOGS, PRIVATE, ensure_dirs

BUILDER_VERSION = "kidneyct2027-archive/1.0.0"
CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
NIFTI_ROLES = {"axial_std"}


def uid_hash(uid: str, salt: str) -> str:
    return hashlib.sha256((salt + uid).encode("utf-8")).hexdigest()[:16]


def order_slices(files: pd.DataFrame, normal: np.ndarray) -> pd.DataFrame:
    """SOPInstanceUID'e göre tekilleştir, normale izdüşümle sırala."""
    f = files.sort_values(["sop_instance_uid", "file_size"], ascending=[True, False])
    f = f.drop_duplicates("sop_instance_uid", keep="first").copy()
    ipp = f[["ipp_x", "ipp_y", "ipp_z"]].to_numpy(dtype=float)
    f["proj"] = ipp @ normal
    f = f.sort_values("proj", kind="stable").reset_index(drop=True)
    return f


def build_series(case_id: str, role: str, sel: pd.Series, files: pd.DataFrame,
                 out_root: Path, salt: str) -> dict:
    normal = np.array([sel["normal_x"], sel["normal_y"], sel["normal_z"]], dtype=float)
    f = order_slices(files, normal)

    out_dir = out_root / case_id / role
    out_dir.mkdir(parents=True, exist_ok=True)

    z_mm, sha_list, warnings = [], [], []
    hu_min_all, hu_max_all = 1 << 30, -(1 << 30)
    first_ipp = None
    n_written = 0
    want_nifti = role in NIFTI_ROLES
    # Hacmi bellekte tut: NIfTI'yi PNG'leri geri okumadan yazmak için
    # (geri okuma + gzip-9 vaka başına ~90 saniye tutuyordu).
    vol = np.zeros((512, 512, len(f)), dtype=np.int16) if want_nifti else None
    for i, row in enumerate(f.itertuples()):
        try:
            ds = pydicom.dcmread(row.file_path, force=True)
            hu = apply_modality_lut(ds.pixel_array, ds).astype(np.int32)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"slice {i}: okunamadı ({type(exc).__name__}: {exc})")
            continue
        if hu.shape != (512, 512):
            warnings.append(f"slice {i}: beklenmeyen boyut {hu.shape}")
            continue
        png = hu_to_png16(hu)
        path = out_dir / f"{n_written:04d}.png"
        cv2.imwrite(str(path), png, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if vol is not None:
            vol[:, :, n_written] = hu.astype(np.int16)
        sha_list.append(hashlib.sha256(path.read_bytes()).hexdigest()[:16])
        z_mm.append(float(row.proj))
        if first_ipp is None:
            first_ipp = [float(row.ipp_x), float(row.ipp_y), float(row.ipp_z)]
        hu_min_all = min(hu_min_all, int(hu.min()))
        hu_max_all = max(hu_max_all, int(hu.max()))
        n_written += 1

    dz = np.diff(np.array(z_mm)) if len(z_mm) > 1 else np.array([np.nan])
    median_dz = float(np.median(dz)) if np.isfinite(dz).any() else float("nan")
    irregular = bool(np.isfinite(dz).any() and float(np.std(dz)) > 0.01)
    has_gap = bool(np.isfinite(dz).any() and float(np.max(dz)) > 2 * median_dz)

    meta = {
        "case_id": case_id,
        "series_role": role,
        "series_uid_hash": uid_hash(sel["series_instance_uid"], salt),
        "study_uid_hash": uid_hash(sel["study_instance_uid"], salt),
        "plane": sel["plane"],
        "n_slices": n_written,
        "rows": 512, "cols": 512,
        "pixel_spacing_mm": [float(sel["pixel_spacing_r"]), float(sel["pixel_spacing_c"])],
        "slice_thickness_mm": float(sel["slice_thickness"]),
        "median_dz_mm": median_dz,
        "irregular_spacing": irregular,
        "has_gap": has_gap,
        "slice_normal": [float(x) for x in normal],
        "first_ipp": first_ipp,
        "z_mm": z_mm,
        "hu_offset": HU_OFFSET, "hu_scale": 1,
        "hu_min_observed": hu_min_all, "hu_max_observed": hu_max_all,
        "png_bitdepth": 16, "png_channels": 1,
        "rescale_intercept": float(sel["rescale_intercept"]),
        "rescale_slope": float(sel["rescale_slope"]),
        "transfer_syntaxes": sel["transfer_syntaxes"],
        "series_description": sel["series_description"],
        "source_roots": sel["roots"],
        "selection_reason": sel["selection_reason"],
        "sha256_per_slice": sha_list,
        "builder_version": BUILDER_VERSION,
        "warnings": warnings,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    if vol is not None and n_written > 1:
        write_nifti(vol[:, :, :n_written], out_root / case_id / f"volume_{role}.nii.gz", meta)
    return meta


def write_nifti(vol: np.ndarray, out_path: Path, meta: dict) -> None:
    """Geometri taşıyan ayna kopya (reformat / segmentasyon araçları için).

    gzip seviyesi 1: dosya ~%15 büyür ama yazma süresi vaka başına ~60s'den
    ~6s'ye düşer. NIfTI zaten türetilmiş bir artefakt; kanonik arşiv PNG'lerdir.
    """
    import gzip

    import nibabel as nib

    ps = meta["pixel_spacing_mm"]
    dz = meta["median_dz_mm"] if np.isfinite(meta["median_dz_mm"]) else meta["slice_thickness_mm"]
    nrm = np.array(meta["slice_normal"], dtype=float)
    # RAS olmayan ama tutarlı bir affine: satır/sütun aralığı + kesit normali
    affine = np.eye(4)
    affine[0, 0] = ps[1]
    affine[1, 1] = ps[0]
    affine[:3, 2] = nrm * dz
    if meta["first_ipp"]:
        affine[:3, 3] = meta["first_ipp"]
    img = nib.Nifti1Image(vol, affine)
    with gzip.GzipFile(str(out_path), "wb", compresslevel=1) as fh:
        fh.write(img.to_bytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="labeled", choices=["labeled", "holdout_unlabeled", "all"])
    ap.add_argument("--roles", default="axial_std,coronal_native",
                    help="virgülle ayrılmış roller (varsayılan: axial_std,coronal_native)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    salt = (PRIVATE / "anon_salt.txt").read_text(encoding="utf-8").strip()
    sel = pd.read_csv(PRIVATE / "series_selection.csv")
    idx = pd.read_parquet(PRIVATE / "dicom_index.parquet")
    idx = idx[idx["sop_class_uid"] == CT_IMAGE_STORAGE]

    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    sel = sel[sel["role"].isin(roles)]
    if args.group != "all":
        sel = sel[sel["group"] == args.group]
    if args.limit:
        keep = sorted(sel["case_id"].unique())[: args.limit]
        sel = sel[sel["case_id"].isin(keep)]

    series_meta = pd.read_csv(PRIVATE / "series_index_full.csv").set_index("series_instance_uid")
    out_root = ARCHIVE if args.group != "holdout_unlabeled" else HOLDOUT / "02_archive"

    cases = sorted(sel["case_id"].unique())
    print(f"{len(cases)} vaka, {len(sel)} seri -> {out_root}")
    t0 = time.time()
    log = []
    for n, case_id in enumerate(cases, 1):
        for _, s in sel[sel["case_id"] == case_id].iterrows():
            role = s["role"]
            meta_path = out_root / case_id / role / "meta.json"
            if meta_path.exists() and not args.force:
                try:
                    if json.loads(meta_path.read_text(encoding="utf-8")).get("builder_version") == BUILDER_VERSION:
                        continue
                except json.JSONDecodeError:
                    pass
            row = s.to_dict()
            row.update(series_meta.loc[s["series_instance_uid"]].to_dict())
            files = idx[idx["series_instance_uid"] == s["series_instance_uid"]]
            meta = build_series(case_id, role, pd.Series(row), files, out_root, salt)
            log.append({"case_id": case_id, "role": role, "n_slices": meta["n_slices"],
                        "hu_min": meta["hu_min_observed"], "hu_max": meta["hu_max_observed"],
                        "irregular": meta["irregular_spacing"], "gap": meta["has_gap"],
                        "n_warnings": len(meta["warnings"])})
        if n % 10 == 0 or n == len(cases):
            el = time.time() - t0
            print(f"  {n}/{len(cases)} vaka  {el:.0f}s  (kalan ~{el / n * (len(cases) - n):.0f}s)")

    if log:
        df = pd.DataFrame(log)
        out = LOGS / f"04_archive_{args.group}.csv"
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"\ntoplam kesit yazıldı : {int(df['n_slices'].sum())}")
        print(f"HU aralığı           : [{int(df['hu_min'].min())}, {int(df['hu_max'].max())}]")
        print(f"düzensiz aralık      : {int(df['irregular'].sum())} seri")
        print(f"boşluklu             : {int(df['gap'].sum())} seri")
        print(f"uyarılı seri         : {int((df['n_warnings'] > 0).sum())}")
        print(f"log -> {out}")
    else:
        print("yeni iş yok (hepsi güncel)")
    print(f"süre: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
