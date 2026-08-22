"""Aşama 1 — beş DICOM kökünü tara, dosya başına başlık indeksini çıkar.

Birleştirme kuralı: aynı akquizisyon her kopyada aynı SOPInstanceUID'yi taşır,
tamamlayıcı seriler farklı UID taşır. Bu yüzden birden çok kaynak kökünün
doğru birleşimi SOPInstanceUID üzerinden tekilleştirmedir (aşama 3'te uygulanır;
burada tüm satırlar ham olarak saklanır).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pydicom

from common.paths import DICOM_ROOTS, INDEX, LOGS, PRIVATE, ensure_dirs

CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"


def _get(ds, name, default=None):
    v = getattr(ds, name, default)
    return default if v is None else v


def _seq(ds, name, n):
    """Çok değerli alanı sabit uzunlukta float listesine çevir."""
    v = getattr(ds, name, None)
    if v is None:
        return [None] * n
    try:
        out = [float(x) for x in v]
    except (TypeError, ValueError):
        return [None] * n
    return (out + [None] * n)[:n]


def scan_file(path: Path, root_id: str, collection: str, patient_folder: str) -> dict:
    rec: dict = {
        "file_path": str(path),
        "root_id": root_id,
        "collection": collection,
        "patient_folder": patient_folder,
        "file_size": path.stat().st_size,
        "read_error": None,
    }
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception as exc:  # noqa: BLE001
        rec["read_error"] = f"{type(exc).__name__}: {exc}"
        return rec

    try:
        rec["transfer_syntax"] = str(ds.file_meta.TransferSyntaxUID)
    except Exception:  # noqa: BLE001
        rec["transfer_syntax"] = None

    iop = _seq(ds, "ImageOrientationPatient", 6)
    ipp = _seq(ds, "ImagePositionPatient", 3)
    ps = _seq(ds, "PixelSpacing", 2)

    image_type = _get(ds, "ImageType", [])
    try:
        image_type = "\\".join(str(x) for x in image_type)
    except TypeError:
        image_type = str(image_type)

    rec.update(
        {
            "sop_class_uid": str(_get(ds, "SOPClassUID", "")),
            "sop_instance_uid": str(_get(ds, "SOPInstanceUID", "")),
            "series_instance_uid": str(_get(ds, "SeriesInstanceUID", "")),
            "study_instance_uid": str(_get(ds, "StudyInstanceUID", "")),
            "series_number": _get(ds, "SeriesNumber"),
            "instance_number": _get(ds, "InstanceNumber"),
            "series_description": str(_get(ds, "SeriesDescription", "")),
            "image_type": image_type,
            "modality": str(_get(ds, "Modality", "")),
            "iop_0": iop[0], "iop_1": iop[1], "iop_2": iop[2],
            "iop_3": iop[3], "iop_4": iop[4], "iop_5": iop[5],
            "ipp_x": ipp[0], "ipp_y": ipp[1], "ipp_z": ipp[2],
            "slice_thickness": _get(ds, "SliceThickness"),
            "spacing_between_slices": _get(ds, "SpacingBetweenSlices"),
            "pixel_spacing_r": ps[0], "pixel_spacing_c": ps[1],
            "rows": _get(ds, "Rows"),
            "columns": _get(ds, "Columns"),
            "rescale_intercept": _get(ds, "RescaleIntercept"),
            "rescale_slope": _get(ds, "RescaleSlope"),
            "bits_stored": _get(ds, "BitsStored"),
            "bits_allocated": _get(ds, "BitsAllocated"),
            "pixel_representation": _get(ds, "PixelRepresentation"),
            "photometric_interpretation": str(_get(ds, "PhotometricInterpretation", "")),
            "convolution_kernel": str(_get(ds, "ConvolutionKernel", "")),
            "kvp": _get(ds, "KVP"),
            "manufacturer": str(_get(ds, "Manufacturer", "")),
            "manufacturer_model": str(_get(ds, "ManufacturerModelName", "")),
            "patient_position": str(_get(ds, "PatientPosition", "")),
            "study_date": str(_get(ds, "StudyDate", "")),
            # PHI: yalnızca 00_private tarafında kullanılacak, yayınlanan ağaca girmez
            "dicom_patient_name": str(_get(ds, "PatientName", "")),
            "dicom_patient_sex": str(_get(ds, "PatientSex", "")),
            "dicom_birth_date": str(_get(ds, "PatientBirthDate", "")),
        }
    )
    return rec


def main() -> int:
    ensure_dirs()
    rows: list[dict] = []
    t0 = time.time()
    for root_id, root, collection in DICOM_ROOTS:
        if not root.is_dir():
            print(f"UYARI: kök yok, atlanıyor -> {root}")
            continue
        n_root = 0
        for patient_dir in sorted(root.iterdir()):
            if not patient_dir.is_dir():
                continue
            for path in patient_dir.rglob("*"):
                if not path.is_file() or path.name == "DICOMDIR":
                    continue
                rows.append(scan_file(path, root_id, collection, patient_dir.name))
                n_root += 1
        print(f"{root_id:10s} {n_root:7d} dosya  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    INDEX.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PRIVATE / "dicom_index.parquet", index=False)

    n_err = int(df["read_error"].notna().sum())
    ct = df[df["sop_class_uid"] == CT_IMAGE_STORAGE] if "sop_class_uid" in df else df.iloc[0:0]
    print(f"\ntoplam dosya          : {len(df)}")
    print(f"okuma hatası          : {n_err}")
    print(f"CT görüntüsü          : {len(ct)}")
    print(f"CT dışı DICOM nesnesi : {len(df) - len(ct) - n_err}")
    print(f"benzersiz SOPInstanceUID (CT) : {ct['sop_instance_uid'].nunique()}")
    print(f"benzersiz SeriesInstanceUID   : {ct['series_instance_uid'].nunique()}")
    print(f"hasta klasörü                 : {df['patient_folder'].nunique()}")

    if n_err:
        err = df[df["read_error"].notna()][["file_path", "root_id", "read_error", "file_size"]]
        err.to_csv(LOGS / "01_read_errors.csv", index=False, encoding="utf-8")
        print("\nokunamayan dosyalar -> logs/01_read_errors.csv")
        print(err.to_string(index=False)[:2000])

    print(f"\nsüre: {time.time() - t0:.0f}s -> {INDEX / 'dicom_index.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
