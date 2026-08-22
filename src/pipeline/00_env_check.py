"""Aşama 0 — ortam doğrulaması.

Kapı: veri setinde bulunan üç transfer sözdiziminin de çözülebildiğini ve
int16 HU verdiğini kanıtla. Geçemezse sonraki hiçbir aşama çalıştırılmaz.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut

from common.paths import DICOM_ROOTS, LOGS, ensure_dirs

# Kohortta CT görüntüsü (SOP 1.2.840.10008.5.1.4.1.1.2) taşıyan transfer sözdizimleri.
# Implicit VR (1.2.840.10008.1.2) bu veri setinde YALNIZCA Structured Report
# nesnelerinde geçiyor (Dose Record 88.67 ve ORU2SR 88.11; 220 dosya, 0 CT görüntüsü)
# -> piksel kod çözmesi gerektirmiyor, bu yüzden kapı listesinde değil.
WANTED = {
    "1.2.840.10008.1.2.1": "Explicit VR Little Endian",
    "1.2.840.10008.1.2.4.80": "JPEG-LS Lossless",
}

CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"

# Arşiv ofseti. GE Revolution EVO FOV dışı dolguda HU -3024'e kadar iniyor
# (00_env_check ile ölçüldü), bu yüzden plandaki 1024 yetersiz -> uint16 taşması.
# 4096 hem negatif ucu hem de ~61000 HU'ya kadar pozitif ucu güvenle kapsar.
HU_OFFSET = 4096


def find_samples() -> dict[str, Path]:
    """Her transfer sözdizimi için piksel verisi OLAN bir örnek dosya bul."""
    found: dict[str, Path] = {}
    for _root_id, root, _coll in DICOM_ROOTS:
        if not root.is_dir():
            continue
        for patient in sorted(root.iterdir()):
            if not patient.is_dir():
                continue
            for path in patient.rglob("*"):
                if not path.is_file() or path.name == "DICOMDIR":
                    continue
                try:
                    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                    ts = str(ds.file_meta.TransferSyntaxUID)
                    sop = str(getattr(ds, "SOPClassUID", ""))
                except Exception:
                    continue
                # Doz Raporu / Secondary Capture gibi piksel taşımayan nesneleri ele
                if sop != CT_IMAGE_STORAGE:
                    continue
                if ts in WANTED and ts not in found:
                    found[ts] = path
                if len(found) == len(WANTED):
                    return found
    return found


def main() -> int:
    ensure_dirs()
    print(f"python      : {sys.version.split()[0]}")
    print(f"pydicom     : {pydicom.__version__}")
    import cv2

    print(f"opencv      : {cv2.__version__}")
    print(f"numpy       : {np.__version__}")

    samples = find_samples()
    results = []
    ok = True
    for ts, name in WANTED.items():
        if ts not in samples:
            print(f"FAIL  {name:28s} {ts}  -> örnek dosya bulunamadı")
            ok = False
            results.append({"transfer_syntax": ts, "name": name, "status": "no_sample"})
            continue
        path = samples[ts]
        try:
            ds = pydicom.dcmread(path, force=True)
            raw = ds.pixel_array
            hu = apply_modality_lut(raw, ds).astype(np.int32)
            shape_ok = raw.shape == (ds.Rows, ds.Columns)
            # Arşiv kodlaması hu+HU_OFFSET'i uint16'ya sığdırabilmeli
            offset_ok = bool(hu.min() + HU_OFFSET >= 0 and hu.max() + HU_OFFSET <= 65535)
            passed = shape_ok and offset_ok
            status = "ok" if passed else "suspect"
            ok = ok and passed
            print(
                f"{'PASS' if passed else 'FAIL'}  {name:28s} raw={raw.dtype} shape={raw.shape} "
                f"HU[{hu.min()},{hu.max()}] offset_ok={offset_ok}"
            )
            results.append(
                {
                    "transfer_syntax": ts,
                    "name": name,
                    "status": status,
                    "raw_dtype": str(raw.dtype),
                    "shape": list(raw.shape),
                    "hu_min": int(hu.min()),
                    "hu_max": int(hu.max()),
                    "rescale_intercept": float(getattr(ds, "RescaleIntercept", 0)),
                    "rescale_slope": float(getattr(ds, "RescaleSlope", 1)),
                }
            )
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"FAIL  {name:28s} {type(exc).__name__}: {exc}")
            results.append({"transfer_syntax": ts, "name": name, "status": f"error: {exc}"})

    # 16-bit PNG gidiş-dönüş kontrolü (arşiv formatının temel varsayımı)
    import cv2

    rng = np.random.default_rng(0)
    probe = rng.integers(0, 65536, size=(512, 512), dtype=np.uint16)
    tmp = LOGS / "_png16_probe.png"
    cv2.imwrite(str(tmp), probe, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    back = cv2.imread(str(tmp), cv2.IMREAD_UNCHANGED)
    roundtrip = back is not None and back.dtype == np.uint16 and np.array_equal(back, probe)
    tmp.unlink(missing_ok=True)
    print(f"{'PASS' if roundtrip else 'FAIL'}  16-bit PNG gidiş-dönüş (cv2)")
    ok = ok and roundtrip
    results.append({"check": "png16_roundtrip", "status": "ok" if roundtrip else "fail"})

    (LOGS / "00_env_check.json").write_text(
        json.dumps({"passed": ok, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nSONUÇ:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
