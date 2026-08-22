"""Aşama 15 — yayınlanabilir alt ağacın denetimi ve dışa aktarımı.

Varsayılan mod `audit`: hiçbir şey kopyalamaz, yalnızca yayınlanacak dosyaları
listeler ve üzerlerinde PHI taraması yapar (arşiv 15+ GB olduğu için gereksiz
kopya üretmemek adına). `link` sabit bağ (hardlink) ağacı kurar, `copy` gerçek
kopya çıkarır.

00_private/ ve 07_holdout_unlabeled/ hiçbir modda dahil edilmez.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.paths import ARCHIVE, DERIVED, INDEX, LABELS, PRIVATE, QA, ROOT, SPLITS

PUBLIC_DIRS = [INDEX, ARCHIVE, DERIVED, LABELS, SPLITS, QA]
TEXT_SUFFIXES = {".json", ".csv", ".md", ".txt"}


def phi_terms() -> tuple[set[str], set[str]]:
    phi = pd.read_csv(PRIVATE / "phi_map.csv")
    names = set()
    for col in ("name_raw_secondary", "name_raw_primary", "folder_names", "name_key"):
        if col not in phi.columns:
            continue
        for v in phi[col].dropna():
            for part in str(v).split("|"):
                p = part.strip().upper()
                if len(p) > 6:
                    names.add(p)
    uids = set()
    idx_path = PRIVATE / "dicom_index.parquet"
    if idx_path.exists():
        idx = pd.read_parquet(idx_path, columns=["series_instance_uid", "study_instance_uid",
                                                 "dicom_patient_name"])
        for col in ("series_instance_uid", "study_instance_uid"):
            for v in idx[col].dropna().unique():
                s = str(v).strip()
                # Boş dize her metinde eşleşir ve taramayı tamamen anlamsız kılar
                # (ilk koşuda 631 sahte isabetin sebebi buydu). DICOM UID'leri
                # zaten en az ~20 karakterdir.
                if len(s) >= 20:
                    uids.add(s)
        for v in idx["dicom_patient_name"].dropna().unique():
            s = str(v).upper().strip()
            if len(s) > 6:
                names.add(s)
                names.add(s.replace("^", " "))
    names = {n for n in names if len(n.strip()) > 6}
    return names, uids


def scan(files: list[Path], names: set[str], uids: set[str]) -> list[dict]:
    hits = []
    forbidden = ["PATIENTNAME", "PATIENTBIRTHDATE", "ACCESSIONNUMBER", "INSTITUTIONNAME"]
    for p in files:
        up_name = p.name.upper()
        for nm in names:
            if nm in up_name:
                hits.append({"file": str(p), "kind": "filename", "value": nm})
        if p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore").upper()
        except OSError:
            continue
        for nm in names:
            if nm in txt:
                hits.append({"file": str(p), "kind": "name", "value": nm})
                break
        for u in uids:
            if u in txt:
                hits.append({"file": str(p), "kind": "uid", "value": u})
                break
        for f in forbidden:
            if f in txt:
                hits.append({"file": str(p), "kind": "tag", "value": f})
                break
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["audit", "link", "copy"], default="audit")
    ap.add_argument("--out", default=str(ROOT.parent / "public_release"))
    args = ap.parse_args()

    files: list[Path] = []
    for d in PUBLIC_DIRS:
        if d.exists():
            files += [p for p in d.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    print(f"yayınlanabilir dosya : {len(files)}  ({total_bytes / 1e9:.1f} GB)")
    for d in PUBLIC_DIRS:
        n = sum(1 for p in d.rglob("*") if p.is_file()) if d.exists() else 0
        print(f"  {d.name:24s} {n:7d} dosya")

    names, uids = phi_terms()
    print(f"\nPHI terimleri: {len(names)} ad, {len(uids)} UID")
    hits = scan(files, names, uids)
    QA.mkdir(parents=True, exist_ok=True)
    # Bulgu raporu bulduğu PHI değerlerini taşır -> yayınlanan ağaca yazılamaz
    pd.DataFrame(hits).to_csv(PRIVATE / "qa_15_public_phi_findings.csv", index=False, encoding="utf-8")
    print(f"PHI isabeti: {len(hits)}")
    for h in hits[:10]:
        print(f"  {h['kind']:9s} {h['value'][:40]:42s} {h['file']}")

    # .dcm benzeri dosya olmamalı
    dcm = [p for p in files if p.suffix.lower() in (".dcm", ".dicom")]
    print(f"DICOM dosyası (olmamalı): {len(dcm)}")

    if hits or dcm:
        print("\nDIŞA AKTARIM DURDURULDU: önce PHI temizlenmeli.")
        return 1

    if args.mode == "audit":
        print("\nmod=audit: kopya üretilmedi. Gerçek dışa aktarım için --mode link|copy")
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in files:
        rel = p.relative_to(ROOT)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        if args.mode == "link":
            try:
                os.link(p, dst)
            except OSError:
                shutil.copy2(p, dst)
        else:
            shutil.copy2(p, dst)
        n += 1
    (out / "EXPORT_INFO.json").write_text(json.dumps(
        {"mode": args.mode, "n_files": len(files), "bytes": total_bytes,
         "excluded": ["00_private", "07_holdout_unlabeled"]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{n} dosya -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
