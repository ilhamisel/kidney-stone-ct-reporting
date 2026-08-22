"""Aşama 5 — 16-bit arşivden 8-bit model-hazır türevler.

matplotlib YOK, savefig YOK, dolgu YOK, yeniden örnekleme YOK. Çıktı tam 512x512.
Önceki hattın hatası tam buradaydı: DICOM başlığındaki yumuşak doku penceresi
(WC=40/WW=400) kullanıldığı için 240 HU üstü doyuyordu ve taş yoğunluğu
piksel verisinden siliniyordu.

Üretilenler:
  axial_stone  8-bit gri, WL=400/WW=1500  -> BİRİNCİL (taş yoğunluğu monoton rampada)
  axial_mw3    RGB, R=soft G=stone B=wide -> ImageNet ön-eğitimli omurgalar için
Türevler kayıpsız arşivden her an yeniden üretilebilir.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

from common.hu import WINDOWS, multiwindow3, png16_to_hu, window8
from common.paths import ARCHIVE, DERIVED, LOGS, ensure_dirs

BUILDER_VERSION = "kidneyct2027-derived/1.0.0"


def build_case(case_id: str, role: str, kinds: list[str]) -> dict:
    src = ARCHIVE / case_id / role
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    n = meta["n_slices"]
    offset = meta["hu_offset"]

    dirs = {k: DERIVED / case_id / f"{role.replace('axial_std', 'axial')}_{k}" for k in kinds}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    sat_num = sat_den = 0
    for i in range(n):
        png = cv2.imread(str(src / f"{i:04d}.png"), cv2.IMREAD_UNCHANGED)
        hu = png.astype(np.int32) - offset
        if "stone" in kinds:
            img = window8(hu, *WINDOWS["stone"])
            cv2.imwrite(str(dirs["stone"] / f"{i:04d}.png"), img,
                        [cv2.IMWRITE_PNG_COMPRESSION, 6])
            # doygunluk: gövde içinde (HU > -500) 255'e dayanan piksel oranı
            body = hu > -500
            sat_num += int(((img == 255) & body).sum())
            sat_den += int(body.sum())
        if "mw3" in kinds:
            rgb = multiwindow3(hu)
            cv2.imwrite(str(dirs["mw3"] / f"{i:04d}.png"), rgb[:, :, ::-1],
                        [cv2.IMWRITE_PNG_COMPRESSION, 6])
        if "soft" in kinds:
            cv2.imwrite(str(dirs["soft"] / f"{i:04d}.png"), window8(hu, *WINDOWS["soft"]),
                        [cv2.IMWRITE_PNG_COMPRESSION, 6])

    info = {
        "case_id": case_id, "role": role, "n_slices": n, "kinds": kinds,
        "windows": {k: list(WINDOWS[k]) for k in WINDOWS},
        "mw3_channel_order": ["soft", "stone", "wide"],
        "saturation_fraction_stone_window": round(sat_num / sat_den, 6) if sat_den else None,
        "builder_version": BUILDER_VERSION,
    }
    (DERIVED / case_id / "derived_meta.json").write_text(
        json.dumps(info, ensure_ascii=False), encoding="utf-8")
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="stone,mw3")
    ap.add_argument("--role", default="axial_std")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure_dirs()

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    cases = sorted(d.name for d in ARCHIVE.iterdir()
                   if d.is_dir() and (d / args.role / "meta.json").exists())
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} vaka, türevler: {kinds}")

    rows, t0 = [], time.time()
    for n, case_id in enumerate(cases, 1):
        dm = DERIVED / case_id / "derived_meta.json"
        if dm.exists() and not args.force:
            try:
                old = json.loads(dm.read_text(encoding="utf-8"))
                if old.get("builder_version") == BUILDER_VERSION and set(old.get("kinds", [])) >= set(kinds):
                    rows.append(old)
                    continue
            except json.JSONDecodeError:
                pass
        rows.append(build_case(case_id, args.role, kinds))
        if n % 20 == 0 or n == len(cases):
            el = time.time() - t0
            print(f"  {n}/{len(cases)}  {el:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(LOGS / "05_derivatives.csv", index=False, encoding="utf-8")
    sat = df["saturation_fraction_stone_window"].dropna()
    print(f"\ntoplam kesit  : {int(df['n_slices'].sum())}")
    if len(sat):
        print(f"taş penceresi doygunluk oranı: medyan={sat.median():.5f} maks={sat.max():.5f}")
        bad = int((sat > 0.005).sum())
        print(f"  eşiği (>0.005) aşan vaka: {bad}/{len(sat)}")
        print(f"  KAPI: doygunluk < %0.5 -> {'PASS' if bad == 0 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
