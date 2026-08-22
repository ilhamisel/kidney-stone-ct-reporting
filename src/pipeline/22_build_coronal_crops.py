# -*- coding: utf-8 -*-
"""Aşama 22 — böbrek başına KORONAL kırpma yığınları (zon görevi için).

Gerekçe (E4/E5 bulgusu): aksiyel kurguda zon bilgisi kesit İNDEKSİNDEDİR ve
LSE-MIL bunu iki denemede de öğrenemedi (makro AUC 0.55/0.58 ≈ şans). Koronal
düzlemde böbreğin üst-orta-alt tamamı TEK görüntüde görünür; zon, görüntü-içi
dikey konuma dönüşür — CNN'in doğal işi. §3'ün oracle kanıtı (d=3.82) sinyalin
tam bu çerçevede olduğunu gösteriyor.

Üretim: 02_archive aksiyel HU yığınından, 08'in (21 ile onarılan) padli bbox'u
içinde her satır r için koronal düzlem [s0:s1, r, c0:c1] alınır, z ekseni
dz/ps oranında yeniden örneklenerek izotropik yapılır, baş yukarı çevrilir,
taş penceresiyle (E1 kazananı) 8-bit yazılır:
    03_derived/KS####/corcrop_{side}_stone/{r:04d}.png
Dikey eksen = böbreğin kraniyokaudal uzanımı (s0..s1 padsizdir), yani görüntünün
üst üçte biri ≈ ÜST zon. Dosya adı = aksiyel satır indeksi (ön→arka).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from common.hu import WINDOWS, window8
from common.paths import ARCHIVE, DERIVED, LABELS

KIND = "stone"


def build_case(case_id: str) -> int:
    doc = json.loads((LABELS / "reports" / f"{case_id}.json").read_text(encoding="utf-8"))
    meta = json.loads((ARCHIVE / case_id / "axial_std" / "meta.json").read_text(encoding="utf-8"))
    hu_off = meta["hu_offset"]
    ps_c = meta["pixel_spacing_mm"][1]
    dz = meta["median_dz_mm"] if np.isfinite(meta["median_dz_mm"]) else meta["slice_thickness_mm"]
    z = np.array(meta["z_mm"])

    crops = {}
    for side in ("left", "right"):
        c = doc["targets"]["kidneys"][side].get("crop") or {}
        if c.get("source") == "totalsegmentator" and not c.get("mask_empty"):
            crops[side] = c["bbox_voxel_padded"]
    if not crops:
        return 0

    s_lo = min(b[4] for b in crops.values())
    s_hi = max(b[5] for b in crops.values())
    src = ARCHIVE / case_id / "axial_std"
    vol = np.stack([
        cv2.imread(str(src / f"{i:04d}.png"), cv2.IMREAD_UNCHANGED).astype(np.int32) - hu_off
        for i in range(s_lo, s_hi)])                       # (S, 512, 512) HU

    n_png = 0
    for side, (r0, r1, c0, c1, s0, s1) in crops.items():
        out = DERIVED / case_id / f"corcrop_{side}_{KIND}"
        out.mkdir(parents=True, exist_ok=True)
        sub = vol[s0 - s_lo:s1 - s_lo, :, c0:c1]           # (S, 512, W)
        # baş yukarı: kesit indeksi z_mm'de artıyorsa dizinin sonu baş demektir
        if z[min(s1 - 1, len(z) - 1)] > z[s0]:
            sub = sub[::-1]
        h_iso = max(2, int(round(sub.shape[0] * dz / ps_c)))  # izotropik yükseklik
        for r in range(r0, r1):
            img = sub[:, r, :].astype(np.float32)          # (S, W) koronal düzlem
            img = cv2.resize(img, (img.shape[1], h_iso), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(out / f"{r:04d}.png"), window8(img, *WINDOWS[KIND]),
                        [cv2.IMWRITE_PNG_COMPRESSION, 6])
            n_png += 1
    return n_png


def main() -> int:
    cases = sorted(d.name for d in ARCHIVE.iterdir()
                   if d.is_dir() and (d / "axial_std").is_dir())
    total = 0
    for n, case_id in enumerate(cases, 1):
        total += build_case(case_id)
        if n % 20 == 0 or n == len(cases):
            print(f"  {n}/{len(cases)}  ({total} png)", flush=True)
    print(f"bitti: {total} koronal kırpma PNG'si")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
