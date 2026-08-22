"""Aşama 7c — radyolog satırlarının GÖRÜNTÜ ile tutarlılığını sayısal sınama.

7b indeksin *anlamını* kanıtladı (satır = y koordinatı). Bu script bir adım öteye
geçip etiket ↔ görüntü bağını sınar:

  Radyolog taş gördüğü satırı seçtiyse, o satırdaki koronal düzlemde izole
  yüksek-HU odak sayısı, aynı hastanın rastgele satırlarından anlamlı biçimde
  yüksek olmalıdır.

"İzole odak" = >=250 HU, 3–400 piksel alanlı bağlantılı bileşen. Omurga/pelvis/
kaburga gibi kemik yapılar bu alan aralığının çok üstünde kaldığı için elenir.
Eşleştirilmiş Wilcoxon testi ile raporlanır.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

from common.paths import ARCHIVE, LABELS, LOGS

HI_HU = 250
MIN_AREA, MAX_AREA = 3, 400
N_RANDOM = 25


def load_volume(case_dir: Path) -> tuple[np.ndarray, dict]:
    meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    n = meta["n_slices"]
    vol = np.zeros((512, 512, n), dtype=np.int16)
    for i in range(n):
        png = cv2.imread(str(case_dir / f"{i:04d}.png"), cv2.IMREAD_UNCHANGED)
        vol[:, :, i] = (png.astype(np.int32) - meta["hu_offset"]).astype(np.int16)
    return vol, meta


# Kaba böbrek kutusu. Segmentasyon olmadan yapılabilecek en özgül kısıtlama:
# x'te omurga (merkez) ve kaburga/gövde duvarı (kenar) dışlanır, z'de pelvis ve
# akciğer tabanı dışlanır. Aşama 08 (TotalSegmentator) bunu gerçek maskeyle değiştirir.
X_RIGHT = (0.20, 0.45)   # DICOM LPS: sütun indeksi arttıkça hastanın SOLUNA gidilir
X_LEFT = (0.55, 0.80)
Z_BAND = (0.35, 0.85)


def focus_count(vol: np.ndarray, row: int, side: str | None = None) -> int:
    """Koronal düzlemde izole yüksek-HU odak sayısı.
    plane ekseni 0 = x (aksiyel sütun), eksen 1 = z (kesit indeksi)."""
    plane = vol[row, :, :]
    w, nz = plane.shape
    z0, z1 = int(Z_BAND[0] * nz), int(Z_BAND[1] * nz)
    if side == "RIGHT":
        x0, x1 = int(X_RIGHT[0] * w), int(X_RIGHT[1] * w)
    elif side == "LEFT":
        x0, x1 = int(X_LEFT[0] * w), int(X_LEFT[1] * w)
    else:
        x0, x1 = int(X_RIGHT[0] * w), int(X_LEFT[1] * w)
    roi = plane[x0:x1, z0:z1]
    mask = (roi >= HI_HU).astype(np.uint8)
    if mask.sum() == 0:
        return 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int(((areas >= MIN_AREA) & (areas <= MAX_AREA)).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    sel = json.loads((LABELS / "selected_slices.json").read_text(encoding="utf-8"))
    master = pd.read_csv(LABELS / "labels_master.csv").set_index("case_id")
    cases = [c for c in sorted(sel) if (ARCHIVE / c / "axial_std" / "meta.json").exists()]
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} vaka sınanıyor")

    rng = random.Random(args.seed)
    rows = []
    for n_done, case_id in enumerate(cases, 1):
        vol, meta = load_volume(ARCHIVE / case_id / "axial_std")
        items = [it for it in sel[case_id]["items"] if it.get("j_new_axial_std") is not None]
        if not items:
            continue
        sel_rows = [int(it["j_new_axial_std"]) for it in items]
        lo, hi = min(sel_rows) - 60, max(sel_rows) + 60
        pool = [r for r in range(max(lo, 0), min(hi, 512)) if r not in sel_rows]
        rand_rows = rng.sample(pool, min(N_RANDOM, len(pool)))

        # Yalnız raporda taş bildirilen tarafa bak; bilateralde iki taraf da serbest
        lat = master.loc[case_id, "laterality"] if case_id in master.index else None
        side = lat if lat in ("RIGHT", "LEFT") else None

        sel_counts = [focus_count(vol, r, side) for r in sel_rows]
        rnd_counts = [focus_count(vol, r, side) for r in rand_rows]
        rows.append({
            "case_id": case_id, "side": side or "BOTH",
            "n_selected": len(sel_rows),
            "sel_mean": float(np.mean(sel_counts)),
            "sel_max": int(np.max(sel_counts)),
            "rnd_mean": float(np.mean(rnd_counts)) if rnd_counts else np.nan,
            "rnd_p90": float(np.percentile(rnd_counts, 90)) if rnd_counts else np.nan,
            "sel_gt_rnd": bool(np.mean(sel_counts) > (np.mean(rnd_counts) if rnd_counts else 0)),
        })
        if n_done % 20 == 0:
            print(f"  {n_done}/{len(cases)}")

    df = pd.DataFrame(rows)
    df.to_csv(LOGS / "07c_selected_row_validation.csv", index=False, encoding="utf-8")

    from scipy.stats import wilcoxon
    d = df.dropna(subset=["rnd_mean"])
    stat = wilcoxon(d["sel_mean"], d["rnd_mean"], alternative="greater")
    win = int(d["sel_gt_rnd"].sum())
    print("\n=== seçilmiş satır vs rastgele satır (izole >=250 HU odak sayısı) ===")
    print(f"  vaka                      : {len(d)}")
    print(f"  seçilmiş satır ortalaması : {d['sel_mean'].mean():.2f}")
    print(f"  rastgele satır ortalaması : {d['rnd_mean'].mean():.2f}")
    print(f"  oran                      : {d['sel_mean'].mean() / max(d['rnd_mean'].mean(), 1e-9):.2f}x")
    print(f"  seçilmişin kazandığı vaka : {win}/{len(d)} ({win / len(d):.3f})")
    print(f"  Wilcoxon (eşleştirilmiş)  : p = {stat.pvalue:.3e}")

    ok = bool(stat.pvalue < 0.001 and d["sel_mean"].mean() > d["rnd_mean"].mean())
    print(f"\nSONUÇ: etiket-görüntü bağı {'DOĞRULANDI' if ok else 'doğrulanamadı'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
