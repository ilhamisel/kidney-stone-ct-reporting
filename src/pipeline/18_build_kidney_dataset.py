"""Aşama 18 — böbrek düzeyi veri manifesti (modelleme aşamasının girdisi).

Modelleme birimi BÖBREKTİR, vaka değil. Half-crop bulgusu lateraliteyi "hangi
taraf?" tek problemi yerine böbrek başına iki bağımsız problem olarak
modellemeyi gerektiriyor; bu dosya o birimi tek satırda toplar:

  hedefler        has_stone, max_size_class, zone_* (çok etiketli), dominant_zone
  girdi yolları   halfcrop_{side}_{stone,mw3}
  geometri        böbrek z aralığı ve BÖBREĞE GÖRELİ zon bantları
  bölmeler        5 tohumun katman indeksleri (hasta düzeyinde, sızıntısız)

Aynalama notu: sağ ve sol half-crop'lar yatay aynalamayla tek bir "bir böbrek"
problemine indirgenebilir — bu etiket koruyucudur ve etkin örneklemi ikiye katlar.
Tüm gövde görüntülerinde yatay çevirme ise YASAKTIR (böbrekleri takas eder).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.paths import DERIVED, LABELS, ROOT

SEG_ROOT = ROOT / "08_seg"


def main() -> int:
    rep_dir = LABELS / "reports"
    kd = pd.read_csv(LABELS / "labels_kidney.csv")
    fold_cols = [c for c in kd.columns if c.startswith("cv5_seed")]

    rows = []
    for _, r in kd.iterrows():
        case_id, side = r["case_id"], r["side"]
        side_key = side.lower()
        doc = json.loads((rep_dir / f"{case_id}.json").read_text(encoding="utf-8"))
        study_date = str((doc.get("cohort") or {}).get("study_date") or "")
        crop = doc["targets"]["kidneys"][side_key].get("crop") or {}
        has_mask = crop.get("source") == "totalsegmentator" and not crop.get("mask_empty")

        stone_dir = DERIVED / case_id / f"halfcrop_{side_key}_stone"
        mw3_dir = DERIVED / case_id / f"halfcrop_{side_key}_mw3"
        n_png = len(list(stone_dir.glob("*.png"))) if stone_dir.is_dir() else 0

        bands = crop.get("zone_bands_mm") or {}
        rows.append({
            "kidney_id": f"{case_id}_{side_key[0].upper()}",
            "case_id": case_id, "patient_id": r["patient_id"], "side": side,
            "study_date": study_date,
            # --- hedefler
            "has_stone": r["has_stone"],
            "n_characterized": r["n_characterized"],
            "max_mm": r["max_mm"], "max_class": r["max_class"],
            "zone_upper": r["zone_upper"], "zone_mid": r["zone_mid"],
            "zone_lower": r["zone_lower"], "dominant_zone": r["dominant_zone"],
            # --- girdi
            "halfcrop_stone_dir": str(stone_dir.relative_to(ROOT)) if n_png else None,
            "halfcrop_mw3_dir": str(mw3_dir.relative_to(ROOT)) if mw3_dir.is_dir() else None,
            "n_crop_slices": n_png,
            "crop_h": (crop.get("crop_size") or [None, None])[0],
            "crop_w": (crop.get("crop_size") or [None, None])[1],
            # --- geometri
            "has_mask": has_mask,
            "mask_volume_mm3": crop.get("volume_mm3"),
            "kidney_z_min_mm": crop.get("z_min_mm"),
            "kidney_z_max_mm": crop.get("z_max_mm"),
            "kidney_z_extent_mm": crop.get("z_extent_mm"),
            "zone_band_lower": json.dumps(bands.get("LOWER")) if bands else None,
            "zone_band_mid": json.dumps(bands.get("MID")) if bands else None,
            "zone_band_upper": json.dumps(bands.get("UPPER")) if bands else None,
            "zone_frame": crop.get("zone_frame"),
            "orientation_check_passed": crop.get("orientation_check_passed"),
            **{c: r[c] for c in fold_cols},
        })

    df = pd.DataFrame(rows)
    # Maske akla yatkınlık denetimi. Erişkin böbrek kraniyokaudal uzunluğu
    # ~90-130 mm, hacmi ~100-250 cm3. Bu aralıkların dışı segmentasyon hatası
    # ya da atrofik/kısmi FOV böbrek demektir; sessizce kullanılmamalı.
    # nullable boolean: maskesi olmayan böbrek için "bilinmiyor" (NA) gerekiyor
    df["mask_plausible"] = (
        df["kidney_z_extent_mm"].between(70, 150)
        & df["mask_volume_mm3"].between(50_000, 400_000)
    ).astype("boolean")
    df.loc[~df["has_mask"].astype(bool), "mask_plausible"] = pd.NA

    # --- birincil tetkik: hasta başına TEK tetkik -------------------------------
    # 2 hastanın 2 tetkiki var (KS0140, KS0178). Bunlar hasta düzeyi bölmede zaten
    # aynı katta; yani train/test SIZINTISI yok. Ancak aynı hasta test katında iki
    # kez sayıldığında birimler bağımsız olmaz (sözde-yineleme) ve güven aralıkları
    # olduğundan dar çıkar. Bu yüzden hasta başına tek tetkik tutulur.
    #
    # Ölçüt: EN ERKEN tetkik tarihi (indeks tetkik), eşitlikte case_id.
    # Tarih etiketten bağımsızdır; "daha çok taşı olanı seç" gibi bir ölçüt
    # kohortu sonuca göre seçmek olurdu. (Bu veri setinde iki ölçüt aynı
    # tetkikleri veriyor: KS0140_B ve KS0178_A.)
    order = df.sort_values(["patient_id", "study_date", "case_id"])
    primary = order.groupby("patient_id")["case_id"].first()
    df["primary_exam"] = [c == primary[p] for c, p in zip(df["case_id"], df["patient_id"])]
    out = LABELS / "kidney_dataset.csv"
    df.to_csv(out, index=False, encoding="utf-8")

    usable = df[df["n_crop_slices"] > 0]
    dropped = df[~df["primary_exam"]]
    print(f"böbrek kaydı        : {len(df)}  (hasta {df['patient_id'].nunique()})")
    print(f"birincil tetkik     : {int(df['primary_exam'].sum())} böbrek / "
          f"{df.loc[df['primary_exam'], 'case_id'].nunique()} tetkik  "
          f"(düşen: {sorted(dropped['case_id'].unique())})")
    print(f"maskesi olan        : {int(df['has_mask'].sum())}")
    print(f"half-crop üretilmiş : {len(usable)}")
    print(f"taşlı böbrek        : {int(df['has_stone'].sum())} / {len(df)}")
    print(f"yönelim denetimi    : {int((df['orientation_check_passed'] == True).sum())} geçti, "  # noqa: E712
          f"{int((df['orientation_check_passed'] == False).sum())} kaldı")  # noqa: E712
    if len(usable):
        print(f"crop kesit sayısı   : medyan {usable['n_crop_slices'].median():.0f} "
              f"(min {usable['n_crop_slices'].min()}, maks {usable['n_crop_slices'].max()})")
        print(f"crop boyutu (piksel): medyan {usable['crop_h'].median():.0f} x "
              f"{usable['crop_w'].median():.0f}")
        print(f"böbrek uzunluğu mm  : medyan {usable['kidney_z_extent_mm'].median():.0f}")
    imp = df[(df["has_mask"] == True) & (df["mask_plausible"] == False)]  # noqa: E712
    print(f"akla yatkın maske   : {int((df['mask_plausible'] == True).sum())}; "  # noqa: E712
          f"şüpheli {len(imp)} -> {imp['kidney_id'].tolist()}")
    neg = int((~df["has_stone"].astype(bool)).sum())
    print(f"\nBÖBREK DÜZEYİNDE NEGATİF SINIF: {neg}/{len(df)} böbrekte raporlanmış taş yok.")
    print("  (Hasta düzeyinde negatif yoktur — 197/197 vakada taş var. Böbrek düzeyinde")
    print("   ise gerçek bir negatif sınıf mevcut; ancak 'raporlanmamış' kesin 'yok'")
    print("   demek değildir, radyolog taşları sayarak tanımlamıştır.)")
    print("\nhedef dağılımları:")
    print(f"  max_class     : {df[df.has_stone]['max_class'].value_counts().to_dict()}")
    print(f"  dominant_zone : {df[df.has_stone]['dominant_zone'].value_counts().to_dict()}")
    print(f"  çok zonlu böbrek: {int((df[['zone_upper', 'zone_mid', 'zone_lower']].sum(axis=1) > 1).sum())}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
