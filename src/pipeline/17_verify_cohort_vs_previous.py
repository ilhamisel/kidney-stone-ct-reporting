"""Aşama 17 — yeni kohortun önceki 195 hastalık kohortla aynı olduğunu doğrula.

Önceki çalışmanın eşleme tablosu (195 satır) ile bu sürümün kohortunu Türkçe-normalize
anahtar üzerinden karşılaştırır ve eski AnonPt_XXX <-> yeni KS#### köprüsünü yazar.

Girdiler hasta adı taşıdığı için depo dışındadır ve çıktısı 00_private altına yazılır;
bkz. docs/data-availability.md. Dizin yoksa aşama temiz biçimde çıkar.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.paths import LABELS, PRIVATE
from common.trnorm import name_key

_PREV_DIR = Path(os.environ.get("KIDNEYCT_PREVIOUS", PRIVATE / "previous_cohort")).expanduser()
PREV = _PREV_DIR / "anon_mapping.csv"
PREV_LABELS = _PREV_DIR / "labels_master.csv"


def main() -> int:
    if not PREV.exists():
        print(f"HATA: önceki eşleme bulunamadı -> {PREV}")
        return 1
    prev = pd.read_csv(PREV)
    prev["name_key"] = prev["real_name"].map(name_key)
    phi = pd.read_csv(PRIVATE / "phi_map.csv")
    new = phi[phi["group"] == "labeled"].copy()

    pk, nk = set(prev["name_key"]), set(new["name_key"])
    print(f"önceki kohort : {len(prev)} satır, {len(pk)} benzersiz anahtar")
    print(f"yeni kohort   : {len(new)} hasta, {len(nk)} benzersiz anahtar")
    print(f"\nkesişim       : {len(pk & nk)}")
    print(f"yalnız önceki : {len(pk - nk)}  {sorted(prev[prev.name_key.isin(pk - nk)]['real_name'])}")
    print(f"yalnız yeni   : {len(nk - pk)}  {sorted(new[new.name_key.isin(nk - pk)]['name_raw_secondary'].dropna())}")

    # köprü tablosu
    bridge = (
        prev[["real_name", "patient_id", "split", "name_key"]]
        .rename(columns={"patient_id": "prev_patient_id", "split": "prev_split"})
        .merge(new[["patient_id", "name_key", "name_raw_secondary"]]
               .rename(columns={"patient_id": "new_patient_id"}),
               on="name_key", how="outer", indicator=True)
    )
    bridge.to_csv(PRIVATE / "cohort_bridge_prev_vs_new.csv", index=False, encoding="utf-8")
    print(f"\nköprü tablosu -> {PRIVATE / 'cohort_bridge_prev_vs_new.csv'}")
    print(bridge["_merge"].value_counts().to_string())

    # etiket karşılaştırması (lateralite) — önceki denetlenmiş etiketlerle
    if PREV_LABELS.exists():
        pl = pd.read_csv(PREV_LABELS)
        m = bridge[bridge["_merge"] == "both"].merge(
            pl[["patient_id", "laterality", "size_class_max", "Stone_Count_Mentioned"]],
            left_on="prev_patient_id", right_on="patient_id", how="left")
        cur = pd.read_csv(LABELS / "labels_master.csv")
        # bir hastanın birden çok vakası varsa en büyük taşlı vakayı al
        cur = cur.sort_values("max_size_mm", ascending=False).drop_duplicates("patient_id")
        m = m.merge(cur[["patient_id", "laterality", "max_size_class", "n_stones_effective"]]
                    .rename(columns={"patient_id": "new_patient_id",
                                     "laterality": "new_laterality",
                                     "max_size_class": "new_max_class"}),
                    on="new_patient_id", how="left")
        m["lat_agree"] = m["laterality"].str.upper() == m["new_laterality"]
        n = int(m["lat_agree"].notna().sum())
        agree = int(m["lat_agree"].sum())
        print(f"\n=== lateralite karşılaştırması (önceki denetlenmiş etiketler) ===")
        print(f"  karşılaştırılabilir: {n}   uyuşan: {agree} ({agree / max(n, 1):.3f})")
        dis = m[~m["lat_agree"].fillna(True)]
        if len(dis):
            print(f"  uyuşmayan {len(dis)}:")
            print(dis[["prev_patient_id", "new_patient_id", "laterality", "new_laterality"]]
                  .head(20).to_string(index=False))
        m.to_csv(PRIVATE / "cohort_label_comparison.csv", index=False, encoding="utf-8")

    same = pk == nk
    print(f"\nKAPI: kohortlar birebir aynı -> {'PASS' if same else 'FAIL'}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
