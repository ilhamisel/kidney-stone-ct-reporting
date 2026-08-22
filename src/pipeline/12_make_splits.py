"""Aşama 12 — hasta düzeyinde stratifiye 5-kat CV, 5 tohum.

Bölme birimi DAİMA hastadır (vaka değil, kesit değil):
  * aynı kişinin iki tetkiki (KS0140_A/_B, KS0178_A/_B) aynı katmana düşer,
  * bir hastanın iki böbreği aynı katmana düşer — half-crop tasarımında en
    kritik sızıntı önlemi budur.

5 tohum, çünkü önceki çalışmada aynı konfigürasyonun 6 koşusu RIGHT doğruluğunu
0.52–0.76 arasında yaydı; bu genişlik karşılaştırılan tüm mimarilerin farkından
büyüktü. Tek koşu yorumlanamaz.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common.paths import LABELS, SPLITS, ensure_dirs

SEEDS = [1337, 2027, 7, 42, 12345]
K = 5
SIZE_3BUCKET = {
    "MIKROLITIYAZIS": "S", "KUCUK": "S",
    "ORTA": "M",
    "BUYUK": "L", "COK_BUYUK": "L",
}


def main() -> int:
    ensure_dirs()
    m = pd.read_csv(LABELS / "labels_master.csv")

    # hasta düzeyi tablo: bir hastanın birden çok vakası varsa en büyük taşı esas al
    pat = (
        m.assign(size3=m["max_size_class"].map(SIZE_3BUCKET))
        .sort_values("max_size_mm", ascending=False)
        .groupby("patient_id")
        .agg(laterality=("laterality", "first"), size3=("size3", "first"),
             n_cases=("case_id", "size"))
        .reset_index()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    pat["stratum"] = pat["laterality"] + "|" + pat["size3"]
    counts = Counter(pat["stratum"])
    print(f"hasta   : {len(pat)}   vaka: {len(m)}")
    print(f"tabaka  : {dict(sorted(counts.items()))}")

    # k'dan küçük tabakayı, aynı lateralitedeki komşu boyut kovasına birleştir.
    # (Tüm şemayı lateraliteye düşürmek boyut dengesini gereksiz yere feda ederdi.)
    NEIGHBOUR = {"S": "M", "M": "L", "L": "M"}
    merged: dict[str, str] = {}
    strat = dict(counts)
    for s, c in sorted(counts.items(), key=lambda kv: kv[1]):
        if c >= K or s in merged:
            continue
        lat, size = s.split("|")
        tgt = f"{lat}|{NEIGHBOUR[size]}"
        if tgt not in strat:
            tgt = lat  # aynı lateralitede komşu yoksa lateraliteye düş
        merged[s] = tgt
        strat[tgt] = strat.get(tgt, 0) + c
        strat.pop(s, None)
        print(f"UYARI: '{s}' tabakası n={c} < k={K} -> '{tgt}' ile birleştirildi")
    pat["stratum_used"] = pat["stratum"].map(lambda s: merged.get(s, s))
    still_thin = [s for s, c in Counter(pat["stratum_used"]).items() if c < K]
    if still_thin:
        print(f"UYARI: {still_thin} hâlâ küçük -> yalnız lateralite ile stratifiye edilir")
        pat["stratum_used"] = pat["laterality"]
    thin = still_thin

    all_splits = {}
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
        fold_of = {}
        folds: list[list[str]] = []
        for fi, (_, test_idx) in enumerate(skf.split(pat, pat["stratum_used"])):
            ids = sorted(pat.iloc[test_idx]["patient_id"])
            folds.append(ids)
            for p in ids:
                fold_of[p] = fi
        payload = {
            "k": K, "seed": seed, "unit": "patient",
            "stratify_by": "laterality|size3" if not thin else "laterality",
            "n_patients": len(pat),
            "folds": folds,
            "fold_of_patient": fold_of,
        }
        out = SPLITS / f"cv5_patient_seed{seed}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        all_splits[seed] = fold_of
        sizes = [len(f) for f in folds]
        lat = [Counter(pat[pat.patient_id.isin(f)]["laterality"]) for f in folds]
        print(f"  seed {seed:<6} katman boyutları={sizes}  "
              f"bilateral/katman={[c['BILATERAL'] for c in lat]}")

    # vaka düzeyine yay + rapor JSON'larına geri yaz
    m["_"] = 0
    for seed in SEEDS:
        m[f"cv5_seed{seed}"] = m["patient_id"].map(all_splits[seed])
    m = m.drop(columns=["_"])
    m.to_csv(LABELS / "labels_master.csv", index=False, encoding="utf-8")

    kd = pd.read_csv(LABELS / "labels_kidney.csv")
    for seed in SEEDS:
        kd[f"cv5_seed{seed}"] = kd["patient_id"].map(all_splits[seed])
    kd.to_csv(LABELS / "labels_kidney.csv", index=False, encoding="utf-8")

    rep_dir = LABELS / "reports"
    for _, r in m.iterrows():
        p = rep_dir / f"{r['case_id']}.json"
        doc = json.loads(p.read_text(encoding="utf-8"))
        doc["splits"] = {f"cv5_seed{s}": int(all_splits[s][r["patient_id"]]) for s in SEEDS}
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

    # sızıntı kontrolü: aynı hastanın tüm vakaları ve böbrekleri aynı katmanda mı
    leaks = 0
    for seed in SEEDS:
        g = m.groupby("patient_id")[f"cv5_seed{seed}"].nunique()
        leaks += int((g > 1).sum())
        gk = kd.groupby("patient_id")[f"cv5_seed{seed}"].nunique()
        leaks += int((gk > 1).sum())
    print(f"\nsızıntı kontrolü: aynı hastanın vaka/böbrekleri farklı katmanda -> {leaks} (0 olmalı)")

    manifest_path = LABELS / "dataset_manifest.json"
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    man["splits"] = [
        {"file": f"05_splits/cv5_patient_seed{s}.json", "k": K, "seed": s, "unit": "patient",
         "sha256": hashlib.sha256((SPLITS / f"cv5_patient_seed{s}.json").read_bytes()).hexdigest()[:16]}
        for s in SEEDS
    ]
    man["split_protocol"] = {
        "unit": "patient",
        "n_seeds": len(SEEDS), "k": K, "total_runs": len(SEEDS) * K,
        "rule": "Bir hastanın tüm vakaları ve her iki böbreği aynı katmandadır. "
                "Eğitim kodu katmanları diskten okur, asla yeniden hesaplamaz.",
        "inner_validation": "Her eğitim katmanı içinde sabit 80/20 stratifiye iç bölme "
                            "(tohum = outer_seed*100 + fold); test katmanı tam olarak bir kez skorlanır.",
        "forbidden_augmentation": "Tüm gövde görüntülerinde RandomHorizontalFlip yasak "
                                  "(böbrekleri takas eder, etiketi etmez). Böbrek başına half-crop'ta "
                                  "yatay aynalama serbest ve önerilir.",
    }
    manifest_path.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")

    gate = leaks == 0
    print(f"KAPI: sıfır sızıntı -> {'PASS' if gate else 'FAIL'}")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
