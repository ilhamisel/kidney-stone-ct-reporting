"""Aşama 10 — TR + EN rapor üretimi ve gidiş-dönüş kapısı.

SERT KAPI: üretilen 197 Türkçe ve 197 İngilizce raporun HEPSİ ayrıştırılınca
birebir aynı FactSet'i vermeli. Vermiyorsa şablon belirsizdir; model eğitilmeden
önce düzeltilir. Bu kapı, "yapılandırılmış alan F1'i" değerlendirmesinin
geçerliliğinin ön koşuludur.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.factset import build_factset, canonical_stone_tuples, facts_hash
from common.paths import LABELS, LOGS, PRIVATE, ensure_dirs
from common.report_parser import diff_factsets, parse_report
from common.templates import choose_variants, render_en, render_tr

TEMPLATE_VERSION = "tmpl-1.0.0"
GEN_SEED = 1337


def normalize_for_compare(fs: dict) -> dict:
    """Karşılaştırma için sadeleştir: üretecin metne yansıtmadığı alanlar hariç.

    `present` bayrağı, taşı tarif edilmemiş tarafı da işaretler; metinde bunun
    karşılığı açık bir cümle olduğu için ayrıştırıcı da kurtarabilir.
    """
    out = json.loads(json.dumps(fs, sort_keys=True))
    for k in ("right", "left"):
        out["kidneys"][k].pop("max_cls", None) if False else None
    return out


def main() -> int:
    ensure_dirs()
    parsed = json.loads((PRIVATE / "labels_parsed.json").read_text(encoding="utf-8"))

    reports: dict[str, dict] = {}
    rows = []
    failures = []
    for case_id, rec in sorted(parsed.items()):
        fs = build_factset(rec)
        variants = choose_variants(case_id, GEN_SEED)
        tr = render_tr(fs, variants)
        en = render_en(fs, variants)

        fs_tr = parse_report(tr["full"], "tr")
        fs_en = parse_report(en["full"], "en")
        d_tr = diff_factsets(fs, fs_tr)
        d_en = diff_factsets(fs, fs_en)
        if d_tr or d_en:
            failures.append({"case_id": case_id, "tr": d_tr, "en": d_en,
                             "tr_text": tr["full"], "en_text": en["full"]})

        reports[case_id] = {
            "template_version": TEMPLATE_VERSION,
            "generator_seed": GEN_SEED,
            "facts_hash": facts_hash(fs),
            "variant_ids": variants,
            "factset": fs,
            "stone_tuples": canonical_stone_tuples(fs),
            "tr": tr,
            "en": en,
            "roundtrip_parse_ok": not (d_tr or d_en),
            "original_report_tr": rec["original_report_tr"],
        }
        rows.append({
            "case_id": case_id, "facts_hash": facts_hash(fs),
            "laterality": fs["laterality"], "n_char": fs["n_characterized"],
            "total_n": fs["total_n"], "qualifier": fs["total_qualifier"],
            "roundtrip_tr": not d_tr, "roundtrip_en": not d_en,
            "tr_len": len(tr["full"]), "en_len": len(en["full"]),
        })

    (LABELS / "reports_generated.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")
    df = pd.DataFrame(rows)
    df.to_csv(LOGS / "10_report_roundtrip.csv", index=False, encoding="utf-8")

    n = len(df)
    ok_tr, ok_en = int(df["roundtrip_tr"].sum()), int(df["roundtrip_en"].sum())
    print(f"üretilen rapor       : {n} vaka (TR + EN)")
    print(f"gidiş-dönüş TR       : {ok_tr}/{n} ({ok_tr / n:.3f})")
    print(f"gidiş-dönüş EN       : {ok_en}/{n} ({ok_en / n:.3f})")
    print(f"benzersiz facts_hash : {df['facts_hash'].nunique()}")
    print(f"ortalama TR uzunluğu : {df['tr_len'].mean():.0f} karakter")

    if failures:
        print(f"\n=== {len(failures)} vakada gidiş-dönüş başarısız (ilk 3) ===")
        for f in failures[:3]:
            print(f"\n--- {f['case_id']}")
            if f["tr"]:
                print("  TR farkları:", f["tr"][:4])
                print("  TR metin:", f["tr_text"].replace("\n", " ")[:400])
            if f["en"]:
                print("  EN farkları:", f["en"][:4])
                print("  EN metin:", f["en_text"].replace("\n", " ")[:400])
        (LOGS / "10_roundtrip_failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        print("\nörnek rapor (ilk vaka):")
        first = reports[sorted(reports)[0]]
        print(first["tr"]["full"])
        print()
        print(first["en"]["full"])

    gate = ok_tr == n and ok_en == n
    print(f"\nKAPI: %100 gidiş-dönüş -> {'PASS' if gate else 'FAIL'}")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
