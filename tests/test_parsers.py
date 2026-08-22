"""Ayrıştırıcı regresyon testleri.

Çalıştırma:  PYTHONPATH=src python tests/test_parsers.py
(pytest gerekmez; bağımsız çalışır ve sıfır/bir ile çıkar.)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.factset import build_factset, facts_hash
from common.labels import (
    parse_free_text, parse_structured_findings, parse_zone_cell, size_class,
    strip_patient_name,
)
from common.report_parser import diff_factsets, parse_report
from common.templates import choose_variants, render_en, render_tr

FAILS: list[str] = []


def check(name: str, got, want) -> None:
    if got != want:
        FAILS.append(f"{name}: {got!r} != {want!r}")
        print(f"FAIL  {name}: {got!r} != {want!r}")
    else:
        print(f"ok    {name}")


# ---------------------------------------------------------------- boyut sınıfı
def test_size_class() -> None:
    # doğrulanmış sınırlar: <=3 / 4-5 / 6-10 / 11-19 / >=20
    for mm, want in [(2, "MIKROLITIYAZIS"), (3, "MIKROLITIYAZIS"), (3.5, "MIKROLITIYAZIS"),
                     (4, "KUCUK"), (5, "KUCUK"), (5.5, "KUCUK"),
                     (6, "ORTA"), (10, "ORTA"), (10.5, "ORTA"),
                     (11, "BUYUK"), (19, "BUYUK"), (19.9, "BUYUK"),
                     (20, "COK_BUYUK"), (55, "COK_BUYUK")]:
        check(f"size_class({mm})", size_class(mm), want)
    check("size_class(None)", size_class(None), None)


# ---------------------------------------------------------------- zon hücresi
def test_zone_cell() -> None:
    z = parse_zone_cell("SAĞ ALT ZON")
    check("zone SAĞ ALT", (z["side"], z["zone"]), ("RIGHT", "LOWER"))
    z = parse_zone_cell("SOL ORTA-ÜST ZON")
    check("zone SOL ORTA-ÜST", (z["side"], z["zone"]), ("LEFT", "MID_UPPER"))
    z = parse_zone_cell("SOL ORTA ALT ZON")
    check("zone SOL ORTA ALT", (z["side"], z["zone"]), ("LEFT", "MID_LOWER"))
    z = parse_zone_cell("SAĞ OR")  # kesik yazım
    check("zone kesik 'SAĞ OR'", (z["side"], z["zone"], z["zone_inferred"]),
          ("RIGHT", "MID", True))
    z = parse_zone_cell("AT NALI BÖBREK MEVCUT")
    check("anomali at nalı", z["anomaly"], "HORSESHOE_KIDNEY")


# ---------------------------------------------------------------- serbest metin
def test_free_text() -> None:
    # plandaki zorunlu regresyon vakası: OCR '1O' -> 10
    f = parse_free_text("her iki böbrekte büyüğü sağ orta zonda 7 mm boyutta olmak üzere "
                        "1O adet taş dansitesi izlendi.")
    check("1O -> 10", f["declared_count"], 10)
    check("1O onarım bayrağı", f["count_repaired"], True)
    check("lateralite her iki", f["laterality_cue"], "BILATERAL")
    check("en büyük taraf", f["largest_side"], "RIGHT")
    check("en büyük zon", f["largest_zone"], "MID")
    check("en büyük mm", f["largest_mm"], 7.0)

    f = parse_free_text("sol iki böbrekte büyüğü alt zonda 15 mm boyutta olmak üzere5 adet "
                        "taş dansitesi izlendi.")
    check("'sol iki böbrekte' -> LEFT", f["laterality_cue"], "LEFT")
    check("'sol iki böbrekte' belirsiz", f["laterality_ambiguous"], True)

    f = parse_free_text("sağ böbrekte büyüğü orta zonda 20 mm boyutta olmak üzere 5 adet "
                        "taş dansitesi izlendi.")
    check("sağ böbrekte -> RIGHT", f["laterality_cue"], "RIGHT")
    check("taraf cümle başından", f["largest_side"], "RIGHT")


# ---------------------------------------------------------------- yapılandırılmış
def test_structured() -> None:
    txt = ("SAĞ ALT ZON yerleşimli, 6.0 mm (Orta boy taş); "
           "SOL ÜST ZON yerleşimli, 12.0 mm (Büyük taş)")
    ls = parse_structured_findings(txt)
    check("2 lezyon", len(ls), 2)
    check("hepsi ayrıştırıldı", all(x["parsed"] for x in ls), True)
    check("lezyon1", (ls[0]["side"], ls[0]["zone"], ls[0]["size_mm"]), ("RIGHT", "LOWER", 6.0))
    check("lezyon2", (ls[1]["side"], ls[1]["zone"], ls[1]["size_mm"]), ("LEFT", "UPPER", 12.0))

    ls = parse_structured_findings("AT NALI BÖBREK MEVCUT yerleşimli, boyut belirtilmemiş (Bilinmiyor)")
    check("boyutsuz lezyon ayrıştırıldı", ls[0]["parsed"], True)
    check("boyutsuz lezyon mm", ls[0]["size_mm"], None)


# ---------------------------------------------------------------- PHI temizliği
def test_strip_name() -> None:
    t = "Hasta: AYSE YILMAZ\nBulgular: SAĞ ALT ZON...\nSonuç: 5 adet."
    out = strip_patient_name(t, "KS0042")
    check("ad temizlendi", "AYSE YILMAZ" in out, False)
    check("case_id yerleşti", out.startswith("Hasta: KS0042"), True)
    check("gövde korundu", "Bulgular: SAĞ ALT ZON..." in out, True)


# ---------------------------------------------------------------- gidiş-dönüş
def _rec(stones, lat, total=None, qual="EXACT", anomalies=()):
    return {
        "stones": [dict(stone_index=i + 1, side=s[0], zone=s[1], size_mm=s[2],
                        size_known=s[2] is not None, size_class=size_class(s[2]),
                        zone_raw=None, zone_inferred=False, source_column="test")
                   for i, s in enumerate(stones)],
        "laterality": lat, "n_stones_effective": total if total is not None else len(stones),
        "count_qualifier": qual, "stone_count_declared": total, "stone_count_listed": len(stones),
        "anomalies": list(anomalies),
    }


def test_roundtrip() -> None:
    cases = {
        "tek taş": _rec([("LEFT", "LOWER", 12.0)], "LEFT"),
        "bilateral çoklu": _rec([("RIGHT", "MID", 7.0), ("RIGHT", "UPPER", 4.0),
                                 ("LEFT", "LOWER", 8.0)], "BILATERAL"),
        "beyan>tanımlanan": _rec([("RIGHT", "MID", 11.0)], "BILATERAL", total=10),
        "çok sayıda": _rec([("LEFT", "MID", 9.0)], "BILATERAL", total=None, qual="MANY"),
        "at nalı": _rec([("RIGHT", "LOWER", 7.0)], "RIGHT", anomalies=["HORSESHOE_KIDNEY"]),
        "8 mm (an/a testi)": _rec([("LEFT", "MID", 8.0)], "LEFT"),
        "18 mm (an testi)": _rec([("RIGHT", "UPPER", 18.0)], "RIGHT"),
        "ondalık mm": _rec([("LEFT", "LOWER", 4.5)], "LEFT"),
    }
    for name, rec in cases.items():
        fs = build_factset(rec)
        for vi in range(6):  # tüm varyant kombinasyonlarını kabaca tara
            v = choose_variants(f"KS{vi:04d}", 1337)
            for lang, render in (("tr", render_tr), ("en", render_en)):
                txt = render(fs, v)["full"]
                back = parse_report(txt, lang)
                d = diff_factsets(fs, back)
                if d:
                    FAILS.append(f"gidiş-dönüş {name}/{lang}/v{vi}: {d[:2]}")
                    print(f"FAIL  gidiş-dönüş {name}/{lang}/v{vi}: {d[:2]}")
                    print(f"      {txt}")
                    break
        else:
            print(f"ok    gidiş-dönüş {name} (tr+en, 6 varyant)")


def test_facts_hash_stability() -> None:
    rec = _rec([("LEFT", "LOWER", 12.0)], "LEFT")
    a = facts_hash(build_factset(rec))
    b = facts_hash(build_factset(rec))
    check("facts_hash kararlı", a, b)
    rec2 = _rec([("LEFT", "LOWER", 13.0)], "LEFT")
    check("facts_hash içeriğe duyarlı", a != facts_hash(build_factset(rec2)), True)


def main() -> int:
    for fn in (test_size_class, test_zone_cell, test_free_text, test_structured,
               test_strip_name, test_roundtrip, test_facts_hash_stability):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print(f"\n{'BAŞARISIZ: ' + str(len(FAILS)) if FAILS else 'TÜM TESTLER GEÇTİ'}")
    for f in FAILS:
        print("  ", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
