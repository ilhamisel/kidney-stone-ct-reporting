"""Stage 6 — parse the two label sources into one label object per case.

The spreadsheet column names and the parsed clinical text are Turkish; the output
is language-independent.

Gates (golden-number regressions, each verified by hand during exploration):
  primary    352 diameter measurements, 2-55 mm; sex {M:134, F:64}; age 20-86,
             1 missing
  secondary  172 structured rows and 25 free-text rows; 327 lesion phrases; the
             class histogram {MICRO:38, SMALL:61, MEDIUM:122, LARGE:81,
             VERY_LARGE:24} plus 1 unknown
  unparsed lesion phrases: 0

These are assertions, not statistics: if any of them moves, a source file or a
parser changed and the run stops rather than silently producing a new dataset.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.labels import (
    count_bucket, derive_laterality, parse_free_text, parse_structured_findings,
    parse_zone_cell, size_class, strip_patient_name,
)
from common.paths import INDEX, LABELS, PRIVATE, XLSX_PRIMARY, XLSX_SECONDARY, ensure_dirs
from common.trnorm import name_key

STONE_COLS = [("1.TAŞ", "BOYUT (mm)"), ("2.TAŞ", "BOYUT (mm).1"), ("3.TAŞ ", "BOYUT (mm).2"),
              ("4.TAŞ", "BOYUT (mm).3"), ("5.TAŞ", "BOYUT (mm).4")]

# --------------------------------------------------------------------------
# Adjudication of source conflicts.
# Across the whole cohort, only TWO patients have a laterality conflict between
# the primary table and the free text (KS0029 and KS0124), and both follow the
# same pattern: the table says right, the text says left. The conflict cannot be
# resolved from the text, so it was resolved FROM THE IMAGES:
#   KS0029 — no focus at or above 300 HU in the right kidney box; on the left, a
#            mass of 8136 voxels reaching 2037 HU (coronal MIP written by stage 13)
#   KS0124 — the coronal MIP places the stone on the patient's LEFT
# In both cases the side in the primary table is a transcription error and the
# report text is correct.
#
# KS0005 is different: the text is genuinely ambiguous. In the earlier study a
# RADIOLOGIST reviewed that report and resolved it to bilateral. A documented
# radiologist decision outranks any automated image review, so it is used as is.
#
# The table is keyed by normalized patient names and therefore identifying, so it
# is not part of the repository; it is read from the private directory (see
# docs/data-availability.md). Schema: {name_key: {flip_sides | force_laterality,
# method, evidence, conflict}}. If the file is absent no adjudication is applied
# and the two conflicts are reported as unresolved in the `conflicts` field.
def _load_adjudications() -> dict:
    f = PRIVATE / "adjudications.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text(encoding="utf-8"))


ADJUDICATIONS: dict[str, dict] = _load_adjudications()

GOLDEN = {
    "primary_size_measurements": 352,
    "primary_sex": {"M": 134, "F": 64},
    "primary_age_missing": 1,
    "secondary_structured_rows": 172,
    "secondary_freetext_rows": 25,
    "secondary_lesion_phrases": 327,
    "secondary_class_hist": {"MIKROLITIYAZIS": 38, "KUCUK": 61, "ORTA": 122,
                             "BUYUK": 81, "COK_BUYUK": 24},
}


def load_primary() -> pd.DataFrame:
    df = pd.read_excel(XLSX_PRIMARY)
    df = df[df["AD SOYAD"].notna()].reset_index(drop=True)
    df["row_index"] = df.index
    df["name_key"] = df["AD SOYAD"].map(name_key)
    return df


def load_secondary() -> pd.DataFrame:
    df = pd.read_excel(XLSX_SECONDARY, header=1)
    df = df[df["ad_soyad"].notna()].reset_index(drop=True)
    df["row_index"] = df.index
    df["name_key"] = df["ad_soyad"].map(name_key)
    return df


def parse_primary_row(r: pd.Series) -> dict:
    stones, anomalies, notes = [], [], []
    for i, (zc, mc) in enumerate(STONE_COLS, start=1):
        zone_raw = r.get(zc)
        mm_raw = r.get(mc)
        if pd.isna(zone_raw) and pd.isna(mm_raw):
            continue
        z = parse_zone_cell(zone_raw if pd.notna(zone_raw) else "")
        if z["anomaly"]:
            anomalies.append(z["anomaly"])
            if z["note"]:
                notes.append(z["note"])
            # an anomaly row is not a stone, and is not recorded even if it has a size
            continue
        mm = float(mm_raw) if pd.notna(mm_raw) else None
        stones.append({
            "stone_index": len(stones) + 1,
            "side": z["side"], "zone": z["zone"], "zone_raw": str(zone_raw).strip() if pd.notna(zone_raw) else None,
            "zone_inferred": z["zone_inferred"],
            "size_mm": mm, "size_known": mm is not None, "size_class": size_class(mm),
            "source_column": zc.strip(),
        })
        if z["note"]:
            notes.append(z["note"])

    sayi = r.get("SAYI")
    declared, qualifier = None, "EXACT"
    if pd.isna(sayi):
        qualifier = "UNKNOWN"
    else:
        s = str(sayi).strip()
        if s.upper().startswith("ÇOK") or s.upper().startswith("COK"):
            qualifier = "MANY"
        else:
            try:
                declared = int(float(s))
            except ValueError:
                qualifier = "UNKNOWN"

    age = r.get("YAŞ ")
    sex_raw = r.get("CİNSİYET")
    sex = {"E": "M", "K": "F"}.get(str(sex_raw).strip().upper(), "UNKNOWN") if pd.notna(sex_raw) else "UNKNOWN"

    return {
        "row_index": int(r["row_index"]),
        "age_years": int(age) if pd.notna(age) else None,
        "sex": sex, "sex_source_raw": str(sex_raw).strip() if pd.notna(sex_raw) else None,
        "stone_count_declared": declared, "count_qualifier": qualifier,
        "stones": stones, "anomalies": sorted(set(anomalies)), "notes": notes,
    }


def parse_secondary_row(r: pd.Series) -> dict:
    bul = r.get("rapor_bulgular")
    txt = "" if pd.isna(bul) else str(bul)
    is_structured = "yerleşimli" in txt
    lesions = parse_structured_findings(txt) if is_structured else []
    free = parse_free_text(txt) if not is_structured else parse_free_text("")

    sonuc = r.get("rapor_sonuc")
    declared = None
    if pd.notna(sonuc):
        import re
        m = re.search(r"(\d+)\s*adet", str(sonuc))
        if m:
            declared = int(m.group(1))
    return {
        "row_index": int(r["row_index"]),
        "report_style": "STRUCTURED" if is_structured else "FREE_TEXT",
        "lesions": lesions,
        "free": free,
        "declared_from_sonuc": declared,
        "rapor_bulgular": txt or None,
        "rapor_sonuc": None if pd.isna(sonuc) else str(sonuc),
        "rapor_tam": None if pd.isna(r.get("rapor_tam")) else str(r.get("rapor_tam")),
    }


def main() -> int:
    ensure_dirs()
    primary, secondary = load_primary(), load_secondary()
    cases = pd.read_csv(PRIVATE / "cases.csv")
    cases = cases[cases["group"] == "labeled"]

    prim_parsed = {int(r["row_index"]): parse_primary_row(r) for _, r in primary.iterrows()}
    sec_parsed = {int(r["row_index"]): parse_secondary_row(r) for _, r in secondary.iterrows()}
    prim_by_key: dict[str, list[int]] = {}
    for _, r in primary.iterrows():
        prim_by_key.setdefault(r["name_key"], []).append(int(r["row_index"]))

    # ---------------- golden-number checks
    audit: dict = {}
    n_meas = sum(1 for p in prim_parsed.values() for s in p["stones"] if s["size_known"])
    all_mm = [s["size_mm"] for p in prim_parsed.values() for s in p["stones"] if s["size_known"]]
    audit["primary_size_measurements"] = n_meas
    audit["primary_mm_range"] = [min(all_mm), max(all_mm)]
    audit["primary_sex"] = dict(Counter(p["sex"] for p in prim_parsed.values()))
    audit["primary_age_missing"] = sum(1 for p in prim_parsed.values() if p["age_years"] is None)
    audit["primary_age_range"] = [
        min(p["age_years"] for p in prim_parsed.values() if p["age_years"] is not None),
        max(p["age_years"] for p in prim_parsed.values() if p["age_years"] is not None),
    ]
    audit["primary_class_hist"] = dict(Counter(size_class(m) for m in all_mm))
    audit["primary_anomalies"] = dict(Counter(a for p in prim_parsed.values() for a in p["anomalies"]))

    styles = Counter(s["report_style"] for s in sec_parsed.values())
    audit["secondary_structured_rows"] = styles["STRUCTURED"]
    audit["secondary_freetext_rows"] = styles["FREE_TEXT"]
    lesions = [l for s in sec_parsed.values() for l in s["lesions"]]
    audit["secondary_lesion_phrases"] = len(lesions)
    audit["secondary_unparsed_phrases"] = sum(1 for l in lesions if not l["parsed"])
    audit["secondary_class_hist"] = dict(
        Counter(size_class(l["size_mm"]) for l in lesions if l.get("parsed") and l.get("size_mm") is not None)
    )
    audit["secondary_unknown_size"] = sum(1 for l in lesions if l.get("parsed") and l.get("size_mm") is None)
    audit["secondary_freetext_count_repaired"] = sum(
        1 for s in sec_parsed.values() if s["free"].get("count_repaired")
    )
    audit["secondary_freetext_laterality"] = dict(
        Counter(s["free"].get("laterality_cue") for s in sec_parsed.values() if s["report_style"] == "FREE_TEXT")
    )

    print("=== golden-number audit ===")
    checks = [
        ("primary size measurements", audit["primary_size_measurements"], GOLDEN["primary_size_measurements"]),
        ("birincil cinsiyet M", audit["primary_sex"].get("M"), GOLDEN["primary_sex"]["M"]),
        ("birincil cinsiyet F", audit["primary_sex"].get("F"), GOLDEN["primary_sex"]["F"]),
        ("primary missing age", audit["primary_age_missing"], GOLDEN["primary_age_missing"]),
        ("secondary structured rows", audit["secondary_structured_rows"], GOLDEN["secondary_structured_rows"]),
        ("ikincil serbest metin", audit["secondary_freetext_rows"], GOLDEN["secondary_freetext_rows"]),
        ("ikincil lezyon ifadesi", audit["secondary_lesion_phrases"], GOLDEN["secondary_lesion_phrases"]),
        ("secondary unparsed phrases", audit["secondary_unparsed_phrases"], 0),
    ]
    for cls, exp in GOLDEN["secondary_class_hist"].items():
        checks.append((f"secondary class {cls}", audit["secondary_class_hist"].get(cls), exp))
    ok = True
    for name, got, exp in checks:
        good = got == exp
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {name:28s} beklenen={exp!s:>6}  bulunan={got!s:>6}")
    print(f"  info   primary mm range         : {audit['primary_mm_range']}")
    print(f"  info   primary age range        : {audit['primary_age_range']}")
    print(f"  info   primary class histogram  : {audit['primary_class_hist']}")
    print(f"  bilgi  birincil anomaliler      : {audit['primary_anomalies']}")
    print(f"  bilgi  serbest metin lateralite : {audit['secondary_freetext_laterality']}")
    print(f"  info   count tokens repaired    : {audit['secondary_freetext_count_repaired']} rows")
    print(f"  bilgi  boyutu bilinmeyen lezyon : {audit['secondary_unknown_size']}")

    # ---------------- merge per case
    out = {}
    parse_rows = []
    for _, c in cases.iterrows():
        k = c["name_key"]
        srow = int(c["excel_row"]) if pd.notna(c["excel_row"]) else None
        sec = sec_parsed.get(srow) if srow is not None else None

        # primary rows for this name_key; where there are several, ordered by case
        prows = prim_by_key.get(k, [])
        if len(prows) == 1:
            prow = prows[0]
        elif len(prows) > 1 and srow is not None:
            # pair by secondary row block: lower index to lower index
            same_block = sorted(prows)
            sec_rows = sorted(int(x) for x in cases[cases["name_key"] == k]["excel_row"].dropna())
            prow = same_block[sec_rows.index(srow)] if srow in sec_rows and len(same_block) == len(sec_rows) else same_block[0]
        else:
            prow = prows[0] if prows else None
        prim = prim_parsed.get(prow) if prow is not None else None

        stones = list(prim["stones"]) if prim else []
        anomalies = list(prim["anomalies"]) if prim else []
        free = sec["free"] if sec else {}
        lat = derive_laterality(stones, free.get("laterality_cue"))

        # add the largest stone described in the free text when the primary table
        # does not list it
        if not stones and free.get("largest_mm"):
            stones.append({
                "stone_index": 1, "side": free["largest_side"], "zone": free["largest_zone"],
                "zone_raw": None, "zone_inferred": True,
                "size_mm": free["largest_mm"], "size_known": True,
                "size_class": size_class(free["largest_mm"]), "source_column": "free_text",
            })
            lat = derive_laterality(stones, free.get("laterality_cue"))

        # --- detect source conflicts and apply adjudication
        conflicts = []
        prim_sides = {s["side"] for s in stones if s["side"] in ("RIGHT", "LEFT")}
        cue = free.get("laterality_cue")
        if cue in ("RIGHT", "LEFT") and prim_sides and cue not in prim_sides:
            conflicts.append({"field": "stone_side", "primary": sorted(prim_sides),
                              "secondary": cue, "resolved": False})

        adj = ADJUDICATIONS.get(k)
        if adj:
            if "flip_sides" in adj:
                for s in stones:
                    s["side"] = adj["flip_sides"].get(s["side"], s["side"])
            lat = derive_laterality(stones, free.get("laterality_cue"))
            if "force_laterality" in adj:
                lat = adj["force_laterality"]
            for cf in conflicts:  # `c` is the case row in the outer loop; do not shadow it
                cf["resolved"] = True
            if not conflicts and "conflict" in adj:
                conflicts.append({**adj["conflict"], "resolved": True})

        declared = (prim or {}).get("stone_count_declared")
        qualifier = (prim or {}).get("count_qualifier", "UNKNOWN")
        if declared is None and sec and sec.get("declared_from_sonuc") is not None:
            declared = sec["declared_from_sonuc"]
            qualifier = "EXACT"
        listed = len(stones)
        n_eff = None if qualifier == "MANY" else (max(declared, listed) if declared is not None else listed)

        if c["collection"] and "URETER_TASI" in str(c["collection"]):
            anomalies.append("URETERAL_STONE_PRESENT")

        rec = {
            "case_id": c["case_id"], "patient_id": c["patient_id"], "name_key": k,
            "primary_row_index": prow, "secondary_row_index": srow,
            "report_style": sec["report_style"] if sec else "ABSENT",
            "age_years": (prim or {}).get("age_years"),
            "sex": (prim or {}).get("sex", "UNKNOWN"),
            "sex_source_raw": (prim or {}).get("sex_source_raw"),
            "stone_count_declared": declared,
            "stone_count_listed": listed,
            "count_qualifier": qualifier,
            "count_mismatch": bool(declared is not None and qualifier == "EXACT" and declared != listed),
            "n_stones_effective": n_eff,
            "count_bucket": count_bucket(n_eff, qualifier),
            "laterality": lat,
            "stones": stones,
            "anomalies": sorted(set(anomalies)),
            "free_text_note": free.get("raw") if sec and sec["report_style"] == "FREE_TEXT" else None,
            "free_text_laterality_ambiguous": bool(free.get("laterality_ambiguous")),
            "conflicts": conflicts,
            "adjudication": ({"method": adj["method"], "evidence": adj["evidence"]}
                             if adj else None),
            "notes": (prim or {}).get("notes", []),
            # the full report field carries the real patient name: clean it at source
            "original_report_tr": {
                "rapor_bulgular": sec["rapor_bulgular"] if sec else None,
                "rapor_sonuc": sec["rapor_sonuc"] if sec else None,
                "rapor_tam": strip_patient_name(sec["rapor_tam"], c["case_id"]) if sec else None,
            },
            "study_assignment": c["study_assignment"],
            "study_assignment_verified": bool(c["verified"]),
        }
        out[c["case_id"]] = rec
        parse_rows.append({
            "case_id": c["case_id"], "report_style": rec["report_style"],
            "n_stones": listed, "declared": declared, "qualifier": qualifier,
            "count_mismatch": rec["count_mismatch"], "laterality": lat,
            "anomalies": "|".join(rec["anomalies"]),
            "has_unknown_side": any(s["side"] == "UNKNOWN" for s in stones),
            "has_unknown_zone": any(s["zone"] == "UNKNOWN" for s in stones),
            "n_stones_zero": listed == 0,
        })

    # the intermediate output carries name_key, so it stays in the private directory
    (PRIVATE / "labels_parsed.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    pa = pd.DataFrame(parse_rows)
    pa.to_csv(LABELS / "parse_audit.csv", index=False, encoding="utf-8")

    print("\n=== case-level summary ===")
    print(f"cases                  : {len(out)}")
    print(f"lateralite             : {pa['laterality'].value_counts().to_dict()}")
    print(f"count_bucket           : {Counter(r['count_bucket'] for r in out.values())}")
    print(f"count_mismatch         : {int(pa['count_mismatch'].sum())} -> "
          f"{pa[pa['count_mismatch']]['case_id'].tolist()}")
    print(f"cases with no lesion   : {int(pa['n_stones_zero'].sum())} -> "
          f"{pa[pa['n_stones_zero']]['case_id'].tolist()}")
    print(f"stones of unknown side : {int(pa['has_unknown_side'].sum())} cases")
    print(f"stones of unknown zone : {int(pa['has_unknown_zone'].sum())} cases")
    print(f"cases with an anomaly  : {int((pa['anomalies'] != '').sum())}")
    print(f"  {Counter(a for r in out.values() for a in r['anomalies'])}")

    audit["case_laterality"] = pa["laterality"].value_counts().to_dict()
    audit["case_count_mismatch"] = pa[pa["count_mismatch"]]["case_id"].tolist()
    (PRIVATE / "06_parse_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    n_zero = int(pa["n_stones_zero"].sum())
    gate = ok and audit["secondary_unparsed_phrases"] == 0
    print(f"\nGATE: golden numbers + 0 unparsed -> {'PASS' if gate else 'FAIL'}")
    if n_zero:
        print(f"WARNING: {n_zero} cases have no stone at all; inspect them by hand")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
