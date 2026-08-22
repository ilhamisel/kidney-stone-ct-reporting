"""Aşama 11 — vaka başına rapor JSON'u, düz CSV aynaları ve manifest.

Şema kasıtlı olarak böbrek başına alt nesneler taşır (`targets.kidneys.right/left`):
half-crop bulgusu lateraliteyi "hangi taraf?" tek problemi yerine böbrek başına
iki bağımsız problem olarak modellemeyi gerektiriyor. Asıl modelleme birimi
`labels_kidney.csv` (394 satır) olacaktır.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.hu import HU_OFFSET, MW3_ORDER, WINDOWS
from common.labels import size_class
from common.paths import ARCHIVE, INDEX, LABELS, PRIVATE, XLSX_PRIMARY, XLSX_SECONDARY, ensure_dirs

SCHEMA_VERSION = "1.0.0"
BUILDER_VERSION = "kidneyct2027/1.0.0"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def code_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for f in sorted(list((root / "common").glob("*.py")) + list((root / "scripts").glob("*.py"))):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def kidney_block(rec: dict, fs: dict, side: str) -> dict:
    key = "right" if side == "RIGHT" else "left"
    k = fs["kidneys"][key]
    stones = [s for s in rec["stones"] if s["side"] == side]
    idxs = [s["stone_index"] for s in stones]
    zones = {z: False for z in ("upper", "mid", "lower")}
    for s in stones:
        if s["zone"] == "UPPER":
            zones["upper"] = True
        elif s["zone"] == "LOWER":
            zones["lower"] = True
        elif s["zone"] == "MID":
            zones["mid"] = True
        elif s["zone"] == "MID_UPPER":
            zones["mid"] = zones["upper"] = True
        elif s["zone"] == "MID_LOWER":
            zones["mid"] = zones["lower"] = True
    dominant = "UNKNOWN"
    with_mm = [s for s in stones if s["size_known"]]
    if with_mm:
        dominant = max(with_mm, key=lambda s: s["size_mm"])["zone"]
    elif stones:
        dominant = stones[0]["zone"]
    # taş sayısı: bu tarafta tarif edilenler; "çok sayıda" durumunda bilinmiyor
    n_stones = len(stones) if rec["count_qualifier"] != "MANY" else None
    if rec["count_qualifier"] == "EXACT" and rec["stone_count_declared"] and \
            rec["stone_count_declared"] > rec["stone_count_listed"]:
        n_stones = None  # tarif edilmeyen taşlar hangi tarafta bilinmiyor
    return {
        "has_stone": bool(stones) or (rec["laterality"] in ("BILATERAL", side)),
        "n_stones": n_stones,
        "n_characterized": len(stones),
        "max_size_mm": k["max_mm"],
        "max_size_class": k["max_cls"],
        "zones_present": zones,
        "dominant_zone": dominant,
        "stone_indices": idxs,
        "crop": None,  # aşama 08 (böbrek segmentasyonu) dolduracak
    }


def imaging_block(case_id: str, sel: pd.DataFrame, salt: str) -> dict:
    rows = sel[sel["case_id"] == case_id]
    series = []
    for _, r in rows.iterrows():
        meta_path = ARCHIVE / case_id / r["role"] / "meta.json"
        archived = meta_path.exists()
        m = json.loads(meta_path.read_text(encoding="utf-8")) if archived else {}
        series.append({
            "role": r["role"],
            "series_uid_hash": m.get("series_uid_hash") or hashlib.sha256(
                (salt + r["series_instance_uid"]).encode()).hexdigest()[:16],
            "plane": r["plane"],
            "n_slices": int(m.get("n_slices", r["n_slices"])),
            "rows": 512, "cols": 512,
            "pixel_spacing_mm": [float(r["pixel_spacing_r"]), float(r["pixel_spacing_c"])],
            "slice_thickness_mm": float(r["slice_thickness"]),
            "median_dz_mm": float(r["median_dz_mm"]),
            "z_extent_mm": float(r["z_extent_mm"]),
            "irregular_spacing": bool(m.get("irregular_spacing", r["dz_std_mm"] > 0.01)),
            "has_gap": bool(m.get("has_gap", False)),
            "transfer_syntaxes": r["transfer_syntaxes"],
            "series_description": r["series_description"],
            "source_roots": r["roots"],
            "archive_dir": f"02_archive/{case_id}/{r['role']}" if archived else None,
            "archived": archived,
            "selection_reason": r["selection_reason"],
        })
    return {
        "series": series,
        "primary_series_role": "axial_std",
        "has_thin_axial": any(s["role"] == "axial_thin" for s in series),
        "windows": {n: {"wl": WINDOWS[n][0], "ww": WINDOWS[n][1]} for n in WINDOWS},
        "mw3_channel_order": list(MW3_ORDER),
        "hu": {"offset": HU_OFFSET, "scale": 1, "bitdepth": 16,
               "encoding": "png_uint16 = HU + hu_offset"},
    }


def main() -> int:
    ensure_dirs()
    parsed = json.loads((PRIVATE / "labels_parsed.json").read_text(encoding="utf-8"))
    reports = json.loads((LABELS / "reports_generated.json").read_text(encoding="utf-8"))
    selected = json.loads((PRIVATE / "selected_slices_full.json").read_text(encoding="utf-8"))
    sel = pd.read_csv(PRIVATE / "series_selection.csv")
    cases = pd.read_csv(PRIVATE / "cases.csv").set_index("case_id")
    salt = (PRIVATE / "anon_salt.txt").read_text(encoding="utf-8").strip()

    out_dir = LABELS / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ch = code_hash()

    master, kidney_rows, stone_rows = [], [], []
    for case_id, rec in sorted(parsed.items()):
        rep = reports[case_id]
        fs = rep["factset"]
        c = cases.loc[case_id]
        right = kidney_block(rec, fs, "RIGHT")
        left = kidney_block(rec, fs, "LEFT")
        mms = [s["size_mm"] for s in rec["stones"] if s["size_known"]]

        doc = {
            "schema_version": SCHEMA_VERSION,
            "case_id": case_id,
            "patient_id": rec["patient_id"],
            "cohort": {
                "group": "labeled",
                "collection": c["collection"],
                "study_date": str(c["study_date"]),
                "study_assignment": rec["study_assignment"],
                "study_assignment_verified": rec["study_assignment_verified"],
            },
            "demographics": {
                "age_years": rec["age_years"],
                "age_known": rec["age_years"] is not None,
                "sex": rec["sex"],
                "sex_source_raw": rec["sex_source_raw"],
            },
            "labels": {
                "provenance": {
                    "primary_source": "TAS_BT.xlsx",
                    "primary_row_index": rec["primary_row_index"],
                    "secondary_source": "raporlar_duzenlenmis.xlsx",
                    "secondary_row_index": rec["secondary_row_index"],
                    "report_style": rec["report_style"],
                },
                "stone_count_declared": rec["stone_count_declared"],
                "stone_count_listed": rec["stone_count_listed"],
                "count_qualifier": rec["count_qualifier"],
                "count_mismatch": rec["count_mismatch"],
                "stones": rec["stones"],
                "anomalies": rec["anomalies"],
                "free_text_note": rec["free_text_note"],
                "conflicts": rec.get("conflicts", []),
                "adjudication": rec.get("adjudication"),
                "notes": rec["notes"],
                "flags": {
                    "label_ambiguous": not rec["study_assignment_verified"],
                    "free_text_laterality_ambiguous": rec["free_text_laterality_ambiguous"],
                    "size_all_missing": not mms,
                    "ureter_or_bladder_stone_mentioned": "URETERAL_STONE_PRESENT" in rec["anomalies"],
                },
            },
            "targets": {
                "any_stone": True,
                "laterality": rec["laterality"],
                "n_stones_effective": rec["n_stones_effective"],
                "count_bucket": rec["count_bucket"],
                "max_size_mm": max(mms) if mms else None,
                "max_size_class": size_class(max(mms)) if mms else None,
                "min_size_mm": min(mms) if mms else None,
                "sum_size_mm": round(sum(mms), 1) if mms else None,
                "kidneys": {"right": right, "left": left},
            },
            "imaging": imaging_block(case_id, sel, salt),
            # source_folder gerçek hasta klasör adıdır -> yayınlanan JSON'a girmez
            "selected_slices": (
                {k: v for k, v in selected[case_id].items() if k != "source_folder"}
                if case_id in selected else None
            ),
            "stone_candidates": [],  # aşama 09 dolduracak
            "reports": {
                "template_version": rep["template_version"],
                "generator_seed": rep["generator_seed"],
                "facts_hash": rep["facts_hash"],
                "variant_ids": rep["variant_ids"],
                "stone_tuples": rep["stone_tuples"],
                "tr": rep["tr"],
                "en": rep["en"],
                "original_report_tr": rep["original_report_tr"],
                "roundtrip_parse_ok": rep["roundtrip_parse_ok"],
            },
            "splits": {},  # aşama 12 dolduracak
            "provenance": {
                "built_at_utc": now,
                "builder_version": BUILDER_VERSION,
                "code_sha256": ch,
                "python_version": sys.version.split()[0],
            },
        }
        (out_dir / f"{case_id}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

        master.append({
            "case_id": case_id, "patient_id": rec["patient_id"],
            "age_years": rec["age_years"], "sex": rec["sex"],
            "laterality": rec["laterality"], "any_stone": True,
            "n_stones_effective": rec["n_stones_effective"],
            "count_bucket": rec["count_bucket"], "count_qualifier": rec["count_qualifier"],
            "count_mismatch": rec["count_mismatch"],
            "max_size_mm": max(mms) if mms else None,
            "max_size_class": size_class(max(mms)) if mms else None,
            "right_has_stone": right["has_stone"], "right_n_characterized": right["n_characterized"],
            "right_max_mm": right["max_size_mm"], "right_max_class": right["max_size_class"],
            "right_dominant_zone": right["dominant_zone"],
            "left_has_stone": left["has_stone"], "left_n_characterized": left["n_characterized"],
            "left_max_mm": left["max_size_mm"], "left_max_class": left["max_size_class"],
            "left_dominant_zone": left["dominant_zone"],
            "report_style": rec["report_style"], "anomalies": "|".join(rec["anomalies"]),
            "label_ambiguous": not rec["study_assignment_verified"],
            "n_selected_slices": (selected.get(case_id) or {}).get("n", 0),
            "n_slices": next((s["n_slices"] for s in doc["imaging"]["series"]
                              if s["role"] == "axial_std"), None),
            "facts_hash": rep["facts_hash"],
        })
        for side, blk in (("RIGHT", right), ("LEFT", left)):
            kidney_rows.append({
                "case_id": case_id, "patient_id": rec["patient_id"], "side": side,
                "has_stone": blk["has_stone"], "n_stones": blk["n_stones"],
                "n_characterized": blk["n_characterized"],
                "max_mm": blk["max_size_mm"], "max_class": blk["max_size_class"],
                "zone_upper": blk["zones_present"]["upper"], "zone_mid": blk["zones_present"]["mid"],
                "zone_lower": blk["zones_present"]["lower"], "dominant_zone": blk["dominant_zone"],
            })
        for s in rec["stones"]:
            stone_rows.append({"case_id": case_id, "patient_id": rec["patient_id"], **s})

    m = pd.DataFrame(master)
    m.to_csv(LABELS / "labels_master.csv", index=False, encoding="utf-8")
    kd = pd.DataFrame(kidney_rows)
    kd.to_csv(LABELS / "labels_kidney.csv", index=False, encoding="utf-8")
    st = pd.DataFrame(stone_rows)
    st.to_csv(LABELS / "labels_stone.csv", index=False, encoding="utf-8")

    manifest = {
        "dataset_name": "KidneyCT_2027",
        "version": "1.0.0",
        "created_utc": now,
        "schema_version": SCHEMA_VERSION,
        "report_template_version": reports[next(iter(reports))]["template_version"],
        "code_sha256": ch,
        "source_files": [
            {"name": "TAŞ BT.xlsx", "role": "primary_labels", "sha256": sha256_file(XLSX_PRIMARY)},
            {"name": "raporlar_düzenlenmiş.xlsx", "role": "secondary_reports",
             "sha256": sha256_file(XLSX_SECONDARY)},
        ],
        "cohort": {
            "n_cases": len(m), "n_patients": int(m["patient_id"].nunique()),
            "n_kidneys": len(kd), "n_stones": len(st),
            "n_label_ambiguous": int(m["label_ambiguous"].sum()),
        },
        "counts": {
            "sex": m["sex"].value_counts().to_dict(),
            "age": {"min": int(m["age_years"].min()), "max": int(m["age_years"].max()),
                    "missing": int(m["age_years"].isna().sum())},
            "laterality": m["laterality"].value_counts().to_dict(),
            "count_bucket": m["count_bucket"].value_counts().to_dict(),
            "max_size_class": m["max_size_class"].value_counts().to_dict(),
            "stone_size_class": st["size_class"].value_counts().to_dict(),
            "stone_zone": st["zone"].value_counts().to_dict(),
            "report_style": m["report_style"].value_counts().to_dict(),
        },
        "imaging": {
            "scanner": "GE Revolution EVO", "matrix": [512, 512], "patient_position": "HFS",
            "archive_format": f"PNG-16 grayscale, value = HU + {HU_OFFSET}",
            "windows": {n: list(WINDOWS[n]) for n in WINDOWS},
            "mw3_channel_order": list(MW3_ORDER),
        },
        "selected_slices": {
            "n_cases": sum(1 for c in selected if c in parsed),
            "n_slices": sum(v["n"] for c, v in selected.items() if c in parsed),
            "index_semantics": "AXIAL_ROW",
            "verification": "scripts/07b_verify_index_semantics.py — eski koronal PNG ile "
                            "piksel korelasyonu j'de tepe yapıyor (binom p < 1e-30)",
        },
        "tasks_supported": [
            "per_kidney_stone_presence", "laterality", "max_size_class",
            "count_bucket", "zone_multilabel", "report_generation",
        ],
        "tasks_not_supported": [
            "stone_detection — negatif vaka yok: 197/197 vakada taş var. Bu kohort bir "
            "TESPİT değil KARAKTERİZASYON veri setidir.",
        ],
        "known_issues": [
            f"{int(m['count_mismatch'].sum())} vakada beyan edilen taş sayısı tarif edilenden fazla "
            "(radyolog yalnızca en büyükleri tanımlamış); count_qualifier ve n_stones=null ile temsil edildi",
            "2 hastanın 2'şer tetkiki var (KS0140, KS0178); Excel satır bloğu ile koleksiyon "
            "eşleştirmesi kullanıldı, label_ambiguous=true ile işaretlendi",
            "1 vakada at nalı böbrek (KS0077): zon semantiği farklı",
            "1 vakada serbest metin lateralitesi belirsiz ('sol iki böbrekte')",
        ],
    }
    (LABELS / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"rapor JSON       : {len(m)} dosya -> {out_dir}")
    print(f"labels_master    : {len(m)} satır")
    print(f"labels_kidney    : {len(kd)} satır")
    print(f"labels_stone     : {len(st)} satır")
    print(f"\nlateralite       : {manifest['counts']['laterality']}")
    print(f"maks boyut sınıfı: {manifest['counts']['max_size_class']}")
    print(f"taş boyut sınıfı : {manifest['counts']['stone_size_class']}")
    print(f"taş zonu         : {manifest['counts']['stone_zone']}")
    print(f"count_bucket     : {manifest['counts']['count_bucket']}")
    print(f"cinsiyet         : {manifest['counts']['sex']}   yaş: {manifest['counts']['age']}")
    print(f"\nböbrek düzeyi hedef: sağ taşlı={int(kd[kd.side == 'RIGHT'].has_stone.sum())}/197, "
          f"sol taşlı={int(kd[kd.side == 'LEFT'].has_stone.sum())}/197")
    print(f"arşivi hazır seri  : {sum(1 for c in parsed for s in json.loads((out_dir / f'{c}.json').read_text(encoding='utf-8'))['imaging']['series'] if s['archived'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
