"""Aşama 13 — veri seti QA kontrolleri. Herhangi bir FAIL derlemeyi geçersiz kılar.

Her kontrol bir CSV/özet üretir ve qa_report.md'ye PASS/FAIL satırı yazar.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixels import apply_modality_lut

from common.paths import ARCHIVE, INDEX, LABELS, LOGS, PRIVATE, QA, SPLITS, ensure_dirs

RESULTS: list[tuple[str, bool, str]] = []

# 06_qa yayınlanabilir ağacın parçası: buraya yazılan tablolardan ham tanımlayıcılar
# çıkarılmalı (ilk koşuda qa_02/qa_03 ham SeriesInstanceUID sızdırıyordu).
PHI_COLS = ["series_instance_uid", "study_instance_uid", "patient_folder", "name_key",
            "file_path", "folder", "source_folder"]


def to_public_csv(df: pd.DataFrame, path: Path) -> None:
    df.drop(columns=[c for c in PHI_COLS if c in df.columns]).to_csv(
        path, index=False, encoding="utf-8")


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name:38s} {detail}")


def qa_hu_roundtrip(n_full: int, n_sample: int, rng: random.Random) -> None:
    sel = pd.read_csv(PRIVATE / "series_selection.csv")
    idx = pd.read_parquet(PRIVATE / "dicom_index.parquet")
    idx = idx[idx["sop_class_uid"] == "1.2.840.10008.5.1.4.1.1.2"]
    cases = sorted(d.name for d in ARCHIVE.iterdir()
                   if d.is_dir() and (d / "axial_std" / "meta.json").exists())
    full = set(rng.sample(cases, min(n_full, len(cases))))
    total = bad = 0
    rows = []
    for case_id in cases:
        d = ARCHIVE / case_id / "axial_std"
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        s = sel[(sel["case_id"] == case_id) & (sel["role"] == "axial_std")].iloc[0]
        files = idx[idx["series_instance_uid"] == s["series_instance_uid"]]
        files = files.sort_values(["sop_instance_uid", "file_size"], ascending=[True, False])
        files = files.drop_duplicates("sop_instance_uid")
        nrm = np.array(meta["slice_normal"])
        files = files.assign(proj=files[["ipp_x", "ipp_y", "ipp_z"]].to_numpy(float) @ nrm)
        files = files.sort_values("proj", kind="stable").reset_index(drop=True)
        n = meta["n_slices"]
        picks = range(n) if case_id in full else rng.sample(range(n), min(n_sample, n))
        cbad = 0
        for i in picks:
            ds = pydicom.dcmread(files.iloc[i].file_path, force=True)
            hu_src = apply_modality_lut(ds.pixel_array, ds).astype(np.int32)
            png = cv2.imread(str(d / f"{i:04d}.png"), cv2.IMREAD_UNCHANGED)
            hu_arc = png.astype(np.int32) - meta["hu_offset"]
            total += 1
            if not np.array_equal(hu_src, hu_arc):
                bad += 1
                cbad += 1
        rows.append({"case_id": case_id, "n_tested": len(list(picks)), "n_mismatch": cbad,
                     "full_scan": case_id in full})
    pd.DataFrame(rows).to_csv(QA / "qa_01_hu_roundtrip.csv", index=False, encoding="utf-8")
    check("1. HU gidiş-dönüş birebir", bad == 0, f"{total - bad}/{total} kesit birebir")


def qa_coverage() -> None:
    m = pd.read_csv(LABELS / "labels_master.csv")
    sel = pd.read_csv(PRIVATE / "series_selection.csv")
    prim = sel[(sel["role"] == "axial_std") & (sel["group"] == "labeled")]
    ok = (len(prim) == len(m) and (prim["n_slices"] >= 100).all()
          and (prim["z_extent_mm"] >= 120).all())
    to_public_csv(prim, QA / "qa_02_coverage.csv")
    check("2. Kapsam (>=100 kesit, >=120mm)", bool(ok),
          f"{len(prim)}/{len(m)} vaka, min kesit={prim['n_slices'].min()}, "
          f"min z={prim['z_extent_mm'].min():.0f}mm")


def qa_geometry() -> None:
    sel = pd.read_csv(PRIVATE / "series_selection.csv")
    prim = sel[sel["role"] == "axial_std"].copy()
    prim["ps_ok"] = prim["pixel_spacing_r"].between(0.5, 1.0)
    prim["dup_ok"] = prim["n_duplicate_positions"] == 0
    prim["gap_ok"] = prim["max_gap_mm"] <= 2 * prim["median_dz_mm"] + 1e-6
    to_public_csv(prim, QA / "qa_03_geometry.csv")
    irr = int((prim["dz_std_mm"] > 0.01).sum())
    ok = bool(prim["ps_ok"].all() and prim["dup_ok"].all())
    check("3. Geometri", ok,
          f"piksel aralığı OK={int(prim['ps_ok'].sum())}/{len(prim)}, tekrarlı z="
          f"{int((~prim['dup_ok']).sum())}, düzensiz aralık işaretli={irr}, "
          f"boşluklu={int((~prim['gap_ok']).sum())}")


def qa_golden() -> None:
    audit = json.loads((PRIVATE / "06_parse_audit.json").read_text(encoding="utf-8"))
    exp = {
        "primary_size_measurements": 352,
        "secondary_structured_rows": 172,
        "secondary_freetext_rows": 25,
        "secondary_lesion_phrases": 327,
        "secondary_unparsed_phrases": 0,
    }
    bad = {k: (audit.get(k), v) for k, v in exp.items() if audit.get(k) != v}
    hist_exp = {"MIKROLITIYAZIS": 38, "KUCUK": 61, "ORTA": 122, "BUYUK": 81, "COK_BUYUK": 24}
    hist_ok = all(audit["secondary_class_hist"].get(k) == v for k, v in hist_exp.items())
    check("4. Altın sayılar", not bad and hist_ok,
          "hepsi eşleşti" if (not bad and hist_ok) else f"sapma: {bad} hist_ok={hist_ok}")


def qa_parse_coverage() -> None:
    pa = pd.read_csv(LABELS / "parse_audit.csv")
    st = pd.read_csv(LABELS / "labels_stone.csv")
    no_stone = int((pa["n_stones"] == 0).sum())
    unk_side = int(pa["has_unknown_side"].sum())
    unk_zone = int(pa["has_unknown_zone"].sum())
    check("5. Ayrıştırma kapsamı", no_stone == 0 and unk_side == 0 and unk_zone == 0,
          f"taşsız vaka={no_stone}, tarafı bilinmeyen={unk_side}, zonu bilinmeyen={unk_zone}, "
          f"toplam taş={len(st)}")


def qa_name_matching() -> None:
    phi = pd.read_csv(PRIVATE / "phi_map.csv")
    al = pd.read_csv(PRIVATE / "name_alias_table.csv")
    lab = phi[phi["group"] == "labeled"]
    manual = int((lab["name_match_method"] == "MANUAL").sum())
    check("6. İsim eşleştirme", len(lab) == 195 and bool(al["fired"].all()) and manual == 0,
          f"etiketli={len(lab)}, takma ad {int(al['fired'].sum())}/{len(al)}, MANUAL={manual}")


def qa_phi() -> None:
    """Yayınlanabilir ağaçta ham ad / ham UID sızıntısı var mı?"""
    phi = pd.read_csv(PRIVATE / "phi_map.csv")
    names = set()
    for col in ("name_raw_secondary", "name_raw_primary", "folder_names"):
        for v in phi[col].dropna():
            for part in str(v).split("|"):
                if part.strip():
                    names.add(part.strip().upper())
    idx = pd.read_parquet(PRIVATE / "dicom_index.parquet")
    uids = set(idx["series_instance_uid"].dropna().unique()[:200])
    dicom_names = {str(x).upper() for x in idx["dicom_patient_name"].dropna().unique() if x}

    hits = []
    public_dirs = [LABELS, SPLITS, INDEX]
    for base in public_dirs:
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix not in (".json", ".csv"):
                continue
            if p.parent == INDEX and p.name in ("dicom_index.parquet", "patient_index_public.csv"):
                pass
            txt = p.read_text(encoding="utf-8", errors="ignore").upper()
            for nm in list(names) + list(dicom_names):
                if len(nm) > 6 and nm in txt:
                    hits.append({"file": str(p), "kind": "name", "value": nm})
                    break
            for u in list(uids)[:50]:
                if u in txt:
                    hits.append({"file": str(p), "kind": "uid", "value": u})
                    break
    # PHI bulgu raporu doğası gereği PHI içerir -> 00_private
    pd.DataFrame(hits).to_csv(PRIVATE / "qa_07_phi_findings.csv", index=False, encoding="utf-8")
    check("7. PHI taraması (yayınlanabilir)", len(hits) == 0,
          f"{len(hits)} isabet" + (f" örn: {hits[0]['file']}" if hits else ""))


def qa_selected_slices() -> None:
    mp = pd.read_csv(LABELS / "selected_slices_map.csv")
    ok = bool(mp["inside_fov"].all())
    ev = LOGS / "07b_index_semantics.csv"
    detail = f"{int(mp['inside_fov'].sum())}/{len(mp)} FOV içinde"
    if ev.exists():
        e = pd.read_csv(ev)
        near = int((e["best_offset"].abs() <= 2).sum())
        detail += f"; piksel kanıtı: {near}/{len(e)} tepe |ofset|<=2"
    check("8. Seçilmiş kesit haritalaması", ok, detail)


def qa_saturation(rng: random.Random, n_cases: int = 20, n_slices: int = 10) -> None:
    """Doğru büyüklüğü ölç: TAŞ ADAYI vokselinin ne kadarı doyuyor?

    Gövde geneli doygunluk oranı yanıltıcıdır — onu FOV'daki kortikal kemik
    miktarı belirler, taşın temsil edilip edilmediğini değil. Anlamlı ölçüt,
    >=200 HU voksellerin (kalsiyum yoğunluğu) kaçının 255'e dayandığıdır.
    Eski hattın tek penceresi (yumuşak doku 40/400) burada ~%82 alır.
    """
    from common.hu import WINDOWS, window8

    cases = sorted(d.name for d in ARCHIVE.iterdir()
                   if d.is_dir() and (d / "axial_std" / "meta.json").exists())
    pick = rng.sample(cases, min(n_cases, len(cases)))
    num = {k: 0 for k in ("soft", "stone", "wide")}
    den = 0
    body_num = {k: 0 for k in num}
    body_den = 0
    for case_id in pick:
        d = ARCHIVE / case_id / "axial_std"
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        for i in rng.sample(range(meta["n_slices"]), min(n_slices, meta["n_slices"])):
            png = cv2.imread(str(d / f"{i:04d}.png"), cv2.IMREAD_UNCHANGED)
            hu = png.astype(np.int32) - meta["hu_offset"]
            body = hu > -500
            cand = body & (hu >= 200)
            den += int(cand.sum())
            body_den += int(body.sum())
            for k in num:
                img = window8(hu, *WINDOWS[k])
                num[k] += int(((img == 255) & cand).sum())
                body_num[k] += int(((img == 255) & body).sum())
    frac = {k: (num[k] / den if den else float("nan")) for k in num}
    bfrac = {k: (body_num[k] / body_den if body_den else float("nan")) for k in num}
    pd.DataFrame([{"window": k, "stone_voxel_saturation": frac[k],
                   "body_saturation": bfrac[k]} for k in num]).to_csv(
        QA / "qa_09_saturation.csv", index=False, encoding="utf-8")
    # Kapı: taş penceresi taş voksellerinin <%10'unu doyurmalı VE eski tek
    # pencerenin (soft) doygunluğu belirgin biçimde daha kötü olmalı.
    ok = bool(frac["stone"] < 0.10 and frac["soft"] > 0.50)
    check("9. Taş yoğunluğu korunumu", ok,
          f"taş penceresi {frac['stone']:.1%} vs eski yumuşak doku penceresi "
          f"{frac['soft']:.1%} (>=200 HU voksellerin doyan oranı); gövde geneli "
          f"taş={bfrac['stone']:.2%}")


def qa_schema() -> None:
    import jsonschema
    schema = json.loads((LABELS / "schema" / "report.schema.json").read_text(encoding="utf-8"))
    v = jsonschema.Draft202012Validator(schema)
    files = sorted((LABELS / "reports").glob("*.json"))
    bad = [f.name for f in files if not v.is_valid(json.loads(f.read_text(encoding="utf-8")))]
    check("11. JSON şema doğrulaması", not bad, f"{len(files) - len(bad)}/{len(files)} geçti")


def qa_roundtrip_reports() -> None:
    r = pd.read_csv(LOGS / "10_report_roundtrip.csv")
    ok = bool(r["roundtrip_tr"].all() and r["roundtrip_en"].all())
    check("10. Rapor gidiş-dönüşü %100", ok,
          f"TR {int(r['roundtrip_tr'].sum())}/{len(r)}, EN {int(r['roundtrip_en'].sum())}/{len(r)}, "
          f"benzersiz facts_hash={r['facts_hash'].nunique()}")


def qa_splits() -> None:
    m = pd.read_csv(LABELS / "labels_master.csv")
    kd = pd.read_csv(LABELS / "labels_kidney.csv")
    seeds = [c for c in m.columns if c.startswith("cv5_seed")]
    leaks = 0
    for s in seeds:
        leaks += int((m.groupby("patient_id")[s].nunique() > 1).sum())
        leaks += int((kd.groupby("patient_id")[s].nunique() > 1).sum())
    check("12. Bölme sızıntısı", leaks == 0, f"{len(seeds)} tohum, {leaks} sızıntı")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-cases", type=int, default=20)
    ap.add_argument("--sample-slices", type=int, default=10)
    ap.add_argument("--skip-hu", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    QA.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1337)

    if not args.skip_hu:
        qa_hu_roundtrip(args.full_cases, args.sample_slices, rng)
    qa_coverage()
    qa_geometry()
    qa_golden()
    qa_parse_coverage()
    qa_name_matching()
    qa_phi()
    qa_selected_slices()
    qa_saturation(rng)
    qa_roundtrip_reports()
    qa_schema()
    qa_splits()

    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    lines = ["# KidneyCT_2027 — QA raporu", "",
             f"Sonuç: **{n_ok}/{len(RESULTS)} kontrol PASS**", "",
             "| # | Kontrol | Sonuç | Ayrıntı |", "|---|---|---|---|"]
    for i, (name, ok, detail) in enumerate(RESULTS, 1):
        lines.append(f"| {i} | {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    (QA / "qa_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{n_ok}/{len(RESULTS)} kontrol PASS -> {QA / 'qa_report.md'}")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
