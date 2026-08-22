"""Aşama 2 — Excel adlarını DICOM klasörlerine eşle, anonim ID ata.

Çıktılar (hepsi 00_private, PHI içerir, ASLA yayınlanmaz):
  phi_map.csv                    patient_id <-> gerçek ad <-> klasörler
  name_alias_table.csv           tetiklenen takma adlar (kapı: tam 7)
  duplicate_name_resolution.csv  mükerrer adlar için karar
  anon_salt.txt                  ID sırasını belirleyen tuz
  match_audit.csv                üç kaynağın (TAŞ BT / raporlar / klasör) tam ilişkisi
"""
from __future__ import annotations

import hashlib
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.paths import INDEX, PRIVATE, XLSX_PRIMARY, XLSX_SECONDARY, ensure_dirs
from common.trnorm import ALIASES, match_method, name_key

CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
MIN_CT_FILES = 40  # bu eşiğin altı yalnızca doz raporu/scout demek


def load_excels() -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = pd.read_excel(XLSX_PRIMARY)
    primary = primary[primary["AD SOYAD"].notna()].reset_index(drop=True)
    primary["name_key"] = primary["AD SOYAD"].map(name_key)

    secondary = pd.read_excel(XLSX_SECONDARY, header=1)
    secondary = secondary[secondary["ad_soyad"].notna()].reset_index(drop=True)
    secondary["name_key"] = secondary["ad_soyad"].map(name_key)
    return primary, secondary


def folder_inventory() -> pd.DataFrame:
    """Klasör başına CT dosya sayısı ve hangi köklerden geldiği."""
    df = pd.read_parquet(PRIVATE / "dicom_index.parquet")
    ct = df[df["sop_class_uid"] == CT_IMAGE_STORAGE]
    grp = (
        ct.groupby("patient_folder")
        .agg(
            n_ct_files=("sop_instance_uid", "size"),
            n_unique_sop=("sop_instance_uid", "nunique"),
            n_series=("series_instance_uid", "nunique"),
            roots=("root_id", lambda s: "|".join(sorted(set(s)))),
            collections=("collection", lambda s: "|".join(sorted(set(s)))),
            dicom_names=("dicom_patient_name", lambda s: "|".join(sorted(set(s)))),
        )
        .reset_index()
    )
    # CT taşımayan klasörler de görünür kalsın (aksi hâlde sessizce kaybolurlar)
    all_folders = pd.DataFrame({"patient_folder": sorted(df["patient_folder"].unique())})
    grp = all_folders.merge(grp, on="patient_folder", how="left").fillna(
        {"n_ct_files": 0, "n_unique_sop": 0, "n_series": 0, "roots": "", "collections": "", "dicom_names": ""}
    )
    grp["name_key"] = grp["patient_folder"].map(name_key)
    return grp


def main() -> int:
    ensure_dirs()
    primary, secondary = load_excels()
    folders = folder_inventory()

    # --- klasörleri name_key üzerinde topla (aynı hastanın birden çok klasörü olabilir)
    fold_grp = (
        folders.groupby("name_key")
        .agg(
            folder_names=("patient_folder", lambda s: "|".join(sorted(set(s)))),
            n_folders=("patient_folder", "nunique"),
            n_ct_files=("n_ct_files", "sum"),
            n_unique_sop=("n_unique_sop", "sum"),
            n_series=("n_series", "sum"),
            roots=("roots", lambda s: "|".join(sorted({r for v in s for r in str(v).split("|") if r}))),
            collections=("collections", lambda s: "|".join(sorted({c for v in s for c in str(v).split("|") if c}))),
        )
        .reset_index()
    )

    pk = set(primary["name_key"])
    sk = set(secondary["name_key"])
    fk = set(fold_grp["name_key"])

    print("=== kaynak boyutları (name_key bazında) ===")
    print(f"TAŞ BT.xlsx (birincil)     : {len(pk)} hasta ({len(primary)} satır)")
    print(f"raporlar_düzenlenmiş.xlsx  : {len(sk)} hasta ({len(secondary)} satır)")
    print(f"DICOM klasörü              : {len(fk)} hasta ({len(folders)} klasör)")

    print("\n=== üç yönlü kesişim ===")
    print(f"her üçünde birden          : {len(pk & sk & fk)}")
    print(f"TAŞ BT ∩ raporlar          : {len(pk & sk)}")
    print(f"raporlar ∩ klasör          : {len(sk & fk)}")
    print(f"TAŞ BT'de var raporlar'da yok : {sorted(pk - sk)}")
    print(f"raporlar'da var TAŞ BT'de yok : {sorted(sk - pk)}")
    print(f"raporlar'da var klasörü yok   : {sorted(sk - fk)}")

    # --- etiketli kohort: raporlar_düzenlenmiş'te olan hastalar (plan: 195)
    labeled_keys = sorted(sk)
    holdout_keys = sorted(fk - sk)

    # --- takma ad denetimi
    alias_rows = []
    for excel_name, folder_name in ALIASES.items():
        k = name_key(excel_name)
        fired = k in fk
        alias_rows.append(
            {
                "excel_name": excel_name,
                "folder_name": folder_name,
                "name_key": k,
                "fired": fired,
                "in_labeled_cohort": k in sk,
            }
        )
    alias_df = pd.DataFrame(alias_rows)
    alias_df.to_csv(PRIVATE / "name_alias_table.csv", index=False, encoding="utf-8")
    n_fired = int(alias_df["fired"].sum())
    print(f"\n=== takma ad tablosu: {n_fired}/{len(alias_df)} tetiklendi ===")
    print(alias_df.to_string(index=False))

    # --- mükerrer adlar (aynı name_key birden çok Excel satırında)
    dup = secondary[secondary.duplicated("name_key", keep=False)].copy()
    dup_rows = []
    for k, g in dup.groupby("name_key"):
        fg = fold_grp[fold_grp["name_key"] == k]
        n_folders = int(fg["n_folders"].iloc[0]) if len(fg) else 0
        folder_names = fg["folder_names"].iloc[0] if len(fg) else ""
        identical = g["rapor_tam"].nunique() == 1
        dup_rows.append(
            {
                "name_key": k,
                "excel_names": "|".join(g["ad_soyad"]),
                "n_excel_rows": len(g),
                "n_folders": n_folders,
                "folder_names": folder_names,
                "reports_identical": identical,
                # İki klasör varsa Excel satır sırasına göre eşleştirilebilir;
                # tek klasör + farklı rapor varsa hangi satırın bu görüntüye ait
                # olduğu belirlenemez -> label_ambiguous.
                "resolution": (
                    "DEDUP_IDENTICAL_ROWS" if identical
                    else ("PAIR_BY_ROW_ORDER" if n_folders >= len(g) else "LABEL_AMBIGUOUS")
                ),
            }
        )
    dup_df = pd.DataFrame(dup_rows)
    dup_df.to_csv(PRIVATE / "duplicate_name_resolution.csv", index=False, encoding="utf-8")
    print("\n=== mükerrer adlar ===")
    print(dup_df.to_string(index=False) if len(dup_df) else "  (yok)")

    # --- anonim ID ataması: sha256(salt + name_key) sırasına göre
    salt_path = PRIVATE / "anon_salt.txt"
    if salt_path.exists():
        salt = salt_path.read_text(encoding="utf-8").strip()
    else:
        salt = secrets.token_hex(16)
        salt_path.write_text(salt, encoding="utf-8")

    def order_hash(k: str) -> str:
        return hashlib.sha256((salt + k).encode("utf-8")).hexdigest()

    labeled_sorted = sorted(labeled_keys, key=order_hash)
    holdout_sorted = sorted(holdout_keys, key=order_hash)
    ids = {k: f"KS{i:04d}" for i, k in enumerate(labeled_sorted, start=1)}
    ids.update({k: f"KU{i:04d}" for i, k in enumerate(holdout_sorted, start=1)})

    # --- phi_map
    prim_by_key = {r["name_key"]: r for _, r in primary.iterrows()}
    sec_by_key: dict[str, list] = {}
    for _, r in secondary.iterrows():
        sec_by_key.setdefault(r["name_key"], []).append(r)
    fold_by_key = {r["name_key"]: r for _, r in fold_grp.iterrows()}

    rows = []
    for k in labeled_sorted + holdout_sorted:
        s = sec_by_key.get(k, [None])[0]
        p = prim_by_key.get(k)
        f = fold_by_key.get(k)
        excel_name = (s["ad_soyad"] if s is not None else None) or (p["AD SOYAD"] if p is not None else None)
        folder_names = f["folder_names"] if f is not None else ""
        first_folder = folder_names.split("|")[0] if folder_names else ""
        rows.append(
            {
                "patient_id": ids[k],
                "group": "labeled" if k in sk else "holdout_unlabeled",
                "name_key": k,
                "name_raw_secondary": s["ad_soyad"] if s is not None else None,
                "name_raw_primary": p["AD SOYAD"] if p is not None else None,
                "folder_names": folder_names,
                "n_folders": int(f["n_folders"]) if f is not None else 0,
                "n_ct_files": int(f["n_ct_files"]) if f is not None else 0,
                "n_unique_sop": int(f["n_unique_sop"]) if f is not None else 0,
                "n_series": int(f["n_series"]) if f is not None else 0,
                "roots": f["roots"] if f is not None else "",
                "collections": f["collections"] if f is not None else "",
                "in_primary_xlsx": p is not None,
                "in_secondary_xlsx": s is not None,
                "n_secondary_rows": len(sec_by_key.get(k, [])),
                "name_match_method": match_method(excel_name, first_folder) if (excel_name and first_folder) else "NO_IMAGING",
                "has_usable_imaging": bool(f is not None and f["n_ct_files"] >= MIN_CT_FILES),
            }
        )
    phi = pd.DataFrame(rows)
    phi.to_csv(PRIVATE / "phi_map.csv", index=False, encoding="utf-8")

    lab = phi[phi["group"] == "labeled"]
    hol = phi[phi["group"] == "holdout_unlabeled"]
    print("\n=== ID ataması ===")
    print(f"etiketli (KS)  : {len(lab)}   görüntüsü kullanılabilir: {int(lab['has_usable_imaging'].sum())}")
    print(f"holdout  (KU)  : {len(hol)}   görüntüsü kullanılabilir: {int(hol['has_usable_imaging'].sum())}")
    print("\neşleşme yöntemi dağılımı (etiketli):")
    print(lab["name_match_method"].value_counts().to_string())

    bad = lab[~lab["has_usable_imaging"]]
    if len(bad):
        print(f"\nUYARI: görüntüsü yetersiz {len(bad)} etiketli hasta:")
        print(bad[["patient_id", "name_raw_secondary", "n_ct_files", "n_folders", "roots"]].to_string(index=False))

    # Yayınlanabilir kopya: gerçek ad, klasör adı ve name_key (katlanmış ad) çıkarılır
    phi.drop(columns=["name_raw_secondary", "name_raw_primary", "folder_names", "name_key"]).to_csv(
        INDEX / "patient_index_public.csv", index=False, encoding="utf-8"
    )

    # --- denetim izi: her name_key'in üç kaynaktaki durumu
    audit_keys = sorted(pk | sk | fk)
    audit = pd.DataFrame(
        {
            "name_key": audit_keys,
            "in_primary": [k in pk for k in audit_keys],
            "in_secondary": [k in sk for k in audit_keys],
            "has_folder": [k in fk for k in audit_keys],
            "patient_id": [ids.get(k, "") for k in audit_keys],
            "n_ct_files": [int(fold_by_key[k]["n_ct_files"]) if k in fold_by_key else 0 for k in audit_keys],
        }
    )
    audit.to_csv(PRIVATE / "match_audit.csv", index=False, encoding="utf-8")

    n_labeled_ok = int(lab["has_usable_imaging"].sum())
    gate = len(lab) == 195 and n_labeled_ok == 195 and n_fired == len(alias_df)
    print(f"\nKAPI: etiketli=195 ({len(lab)}), görüntülü=195 ({n_labeled_ok}), "
          f"takma ad {n_fired}/{len(alias_df)} -> {'PASS' if gate else 'FAIL'}")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
