"""Stage 3 — study and series selection.

A fully auditable replacement for the earlier pipeline's `os.listdir(...)[0]`,
which could pick a dose report or a scout image. The imaging plane is ALWAYS
derived from ImageOrientationPatient; SeriesDescription is not trustworthy, and
that was verified rather than assumed — series whose descriptions name one plane
turn out to hold reformats in another.

Every rejection is recorded with its reason, so the selection can be reviewed
afterwards instead of taken on trust.

Outputs:
  series_index.csv       one row per series, with plane and rejection reason
  series_selection.csv   the series selected per patient
  rejected_series.csv    the rejected series and why
  cases.csv              case_id <-> patient_id <-> study_instance_uid
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from common.paths import INDEX, PRIVATE, XLSX_SECONDARY, ensure_dirs
from common.trnorm import name_key

CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
MIN_SLICES = 20
MIN_Z_EXTENT_MM = 100.0
AXIAL_STD_THICKNESS = (2.0, 3.5)
AXIAL_THIN_MAX = 1.25

# Resolution for the patients who have two examinations, verified case by case.
# Where a patient has two spreadsheet rows, the row block coincides with the
# source collection, so the examinations are paired on that basis and the pairing
# is marked UNVERIFIED — it is an inference, not a recorded fact.
# Resolution rules for the patients who have two examinations, each verified case
# by case during exploration. The keys are normalized patient names and are
# therefore identifying, so the table is not part of this repository; it is read
# from the private configuration directory (see docs/data-availability.md).
# Modes:
#   TWO_CASES_BY_COLLECTION  two spreadsheet rows; the row block distinguishes the
#                            collections, so each examination is matched to its own
#                            row and both are flagged unverified
#   PREFER_COLLECTION        one spreadsheet row; keep the examination belonging to
#                            the named collection
#   PREFER_LARGEST_AXIAL     one spreadsheet row; keep the examination with the
#                            largest axial series
def _load_study_resolution() -> dict:
    f = PRIVATE / "study_resolution.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text(encoding="utf-8"))


STUDY_RESOLUTION = _load_study_resolution()


def classify_plane(iop: np.ndarray) -> str:
    if np.any(~np.isfinite(iop)):
        return "UNKNOWN"
    normal = np.cross(iop[0:3], iop[3:6])
    mag = np.abs(normal)
    if mag.max() < 0.95:
        return "OBLIQUE"
    return ["SAGITTAL", "CORONAL", "AXIAL"][int(np.argmax(mag))]


def build_series_table(ct: pd.DataFrame) -> pd.DataFrame:
    """One row per series. Thickness and plane are taken from the median over the
    series, so that a single odd reference image cannot misclassify it."""
    recs = []
    for (suid,), g in ct.groupby(["series_instance_uid"], sort=False):
        iop = g[["iop_0", "iop_1", "iop_2", "iop_3", "iop_4", "iop_5"]].median().to_numpy(dtype=float)
        plane = classify_plane(iop)
        normal = np.cross(iop[0:3], iop[3:6]) if np.all(np.isfinite(iop)) else np.array([0, 0, 1.0])
        nrm = np.linalg.norm(normal)
        normal = normal / nrm if nrm > 0 else np.array([0, 0, 1.0])

        uniq = g.drop_duplicates("sop_instance_uid")
        ipp = uniq[["ipp_x", "ipp_y", "ipp_z"]].to_numpy(dtype=float)
        if np.all(np.isfinite(ipp)) and len(ipp):
            proj = ipp @ normal
            z_extent = float(np.ptp(proj)) if len(proj) > 1 else 0.0
            dz = np.diff(np.sort(proj))
            median_dz = float(np.median(dz)) if len(dz) else float("nan")
            dz_std = float(np.std(dz)) if len(dz) else float("nan")
            max_gap = float(dz.max()) if len(dz) else float("nan")
            n_dup_pos = int(len(proj) - len(np.unique(np.round(proj, 3))))
        else:
            z_extent = median_dz = dz_std = max_gap = float("nan")
            n_dup_pos = 0

        it = "|".join(sorted({str(x) for x in g["image_type"] if x}))
        recs.append(
            {
                "series_instance_uid": suid,
                "study_instance_uid": g["study_instance_uid"].iloc[0],
                "study_date": g["study_date"].iloc[0],
                "patient_folder": g["patient_folder"].mode().iloc[0],
                "name_key": name_key(g["patient_folder"].mode().iloc[0]),
                "collections": "|".join(sorted(set(g["collection"]))),
                "roots": "|".join(sorted(set(g["root_id"]))),
                "series_description": g["series_description"].mode().iloc[0],
                "series_number": g["series_number"].iloc[0],
                "image_type": it,
                "is_original": "ORIGINAL" in it,
                "is_localizer": ("LOCALIZER" in it) or ("SCOUT" in it),
                "plane": plane,
                "n_files": int(len(g)),
                "n_slices": int(uniq["sop_instance_uid"].nunique()),
                "slice_thickness": float(uniq["slice_thickness"].median()),
                "pixel_spacing_r": float(uniq["pixel_spacing_r"].median()),
                "pixel_spacing_c": float(uniq["pixel_spacing_c"].median()),
                "rows": int(uniq["rows"].median()) if uniq["rows"].notna().any() else -1,
                "columns": int(uniq["columns"].median()) if uniq["columns"].notna().any() else -1,
                "rescale_intercept": float(uniq["rescale_intercept"].median()),
                "rescale_slope": float(uniq["rescale_slope"].median()),
                "z_extent_mm": z_extent,
                "median_dz_mm": median_dz,
                "dz_std_mm": dz_std,
                "max_gap_mm": max_gap,
                "n_duplicate_positions": n_dup_pos,
                "transfer_syntaxes": "|".join(sorted(set(g["transfer_syntax"].dropna()))),
                "normal_x": float(normal[0]), "normal_y": float(normal[1]), "normal_z": float(normal[2]),
            }
        )
    return pd.DataFrame(recs)


def reject_reason(r: pd.Series) -> str | None:
    if r["is_localizer"]:
        return "localizer"
    if r["n_slices"] < MIN_SLICES:
        return "too_few_slices"
    if r["rows"] != 512 or r["columns"] != 512:
        return "nonstandard_matrix"
    if r["plane"] == "OBLIQUE":
        return "oblique"
    if r["plane"] == "UNKNOWN":
        return "no_orientation"
    if r["plane"] == "AXIAL" and r["z_extent_mm"] < MIN_Z_EXTENT_MM:
        return "insufficient_coverage"
    return None


def resolve_studies(series: pd.DataFrame, phi: pd.DataFrame, row_block: dict[str, list[int]]) -> pd.DataFrame:
    """Build the case list per patient. A case is a (patient, examination) pair."""
    cases = []
    for _, p in phi.iterrows():
        k = p["name_key"]
        g = series[(series["name_key"] == k) & (series["reject_reason"].isna())]
        if g.empty:
            cases.append({"patient_id": p["patient_id"], "name_key": k, "case_id": p["patient_id"],
                          "study_instance_uid": None, "study_date": None, "collection": None,
                          "excel_row": None, "study_assignment": "NO_USABLE_SERIES", "verified": False})
            continue
        studies = (
            g.groupby(["study_instance_uid", "study_date"])
            .agg(n_axial=("plane", lambda s: int((s == "AXIAL").sum())),
                 max_axial_slices=("n_slices", "max"),
                 collections=("collections", lambda s: "|".join(sorted({c for v in s for c in v.split("|")}))))
            .reset_index()
            .sort_values("study_date")
        )
        rows = sorted(row_block.get(k, []))
        rule = STUDY_RESOLUTION.get(k)

        if len(studies) == 1:
            cases.append({"patient_id": p["patient_id"], "name_key": k, "case_id": p["patient_id"],
                          "study_instance_uid": studies.iloc[0]["study_instance_uid"],
                          "study_date": studies.iloc[0]["study_date"],
                          "collection": studies.iloc[0]["collections"],
                          "excel_row": rows[0] if rows else None,
                          "study_assignment": "SINGLE_STUDY", "verified": True})
            continue

        mode = rule["mode"] if rule else "PREFER_LARGEST_AXIAL"
        if mode == "TWO_CASES_BY_COLLECTION" and len(rows) == len(studies):
            # map the spreadsheet row block onto the source collection
            def row_pref(row_idx: int) -> str:
                return "BOBREK_TASI" if row_idx <= 126 else "BOBREK_URETER_TASI"

            used = set()
            for suffix, row_idx in zip("AB", rows):
                want = row_pref(row_idx)
                cand = studies[studies["collections"].str.contains(want) & ~studies["study_instance_uid"].isin(used)]
                if cand.empty:
                    cand = studies[~studies["study_instance_uid"].isin(used)]
                s = cand.iloc[0]
                used.add(s["study_instance_uid"])
                cases.append({"patient_id": p["patient_id"], "name_key": k,
                              "case_id": f"{p['patient_id']}_{suffix}",
                              "study_instance_uid": s["study_instance_uid"], "study_date": s["study_date"],
                              "collection": s["collections"], "excel_row": row_idx,
                              "study_assignment": "ROW_BLOCK_COLLECTION", "verified": False})
            continue

        if mode == "PREFER_COLLECTION":
            cand = studies[studies["collections"].str.contains(rule["collection"])]
            s = (cand if not cand.empty else studies).iloc[0]
            assign = "PREFER_COLLECTION"
        else:
            s = studies.sort_values("max_axial_slices", ascending=False).iloc[0]
            assign = "PREFER_LARGEST_AXIAL"
        cases.append({"patient_id": p["patient_id"], "name_key": k, "case_id": p["patient_id"],
                      "study_instance_uid": s["study_instance_uid"], "study_date": s["study_date"],
                      "collection": s["collections"], "excel_row": rows[0] if rows else None,
                      "study_assignment": assign, "verified": bool(rule and rule.get("verified"))})
    return pd.DataFrame(cases)


def select_for_case(g: pd.DataFrame) -> list[dict]:
    """Select the series roles from the eligible series of one examination."""
    out = []
    ax = g[g["plane"] == "AXIAL"]
    std = ax[ax["slice_thickness"].between(*AXIAL_STD_THICKNESS)]
    thin = ax[ax["slice_thickness"] <= AXIAL_THIN_MAX]

    def pick(df: pd.DataFrame) -> pd.Series | None:
        if df.empty:
            return None
        d = df.sort_values(["is_original", "n_slices"], ascending=[False, False])
        return d.iloc[0]

    primary = pick(std)
    reason = "2.0-3.5mm axial, ORIGINAL preferred, most slices"
    if primary is None:
        primary = pick(thin)
        reason = "fallback: thin axial (no 2.5mm series)"
    if primary is None:
        primary = pick(ax)
        reason = "fallback: any axial"
    if primary is not None:
        out.append({"role": "axial_std", "series_instance_uid": primary["series_instance_uid"],
                    "selection_reason": reason})

    t = pick(thin)
    if t is not None and (primary is None or t["series_instance_uid"] != primary["series_instance_uid"]):
        out.append({"role": "axial_thin", "series_instance_uid": t["series_instance_uid"],
                    "selection_reason": "<=1.25mm axial, most slices"})

    for plane, role in (("CORONAL", "coronal_native"), ("SAGITTAL", "sagittal_native")):
        s = pick(g[g["plane"] == plane])
        if s is not None:
            out.append({"role": role, "series_instance_uid": s["series_instance_uid"],
                        "selection_reason": f"{plane.lower()} reformat with most slices"})
    return out


PHI_COLS = ["patient_folder", "name_key", "series_instance_uid", "study_instance_uid"]


def write_public_index(series: pd.DataFrame, rej: pd.DataFrame, sel: pd.DataFrame,
                       cases: pd.DataFrame) -> None:
    """Publishable copies: the patient folder name and the raw UIDs are removed,
    and the UIDs are replaced by salted hashes."""
    import hashlib

    salt = (PRIVATE / "anon_salt.txt").read_text(encoding="utf-8").strip()

    def h(u: str) -> str:
        return hashlib.sha256((salt + str(u)).encode("utf-8")).hexdigest()[:16]

    for df, name in ((series, "series_index.csv"), (rej, "rejected_series.csv"),
                     (sel, "series_selection.csv"), (cases, "cases.csv")):
        d = df.copy()
        for c in ("series_instance_uid", "study_instance_uid"):
            if c in d.columns:
                d[c.replace("_instance_uid", "_uid_hash")] = d[c].map(h)
        d = d.drop(columns=[c for c in PHI_COLS if c in d.columns])
        d.to_csv(INDEX / name, index=False, encoding="utf-8")


def main() -> int:
    ensure_dirs()
    df = pd.read_parquet(PRIVATE / "dicom_index.parquet")
    ct = df[df["sop_class_uid"] == CT_IMAGE_STORAGE].copy()
    print(f"CT images: {len(ct)} files, {ct['series_instance_uid'].nunique()} series")

    series = build_series_table(ct)
    series["reject_reason"] = series.apply(reject_reason, axis=1)
    # the full version carries the folder name and raw UIDs -> private directory
    series.to_csv(PRIVATE / "series_index_full.csv", index=False, encoding="utf-8")

    rej = series[series["reject_reason"].notna()]
    rej.to_csv(PRIVATE / "rejected_series_full.csv", index=False, encoding="utf-8")
    print("\n=== series rejection reasons ===")
    print(rej["reject_reason"].value_counts().to_string() if len(rej) else "  (yok)")
    ok = series[series["reject_reason"].isna()]
    print("\n=== plane of the accepted series ===")
    print(ok["plane"].value_counts().to_string())

    # spreadsheet row indices, used for the study resolution
    sec = pd.read_excel(XLSX_SECONDARY, header=1)
    sec = sec[sec["ad_soyad"].notna()].reset_index(drop=True)
    row_block: dict[str, list[int]] = {}
    for i, nm in enumerate(sec["ad_soyad"]):
        row_block.setdefault(name_key(nm), []).append(i)

    phi = pd.read_csv(PRIVATE / "phi_map.csv")
    cases = resolve_studies(ok, phi, row_block)
    cases = cases.merge(phi[["patient_id", "group"]], on="patient_id", how="left")
    cases.to_csv(PRIVATE / "cases.csv", index=False, encoding="utf-8")

    print("\n=== case resolution ===")
    print(cases["study_assignment"].value_counts().to_string())
    lab_cases = cases[cases["group"] == "labeled"]
    print(f"etiketli vaka: {len(lab_cases)} (hasta: {lab_cases['patient_id'].nunique()})")
    print(f"unverified assignments: {int((~lab_cases['verified']).sum())}")
    if int((~lab_cases["verified"]).sum()):
        print(lab_cases[~lab_cases["verified"]][
            ["case_id", "name_key", "study_date", "collection", "excel_row", "study_assignment"]
        ].to_string(index=False))

    # --- role selection
    sel_rows = []
    for _, c in cases.iterrows():
        if not c["study_instance_uid"]:
            continue
        g = ok[(ok["name_key"] == c["name_key"]) & (ok["study_instance_uid"] == c["study_instance_uid"])]
        for pick in select_for_case(g):
            s = series[series["series_instance_uid"] == pick["series_instance_uid"]].iloc[0]
            sel_rows.append(
                {
                    "case_id": c["case_id"], "patient_id": c["patient_id"], "group": c["group"],
                    "role": pick["role"], "selection_reason": pick["selection_reason"],
                    "series_instance_uid": s["series_instance_uid"],
                    "study_instance_uid": s["study_instance_uid"], "study_date": s["study_date"],
                    "plane": s["plane"], "n_slices": s["n_slices"],
                    "slice_thickness": s["slice_thickness"], "median_dz_mm": s["median_dz_mm"],
                    "dz_std_mm": s["dz_std_mm"], "max_gap_mm": s["max_gap_mm"],
                    "n_duplicate_positions": s["n_duplicate_positions"],
                    "z_extent_mm": s["z_extent_mm"],
                    "pixel_spacing_r": s["pixel_spacing_r"], "pixel_spacing_c": s["pixel_spacing_c"],
                    "rescale_intercept": s["rescale_intercept"], "rescale_slope": s["rescale_slope"],
                    "series_description": s["series_description"], "roots": s["roots"],
                    "transfer_syntaxes": s["transfer_syntaxes"],
                }
            )
    sel = pd.DataFrame(sel_rows)
    # the full version carries the raw SeriesInstanceUID, which the archive builder needs
    sel.to_csv(PRIVATE / "series_selection.csv", index=False, encoding="utf-8")
    write_public_index(series, rej, sel, cases)

    print("\n=== distribution of selected roles (labelled cohort) ===")
    ls = sel[sel["group"] == "labeled"]
    print(ls["role"].value_counts().to_string())
    prim = ls[ls["role"] == "axial_std"]
    print(f"\naxial_std olan etiketli vaka : {len(prim)} / {len(lab_cases)}")
    print(f"  slice count   min={prim['n_slices'].min()} median={prim['n_slices'].median():.0f} max={prim['n_slices'].max()}")
    print(f"  z extent mm   min={prim['z_extent_mm'].min():.0f} median={prim['z_extent_mm'].median():.0f}")
    print(f"  thickness distribution: {prim['slice_thickness'].value_counts().to_dict()}")
    print(f"  irregular spacing (dz_std>0.01): {int((prim['dz_std_mm'] > 0.01).sum())}")
    print(f"  with duplicate positions       : {int((prim['n_duplicate_positions'] > 0).sum())}")
    print(f"  selected via a fallback        : {int(prim['selection_reason'].str.startswith('fallback').sum())}")

    gate = len(prim) == len(lab_cases)
    print(f"\nKAPI: her etiketli vakada axial_std var -> {'PASS' if gate else 'FAIL'}")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
