"""Aşama 7 — radyolog seçimli 418 kesidin yeni hacme haritalanması.

Eski kod (folderprocess.ipynb) koronal reformatı şöyle üretiyordu:
    img3d[:, :, i] = window_image(slices[i])          # i = aksiyel kesit
    for j in range(...): abc = np.flipud(img3d[j, :, :].T)
Yani `j` aksiyel 512x512 dizisinin SATIR indeksidir -> hasta uzayında y
(ön->arka) koordinatı, kesit numarası değil. IOP=[1,0,0,0,1,0] olduğu için
satır yönü +y'dir ve satır aralığı PixelSpacing[0]'dır.

Üç hipotez kanıta göre sınanır; kazanan manifest'e yazılır.
  H1 AXIAL_ROW             tüm indeksler [0,512) içinde ve posteriora kaymış olmalı
  H2 AXIAL_SLICE           eski serinin kesit sayısından küçük olmalı
  H3 CORONAL_NATIVE_SLICE  koronal reformat kesit sayısından küçük olmalı
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from common.paths import INDEX, LABELS, PRIVATE, SELECTED_SLICES_DIR, ensure_dirs
from common.trnorm import name_key

DESKTOP_ROOT_IDS = {"AX_BOBREK", "AX_BU", "AX_UR"}


def load_selected() -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for d in sorted(SELECTED_SLICES_DIR.iterdir()):
        if not d.is_dir():
            continue
        idx = sorted(int(f.stem) for f in d.glob("*.png") if f.stem.isdigit())
        if idx:
            out[d.name] = idx
    return out


def old_series_for(idx: pd.DataFrame, folder: str) -> pd.DataFrame | None:
    """Eski notebook'un `os.listdir(patient)[0]` ile seçtiği seriyi yeniden kur:
    Kaynak kökünde hasta klasörünün alfabetik ilk alt dizini."""
    g = idx[(idx["patient_folder"] == folder) & (idx["root_id"].isin(DESKTOP_ROOT_IDS))]
    if g.empty:
        return None
    # hasta klasörüne göre göreli ilk alt dizin
    rel_first = {}
    for p in g["file_path"]:
        parts = Path(p).parts
        try:
            i = parts.index(folder)
        except ValueError:
            continue
        if i + 1 < len(parts):
            rel_first[p] = parts[i + 1]
    g = g.assign(subdir=g["file_path"].map(rel_first))
    first = sorted(x for x in g["subdir"].dropna().unique())
    if not first:
        return None
    return g[g["subdir"] == first[0]]


def main() -> int:
    ensure_dirs()
    selected = load_selected()
    idx = pd.read_parquet(PRIVATE / "dicom_index.parquet")
    idx = idx[idx["sop_class_uid"] == "1.2.840.10008.5.1.4.1.1.2"]
    cases = pd.read_csv(PRIVATE / "cases.csv")
    sel_series = pd.read_csv(PRIVATE / "series_selection.csv")
    series_idx = pd.read_csv(PRIVATE / "series_index_full.csv")

    all_idx = [j for v in selected.values() for j in v]
    print(f"seçilmiş kesit klasörü : {len(selected)} hasta, {len(all_idx)} kesit")
    print(f"indeks aralığı         : {min(all_idx)} – {max(all_idx)}")

    # ------------------------------------------------ hipotez testi
    rows = []
    for folder, idxs in selected.items():
        k = name_key(folder)
        old = old_series_for(idx, folder)
        n_old_slices = int(old["sop_instance_uid"].nunique()) if old is not None else 0
        old_rows = int(old["rows"].median()) if old is not None and old["rows"].notna().any() else 512
        cor = sel_series[(sel_series["role"] == "coronal_native")]
        cor_n = 0
        c = cases[cases["name_key"] == k]
        if len(c):
            cc = cor[cor["case_id"] == c.iloc[0]["case_id"]]
            cor_n = int(cc["n_slices"].iloc[0]) if len(cc) else 0
        rows.append({
            "folder": folder, "name_key": k, "n_selected": len(idxs),
            "min_idx": min(idxs), "max_idx": max(idxs),
            "n_old_slices": n_old_slices, "old_rows": old_rows, "n_coronal_native": cor_n,
            "h1_ok": max(idxs) < old_rows,
            "h2_ok": n_old_slices > 0 and max(idxs) < n_old_slices,
            "h3_ok": cor_n > 0 and max(idxs) < cor_n,
        })
    ht = pd.DataFrame(rows)
    n = len(ht)
    posterior = sum(1 for j in all_idx if j > 256) / len(all_idx)
    print("\n=== hipotez testi ===")
    print(f"  H1 AXIAL_ROW            : {int(ht['h1_ok'].sum())}/{n} hasta uyumlu "
          f"(indeks < satır sayısı)")
    print(f"     posterior oranı (j>256): {posterior:.3f}   (böbrek retroperitoneal, >=0.80 beklenir)")
    print(f"  H2 AXIAL_SLICE          : {int(ht['h2_ok'].sum())}/{n} hasta uyumlu")
    print(f"  H3 CORONAL_NATIVE_SLICE : {int(ht['h3_ok'].sum())}/{n} hasta uyumlu "
          f"({int((ht['n_coronal_native'] > 0).sum())} hastada koronal reformat var)")

    # Karar eleme ile verilir: H2 ve H3 hastaların neredeyse tamamında düşer
    # (indeks kesit sayısını aşıyor), H1 ise hepsinde tutar. Posterior oranı
    # yalnızca destekleyici bilgidir, ölçüt değildir — asıl kanıt 07b'deki
    # piksel korelasyon testidir (eski koronal PNG ile tepe j'de, p<1e-30).
    h1 = bool(ht["h1_ok"].all())
    h2 = bool(ht["h2_ok"].mean() > 0.5)
    h3 = bool(ht["h3_ok"].mean() > 0.5)
    semantics = "AXIAL_ROW" if (h1 and not h2 and not h3) else ("AXIAL_SLICE" if h2 else "UNRESOLVED")
    print(f"\n  KAZANAN: {semantics}  (H1 tutuyor={h1}, H2 tutuyor={h2}, H3 tutuyor={h3})")
    print("  Kesin doğrulama için: scripts/07b_verify_index_semantics.py")

    # ------------------------------------------------ haritalama
    prim = sel_series[sel_series["role"] == "axial_std"].set_index("case_id")
    smeta = series_idx.set_index("series_instance_uid")
    out: dict[str, dict] = {}
    map_rows = []
    for folder, idxs in selected.items():
        k = name_key(folder)
        c = cases[cases["name_key"] == k]
        if c.empty:
            continue
        old = old_series_for(idx, folder)
        if old is None or old.empty:
            continue
        ipp_old_y = float(old["ipp_y"].median())
        ps_old = float(old["pixel_spacing_r"].median())
        old_rows = int(old["rows"].median())

        for _, cc in c.iterrows():
            case_id = cc["case_id"]
            if case_id not in prim.index:
                continue
            p = prim.loc[case_id]
            sm = smeta.loc[p["series_instance_uid"]]
            ipp_new_y = float(sm["ipp_y"]) if "ipp_y" in sm else None
            if ipp_new_y is None:
                g = idx[idx["series_instance_uid"] == p["series_instance_uid"]]
                ipp_new_y = float(g["ipp_y"].median())
            ps_new = float(p["pixel_spacing_r"])

            items = []
            for j in idxs:
                y_mm = ipp_old_y + j * ps_old
                j_new = (y_mm - ipp_new_y) / ps_new
                inside = 0 <= j_new < 512
                items.append({
                    "j_old": int(j),
                    "row_fraction": round(j / old_rows, 4),
                    "y_mm": round(y_mm, 3),
                    "j_new_axial_std": int(round(j_new)) if inside else None,
                    "inside_fov": bool(inside),
                })
                map_rows.append({
                    "case_id": case_id, "j_old": j, "y_mm": y_mm,
                    "j_new": j_new, "inside_fov": inside,
                    "same_frame": abs(ps_old - ps_new) < 1e-6 and abs(ipp_old_y - ipp_new_y) < 1e-6,
                })
            out[case_id] = {
                "source": "Nihai Dataset/eski",
                "source_folder": folder,
                "n": len(items),
                "index_semantics": semantics,
                "mapping_confidence": "high" if all(i["inside_fov"] for i in items) else "low",
                "old_series_rows": old_rows,
                "old_pixel_spacing_mm": ps_old,
                "old_ipp_y": ipp_old_y,
                "items": items,
            }

    # source_folder gerçek hasta klasör adıdır -> tam sürüm 00_private'ta,
    # yayınlanabilir sürümde bu alan çıkarılır.
    (PRIVATE / "selected_slices_full.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    public = {
        c: {k: v for k, v in d.items() if k != "source_folder"}
        for c, d in out.items()
    }
    (LABELS / "selected_slices.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=1), encoding="utf-8")
    mp = pd.DataFrame(map_rows)
    mp.to_csv(LABELS / "selected_slices_map.csv", index=False, encoding="utf-8")

    print("\n=== haritalama ===")
    print(f"  haritalanan vaka : {len(out)}")
    print(f"  haritalanan kesit: {len(mp)}")
    print(f"  FOV içinde       : {int(mp['inside_fov'].sum())}/{len(mp)} "
          f"({mp['inside_fov'].mean():.3f})")
    print(f"  aynı çerçeve (eski seri = yeni seri geometrisi): {int(mp['same_frame'].sum())}/{len(mp)}")
    print(f"  j_new sapması |j_new - j_old| medyan: {(mp['j_new'] - mp['j_old']).abs().median():.2f} piksel")
    low = [c for c, v in out.items() if v["mapping_confidence"] == "low"]
    print(f"  düşük güvenli vaka: {len(low)} {low[:10]}")

    labeled_cases = set(cases[cases["group"] == "labeled"]["case_id"])
    cov = len(labeled_cases & set(out))
    print(f"  etiketli vakalardan kapsanan: {cov}/{len(labeled_cases)}")

    gate = semantics == "AXIAL_ROW" and mp["inside_fov"].all()
    print(f"\nKAPI: semantik çözüldü + tüm kesitler FOV içinde -> {'PASS' if gate else 'FAIL'}")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
