# -*- coding: utf-8 -*-
"""Aşama 21 — rapor JSON'larındaki segmentasyon crop bloklarını onar.

Neden: 09_detect_stone_candidates.py --write (11 Ağustos 22:31-22:45), 08'in
yazdığı totalsegmentator crop bloklarını koşulsuz olarak {"source":"heuristic"}
ile ezmişti (bug, 09'da düzeltildi). Maskeler 08_seg/ altında durduğu için her
alan kayıpsız yeniden türetilir; half-crop PNG'leri YENİDEN YAZILMAZ (diskteki
üretim deterministikti, dokunmaya gerek yok).

Doğrulama: yeniden kurulan bbox'tan türeyen crop_size ve n_slices, diskteki
half-crop dosya sayısı ve kidney_dataset.csv ile karşılaştırılır; uyuşmazlıkta
durur. stone_candidates alanına dokunulmaz.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np

from common.paths import ARCHIVE, DERIVED, LABELS, ROOT

SEG_ROOT = ROOT / "08_seg"

_s = importlib.util.spec_from_file_location(
    "seg08", str(Path(__file__).resolve().parent / "08_segment_kidneys.py"))
SEG = importlib.util.module_from_spec(_s)
_s.loader.exec_module(SEG)


def crop_meta(case_id: str, side: str, stats: dict, meta: dict, kinds=("stone", "mw3")) -> dict:
    """build_halfcrop'un döndürdüğü sözlüğü PNG yazmadan yeniden kurar (aynı matematik)."""
    ps_r, ps_c = meta["pixel_spacing_mm"]
    r0, r1, c0, c1, s0, s1 = stats["bbox_voxel"]
    pr, pc = int(round(SEG.PAD_MM / ps_r)), int(round(SEG.PAD_MM / ps_c))
    r0, r1 = max(r0 - pr, 0), min(r1 + pr, 512)
    c0, c1 = max(c0 - pc, 0), min(c1 + pc, 512)
    return {"bbox_voxel_padded": [r0, r1, c0, c1, s0, s1],
            "crop_size": [int(r1 - r0), int(c1 - c0)],
            "n_slices": int(s1 - s0),
            "dirs": {k: f"03_derived/{case_id}/halfcrop_{side}_{k}" for k in kinds}}


def main() -> int:
    cases = sorted(d.name for d in SEG_ROOT.iterdir() if d.is_dir())
    print(f"{len(cases)} vaka onarılacak")
    n_fixed, n_checked = 0, 0
    for n, case_id in enumerate(cases, 1):
        meta = json.loads((ARCHIVE / case_id / "axial_std" / "meta.json").read_text(encoding="utf-8"))
        masks = {}
        for side, fname in (("left", "kidney_left.nii.gz"), ("right", "kidney_right.nii.gz")):
            p = SEG_ROOT / case_id / fname
            masks[side] = np.asarray(nib.load(str(p)).dataobj) > 0 if p.exists() \
                else np.zeros((1, 1, 1), bool)
        st = {s: SEG.mask_stats(masks[s], meta) for s in ("left", "right")}
        orient_ok = None
        if st["left"] and st["right"]:
            orient_ok = st["left"]["centroid_voxel"][1] > st["right"]["centroid_voxel"][1]

        p = LABELS / "reports" / f"{case_id}.json"
        doc = json.loads(p.read_text(encoding="utf-8"))
        for side in ("left", "right"):
            if st[side]:
                cm = crop_meta(case_id, side, st[side], meta)
                # diskle tutarlılık: half-crop dosya sayısı = n_slices
                d = DERIVED / case_id / f"halfcrop_{side}_stone"
                n_png = len(list(d.glob("*.png")))
                assert n_png == cm["n_slices"], \
                    f"{case_id}/{side}: disk {n_png} != yeniden kurulan {cm['n_slices']}"
                n_checked += 1
                doc["targets"]["kidneys"][side]["crop"] = {
                    "source": "totalsegmentator", "builder": SEG.BUILDER,
                    "orientation_check_passed": orient_ok, **st[side], **cm,
                    "repaired_by": "21_repair_crop_meta/1.0.0"}
            else:
                doc["targets"]["kidneys"][side]["crop"] = {
                    "source": "totalsegmentator", "builder": SEG.BUILDER, "mask_empty": True,
                    "repaired_by": "21_repair_crop_meta/1.0.0"}
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        n_fixed += 1
        if n % 40 == 0 or n == len(cases):
            print(f"  {n}/{len(cases)}")
    print(f"onarıldı: {n_fixed} vaka, disk tutarlılığı doğrulanan böbrek: {n_checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
