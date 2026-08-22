"""Aşama 14 — radyolog incelemesi için vaka başına iletişim sayfası.

Her sayfa:
  üst   : taş penceresinde eşit aralıklı 12 aksiyel kesit
  orta  : radyoloğun seçtiği her satırda (haritalanmış) koronal reformat,
          raporda bildirilen taraf/zon işaretli
  alt   : üretilen Türkçe SONUÇ + yapılandırılmış olgular

Amaç: etiket ile görüntünün gerçekten örtüştüğünü göz ile doğrulamak.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common.hu import WINDOWS, window8
from common.paths import ARCHIVE, LABELS, QA

TILE = 224
GRID = (4, 3)  # sütun x satır -> 12 kesit
FONT_CANDIDATES = [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"]


def get_font(size: int) -> ImageFont.FreeTypeFont:
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def load_volume(case_dir: Path) -> tuple[np.ndarray, dict]:
    meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    n = meta["n_slices"]
    vol = np.zeros((512, 512, n), dtype=np.int16)
    for i in range(n):
        png = cv2.imread(str(case_dir / f"{i:04d}.png"), cv2.IMREAD_UNCHANGED)
        vol[:, :, i] = (png.astype(np.int32) - meta["hu_offset"]).astype(np.int16)
    return vol, meta


HI_HU = 250  # taş adayı eşiği (kalsiyum yoğunluğu); kemik de bu eşiğin üstündedir


def coronal_at(vol: np.ndarray, row: int, aspect_dz: float, ps: float) -> tuple[np.ndarray, np.ndarray]:
    """Satır `row`'daki koronal düzlem; kare piksel olacak şekilde ölçeklenir.
    İkinci çıktı: >=HI_HU vokselleri (görsel doğrulama için maske)."""
    plane = np.flipud(vol[row, :, :].T)  # (n_slices, 512)
    img = window8(plane, *WINDOWS["stone"])
    hi = (plane >= HI_HU).astype(np.uint8) * 255
    h = int(round(img.shape[0] * aspect_dz / ps))
    h = max(h, 32)
    img = cv2.resize(img, (img.shape[1], h), interpolation=cv2.INTER_AREA)
    hi = cv2.resize(hi, (hi.shape[1], h), interpolation=cv2.INTER_NEAREST)
    return img, hi


def build_sheet(case_id: str, doc: dict) -> Image.Image | None:
    d = ARCHIVE / case_id / "axial_std"
    if not (d / "meta.json").exists():
        return None
    vol, meta = load_volume(d)
    n = meta["n_slices"]
    ps = meta["pixel_spacing_mm"][0]
    dz = meta["median_dz_mm"]

    cols, rows = GRID
    sheet_w = cols * TILE
    top_h = rows * TILE

    # --- aksiyel ızgara
    idxs = np.linspace(int(0.15 * n), int(0.85 * n), cols * rows).astype(int)
    axial = Image.new("RGB", (sheet_w, top_h), "black")
    dr = ImageDraw.Draw(axial)
    f_small = get_font(13)
    for k, i in enumerate(idxs):
        img = window8(vol[:, :, i], *WINDOWS["stone"])
        tile = Image.fromarray(cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA)).convert("RGB")
        x, y = (k % cols) * TILE, (k // cols) * TILE
        axial.paste(tile, (x, y))
        dr.text((x + 4, y + 3), f"z{i}", fill=(255, 220, 0), font=f_small)

    # --- seçilmiş satırlarda koronal
    sel = doc.get("selected_slices") or {}
    items = [it for it in sel.get("items", []) if it.get("j_new_axial_std") is not None][:4]
    cor_imgs = []
    for it in items:
        row = int(it["j_new_axial_std"])
        cor, hi = coronal_at(vol, row, dz, ps)
        cor_rgb = cv2.cvtColor(cor, cv2.COLOR_GRAY2BGR)
        # GERÇEK kanıt: >=250 HU vokselleri kırmızıya boya. Bildirilen zon'a
        # sabit koordinatta daire çizmek olurdu ki bu tam olarak önceki
        # çalışmanın uydurma "dikkat haritası" hatasıdır — yapılmıyor.
        cor_rgb[hi > 0] = (0, 0, 255)
        cor_imgs.append((row, it["j_old"], Image.fromarray(cor_rgb[:, :, ::-1])))

    cor_h = max((im.height for _, _, im in cor_imgs), default=0)
    cor_w_total = sum(im.width for _, _, im in cor_imgs) + 8 * max(len(cor_imgs) - 1, 0)
    scale = min(1.0, sheet_w / cor_w_total) if cor_w_total else 1.0
    cor_h = int(cor_h * scale)

    cap_h = 150
    sheet = Image.new("RGB", (sheet_w, top_h + cor_h + cap_h + 12), (18, 18, 18))
    sheet.paste(axial, (0, 0))
    x = 0
    dr_c = ImageDraw.Draw(sheet)
    for row, j_old, im in cor_imgs:
        im2 = im.resize((int(im.width * scale), int(im.height * scale)))
        sheet.paste(im2, (x, top_h + 4))
        dr_c.text((x + 4, top_h + 8), f"koronal satır {row} (j_old={j_old})",
                  fill=(255, 220, 0), font=f_small)
        # radyolojik konvansiyon: hastanın SAĞI görüntünün SOLUNDA
        dr_c.text((x + 4, top_h + 4 + im2.height - 18), "SAĞ", fill=(120, 200, 255), font=f_small)
        dr_c.text((x + im2.width - 34, top_h + 4 + im2.height - 18), "SOL",
                  fill=(120, 200, 255), font=f_small)
        x += im2.width + 8

    # --- altyazı
    dr = ImageDraw.Draw(sheet)
    f = get_font(15)
    fb = get_font(17)
    t = doc["targets"]
    stones = "; ".join(
        f"{s['side'][:1]}{'-' if s['side'] else ''}{s['zone']} {s['size_mm']}mm ({s['size_class']})"
        for s in doc["labels"]["stones"]
    )
    y0 = top_h + cor_h + 10
    dr.text((8, y0), f"{doc['case_id']}  |  {t['laterality']}  |  "
                     f"{doc['demographics']['sex']} {doc['demographics']['age_years']}y  |  "
                     f"n={t['n_stones_effective']} ({doc['labels']['count_qualifier']})",
            fill=(255, 255, 255), font=fb)
    dr.text((8, y0 + 24), f"Taşlar: {stones[:150]}", fill=(200, 220, 255), font=f)
    imp = doc["reports"]["tr"]["impression"]
    dr.text((8, y0 + 48), f"SONUÇ: {imp[:110]}", fill=(180, 255, 180), font=f)
    if len(imp) > 110:
        dr.text((8, y0 + 68), imp[110:220], fill=(180, 255, 180), font=f)
    dr.text((8, y0 + 130), f"kırmızı = >={HI_HU} HU voksel (kemik dahil, ham kanıt; "
                           f"bildirilen zona daire ÇİZİLMEZ)", fill=(150, 150, 150), font=f_small)
    anom = doc["labels"]["anomalies"]
    if anom:
        dr.text((8, y0 + 92), f"Anomali: {', '.join(anom)}", fill=(255, 180, 120), font=f)
    if doc["labels"]["flags"].get("label_ambiguous"):
        dr.text((8, y0 + 112), "! ETİKET BELİRSİZ (tetkik ataması doğrulanmamış)",
                fill=(255, 120, 120), font=f)
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="0 = tüm vakalar")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out = QA / "contact_sheets"
    out.mkdir(parents=True, exist_ok=True)
    docs = sorted((LABELS / "reports").glob("*.json"))
    cases = [p.stem for p in docs]
    built = [c for c in cases if (ARCHIVE / c / "axial_std" / "meta.json").exists()]
    if args.sample:
        built = random.Random(args.seed).sample(built, min(args.sample, len(built)))
    print(f"{len(built)} vaka için iletişim sayfası")

    n_ok = 0
    for i, case_id in enumerate(sorted(built), 1):
        doc = json.loads((LABELS / "reports" / f"{case_id}.json").read_text(encoding="utf-8"))
        sheet = build_sheet(case_id, doc)
        if sheet is None:
            continue
        sheet.save(out / f"{case_id}.png", optimize=True)
        n_ok += 1
        if i % 20 == 0:
            print(f"  {i}/{len(built)}")
    print(f"{n_ok} sayfa -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
