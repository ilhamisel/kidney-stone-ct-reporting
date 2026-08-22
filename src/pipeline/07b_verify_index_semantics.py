"""Aşama 7b — seçilmiş kesit indeksinin anlamını DOĞRUDAN kanıtla.

Anatomik varsayım (posterior oranı) zayıf bir kanıttı. Bunun yerine eski koronal
PNG'ler (böbrek_taşı_full/<HASTA>/<j>.png, 512 adet) ile arşivden yeniden
ürettiğimiz koronal düzlemler karşılaştırılır:

  H1 doğruysa, eski PNG j ile bizim vol[j, :, :] düzlemimiz arasındaki
  normalize çapraz korelasyon j ofsetinde tepe yapar.

Bu test anatomi hakkında hiçbir varsayım içermez; tamamen piksel kanıtıdır.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

from common.hu import png16_to_hu, window8
from common.paths import ARCHIVE, LABELS, LEGACY_CORONAL_DIRS, LOGS, PRIVATE, SELECTED_SLICES_DIR

SEARCH = 40  # ±satır arama penceresi


def load_volume(case_dir: Path) -> tuple[np.ndarray, dict]:
    meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    n = meta["n_slices"]
    vol = np.zeros((512, 512, n), dtype=np.int16)
    for i in range(n):
        png = cv2.imread(str(case_dir / f"{i:04d}.png"), cv2.IMREAD_UNCHANGED)
        vol[:, :, i] = png16_to_hu(png).astype(np.int16)
    return vol, meta


def coronal_plane(vol: np.ndarray, j: int) -> np.ndarray:
    """Eski kodun ürettiği yönelim: np.flipud(img3d[j, :, :].T)."""
    return np.flipud(vol[j, :, :].T)


def imread_unicode(path: Path, flags: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray | None:
    """cv2.imread Windows'ta ASCII olmayan yolları açamıyor (Türkçe hasta adları).
    Dosyayı Python ile okuyup bellekten çöz."""
    try:
        buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, flags)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    da, db = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (da * db)) if da > 0 and db > 0 else 0.0


def legacy_path(folder: str, j: int) -> Path | None:
    """Referans görüntü: radyoloğun seçtiği koronal PNG'nin kendisi.

    `*_full` klasörleri sonradan filtrelendiği için (filter_report.csv: hasta
    başına 412 dosya taşındı) orada j bulunamıyor; `Nihai Dataset/eski` ise
    seçilen kesitlerin görüntülerini aynı üretimden korumuş durumda.
    """
    p = SELECTED_SLICES_DIR / folder / f"{j}.png"
    if p.exists():
        return p
    for d in LEGACY_CORONAL_DIRS.values():
        q = d / folder / f"{j}.png"
        if q.exists():
            return q
    return None


def main() -> int:
    sel = json.loads((LABELS / "selected_slices.json").read_text(encoding="utf-8"))
    phi = pd.read_csv(PRIVATE / "phi_map.csv")
    key2folder = {}
    for _, r in phi.iterrows():
        for f in str(r["folder_names"]).split("|"):
            if f:
                key2folder.setdefault(r["name_key"], []).append(f)

    built = {d.name for d in ARCHIVE.iterdir() if d.is_dir() and (d / "axial_std" / "meta.json").exists()}
    todo = [c for c in sel if c in built]
    print(f"arşivi hazır ve seçilmiş kesidi olan vaka: {len(todo)}")

    rows = []
    for case_id in sorted(todo):
        info = sel[case_id]
        folder = info["source_folder"]
        vol, meta = load_volume(ARCHIVE / case_id / "axial_std")
        for item in info["items"]:
            j_old = item["j_old"]
            lp = legacy_path(folder, j_old)
            if lp is None:
                continue
            leg = imread_unicode(lp)
            if leg is None:
                continue
            best_off, best_score, scores = None, -2.0, {}
            for off in range(-SEARCH, SEARCH + 1):
                j = j_old + off
                if not (0 <= j < 512):
                    continue
                plane = coronal_plane(vol, j)
                # eski hat DICOM başlığındaki yumuşak doku penceresini kullanıyordu
                img = window8(plane, 40, 400)
                img = cv2.resize(img, (leg.shape[1], leg.shape[0]), interpolation=cv2.INTER_AREA)
                s = ncc(img, leg)
                scores[off] = s
                if s > best_score:
                    best_score, best_off = s, off
            if best_off is None:
                continue
            rows.append({
                "case_id": case_id, "folder": folder, "j_old": j_old,
                "best_offset": best_off, "best_ncc": round(best_score, 4),
                "ncc_at_0": round(scores.get(0, float("nan")), 4),
                "margin": round(best_score - sorted(scores.values())[-2], 4) if len(scores) > 1 else None,
            })
        print(f"  {case_id}: {len([r for r in rows if r['case_id'] == case_id])} kesit test edildi")

    df = pd.DataFrame(rows)
    if df.empty:
        print("test edilebilecek kesit bulunamadı")
        return 1
    out = LOGS / "07b_index_semantics.csv"
    df.to_csv(out, index=False, encoding="utf-8")

    from scipy.stats import binomtest

    n = len(df)
    exact = int((df["best_offset"] == 0).sum())
    near2 = int((df["best_offset"].abs() <= 2).sum())
    # Boş hipotez: indeksin satırla ilgisi yok -> tepe ±SEARCH penceresinde düzgün dağılır
    n_window = 2 * SEARCH + 1
    p_null = 5 / n_window  # |ofset|<=2 -> 5 konum
    bt = binomtest(near2, n, p_null, alternative="greater")

    print("\n=== indeks semantiği doğrulaması (piksel kanıtı) ===")
    print(f"  test edilen kesit        : {n}")
    print(f"  tepe tam j'de (ofset=0)  : {exact}/{n} ({exact / n:.3f})")
    print(f"  tepe |ofset|<=2          : {near2}/{n} ({near2 / n:.3f})   şans beklentisi {p_null:.3f}")
    print(f"  binom testi p            : {bt.pvalue:.3e}")
    print(f"  ofset medyanı / |ofset| medyanı : {df['best_offset'].median():.1f} / "
          f"{df['best_offset'].abs().median():.1f}")
    print(f"  ofset dağılımı           : {df['best_offset'].value_counts().head(9).to_dict()}")
    print(f"  ofset aralığı            : [{df['best_offset'].min()}, {df['best_offset'].max()}]")
    print(f"  ofset=0'daki NCC medyanı : {df['ncc_at_0'].median():.4f}")
    print(f"  log -> {out}")
    print("\n  Not: eski PNG'ler matplotlib ile (tight bbox + aspect yeniden örnekleme)")
    print("  üretildiği için komşu satırlar neredeyse aynı korelasyonu verir; bu yüzden")
    print("  ölçüt 'tam 0' değil, tepenin j'nin birkaç satırı içinde kalmasıdır.")

    ok = bool(near2 / n >= 0.60 and bt.pvalue < 1e-6 and abs(df["best_offset"].median()) <= 2)
    print(f"\nSONUÇ: {'H1 (AXIAL_ROW) DOĞRULANDI' if ok else 'H1 doğrulanamadı'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
