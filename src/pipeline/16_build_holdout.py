"""Aşama 16 — etiketsiz holdout kohortu (54 üreter-only + 4 ekstra).

Bu hastaların Excel'de böbrek taşı etiketi YOK. Kullanıcı kararı gereği
etiketlenmiyor, ayrı tutuluyor. İleride radyolog "böbrekte taş yok" teyidi
verirse negatif sınıf olarak devreye alınabilir — bu, kohortun tespit
(detection) veri setine dönüşmesi için gereken tek eksik parçadır.

Bu script yalnızca arşiv/türev üretimini holdout grubu için tetikler ve
kapsam raporu yazar; etiket üretmez.
"""
from __future__ import annotations

import subprocess
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.paths import HOLDOUT, LOGS, PRIVATE, ROOT

# Aşamaları çalıştıran yorumlayıcı. Varsayılan: bu scripti çalıştıran
# yorumlayıcı; KIDNEYCT_PYTHON ile başka bir ortam gösterilebilir.
PY = Path(os.environ.get("KIDNEYCT_PYTHON", sys.executable))


def main() -> int:
    phi = pd.read_csv(PRIVATE / "phi_map.csv")
    hol = phi[phi["group"] == "holdout_unlabeled"]
    cases = pd.read_csv(PRIVATE / "cases.csv")
    hc = cases[cases["group"] == "holdout_unlabeled"]
    sel = pd.read_csv(PRIVATE / "series_selection.csv")
    hs = sel[sel["group"] == "holdout_unlabeled"]

    print(f"holdout hasta : {len(hol)}")
    print(f"holdout vaka  : {len(hc)}")
    print(f"koleksiyon    : {hc['collection'].value_counts().to_dict()}")
    print(f"axial_std olan: {int((hs['role'] == 'axial_std').sum())}")

    HOLDOUT.mkdir(parents=True, exist_ok=True)
    rep = hol[["patient_id", "n_ct_files", "n_series", "collections", "has_usable_imaging"]]
    rep.to_csv(HOLDOUT / "holdout_coverage.csv", index=False, encoding="utf-8")

    (HOLDOUT / "README.md").write_text(
        "# Holdout — etiketsiz kohort\n\n"
        f"{len(hol)} hasta / {len(hc)} vaka. Excel'de böbrek taşı etiketi yok.\n\n"
        "- 54 hasta `ÜRETER TAŞI` koleksiyonundan: üreter taşı nedeniyle çekilmiş, "
        "böbrek taşı durumu **bilinmiyor**.\n"
        "- 4 ekstra hasta (ör. 'hiperdens piramis', 'komplike kist') taş **taklitçisi** "
        "olarak değerli zor negatif adaylarıdır.\n\n"
        "## Neden ayrı tutuluyor\n"
        "Etiketli kohortun 197/197 vakasında taş var; negatif sınıf yok. Bu yüzden "
        "veri seti şu an bir **karakterizasyon** veri setidir, tespit değil. "
        "Radyolog bu kohort için 'böbrekte taş yok' teyidi verirse negatif sınıf "
        "kurulabilir ve tespit görevi geçerli hale gelir.\n\n"
        "## Üretim\n"
        "```\n"
        "python scripts/04_build_archive_png.py --group holdout_unlabeled --roles axial_std\n"
        "```\n",
        encoding="utf-8",
    )

    if "--build" in sys.argv:
        print("\narşiv üretiliyor (holdout)...")
        subprocess.run([str(PY), str(ROOT / "scripts" / "04_build_archive_png.py"),
                        "--group", "holdout_unlabeled", "--roles", "axial_std"], check=False)
    else:
        print("\nkapsam raporu yazıldı. Arşiv üretimi için: --build")
    print(f"-> {HOLDOUT / 'holdout_coverage.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
