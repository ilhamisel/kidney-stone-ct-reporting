"""Arşiv tamamlandıktan sonra kalan adımları sırayla çalıştırır.

  05  8-bit türevler (stone + mw3)
  07c seçilmiş satır doğrulaması (tam kohort)
  14  iletişim sayfaları
  13  QA (HU gidiş-dönüşü dahil, tam)
  15  yayınlanabilir ağaç PHI denetimi
"""
from __future__ import annotations

import subprocess
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Aşamaları çalıştıran yorumlayıcı. Varsayılan: bu scripti çalıştıran
# yorumlayıcı; KIDNEYCT_PYTHON ile başka bir ortam gösterilebilir.
PY = Path(os.environ.get("KIDNEYCT_PYTHON", sys.executable))

STEPS = [
    ("05 türevler", ["05_build_derivatives.py", "--kinds", "stone,mw3"]),
    ("07c satır doğrulaması", ["07c_validate_selected_rows.py"]),
    ("14 iletişim sayfaları", ["14_contact_sheets.py"]),
    ("13 QA (tam)", ["13_qa_checks.py", "--full-cases", "20", "--sample-slices", "10"]),
    ("15 PHI denetimi", ["15_export_public.py", "--mode", "audit"]),
]


def main() -> int:
    results = []
    for name, args in STEPS:
        print(f"\n{'=' * 70}\n### {name}\n{'=' * 70}", flush=True)
        t0 = time.time()
        p = subprocess.run([str(PY), str(ROOT / "scripts" / args[0]), *args[1:]])
        results.append((name, p.returncode, time.time() - t0))
        print(f"--- {name}: çıkış={p.returncode}  {time.time() - t0:.0f}s", flush=True)

    print(f"\n{'=' * 70}\nÖZET")
    for name, rc, el in results:
        print(f"  {'OK  ' if rc == 0 else 'HATA'}  {name:28s} {el:7.0f}s  (çıkış={rc})")
    return 0 if all(rc == 0 for _, rc, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
