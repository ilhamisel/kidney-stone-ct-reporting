# -*- coding: utf-8 -*-
"""common/ml birim testleri. Bağımsız assert'ler; pytest gerektirmez.

Çalıştırma: PYTHONPATH=src python tests/test_ml_units.py
Kohort tablosu yoksa testler temiz biçimde atlanır.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from common.ml import data as D
from common.ml import metrics as MET
from common.ml import models as M
from common.paths import LABELS
from common.ml.transforms import letterbox


def test_lse_single_instance():
    """Tek kesitlik çantada LSE çıktısı = o kesidin logiti."""
    z = torch.randn(3, 1, 4)
    out = M.LSEPool()(z)
    assert torch.allclose(out, z[:, 0, :], atol=1e-6), "LSE tek-kesit kimliği bozuk"


def test_numpy_lse_matches_torch():
    z = torch.randn(37, 3)
    a = M.numpy_lse(z.numpy())
    b = M.LSEPool()(z.unsqueeze(0))[0].numpy()
    assert np.allclose(a, b, atol=1e-5), "numpy_lse != LSEPool"


def test_ordinal_roundtrip():
    y = torch.tensor([0, 1, 2, 2, 0, 1])
    tgt = M.ordinal_targets(y)
    logits = (tgt * 2 - 1) * 5.0  # hedefe kesin uyan logit
    cls, probs = M.ordinal_to_class(logits)
    assert torch.equal(cls, y), f"ordinal gidiş-dönüş: {cls} != {y}"
    assert torch.allclose(probs.sum(1), torch.ones(len(y)), atol=1e-5)


def test_cohort_one_exam_per_patient():
    """Modelleme kohortu: hasta başına tek tetkik (sözde-yineleme yok)."""
    df = D.load_table()
    assert len(df) == D.N_KIDNEYS and df["patient_id"].nunique() == D.N_PATIENTS
    assert df.groupby("patient_id")["case_id"].nunique().max() == 1
    full = D.load_table(all_exams=True)
    assert len(full) == 394, "tam kohort betimleme için erişilebilir olmalı"


def test_fold_leakage():
    """5 tohum x 5 kat: hasta çakışması yok, bir hastanın iki böbreği aynı bölmede."""
    df = D.load_table()
    for seed in D.SEEDS:
        col = f"cv5_seed{seed}"
        per_patient = df.groupby("patient_id")[col].nunique()
        assert (per_patient == 1).all(), f"tohum {seed}: hasta iki kata bölünmüş"
        for fold in range(D.N_FOLDS):
            tr, va, te = D.fold_split(df, seed, fold)  # assert'ler içeride
            assert len(tr) + len(va) + len(te) == D.N_KIDNEYS


def test_task_subsets():
    df = D.load_table()
    assert len(D.task_subset(df, "has_stone")) == D.N_KIDNEYS
    sz = D.task_subset(df, "size3")
    assert sz["has_stone"].all() and sz["size3"].notna().all()
    zn = D.task_subset(df, "zone")
    assert zn[list(D.ZONE_COLS)].any(axis=1).all()
    y = D.targets_of(df, "has_stone")
    assert (y.sum(), len(y)) == (246, 390), f"has_stone dağılımı beklenmedik: {y.sum()}/{len(y)}"


def test_letterbox_aspect():
    img = np.zeros((60, 100, 3), dtype=np.uint8)
    img[:, :, :] = 200
    out = letterbox(img)
    assert out.shape == (224, 224, 3)
    rows = np.where(out[:, 112, 0] > 0)[0]  # içerik dikeyde 60/100 oranında dolu
    ratio = len(rows) / 224
    assert abs(ratio - 0.6) < 0.05, f"letterbox oran bozuldu: {ratio:.3f}"
    assert out[0, 112, 0] == 0 and out[223, 112, 0] == 0, "dolgu sıfır değil"


def test_sample_indices():
    torch.manual_seed(0)
    for _ in range(20):
        idx = D.sample_indices(50, 16)
        v = idx.numpy()
        assert len(v) == 16 and v.min() >= 0 and v.max() < 50
        assert (np.diff(v) > 0).all(), "katmanlı örnek sıralı/tekil değil"
    v = D.sample_indices(5, 16).numpy()
    assert len(v) == 16 and v.min() == 0 and v.max() == 4, "n<K kapsaması bozuk"


def test_norm_positions():
    p = D.norm_positions(torch.tensor([0, 5, 10]), 11).numpy()
    assert np.allclose(p, [0.0, 0.5, 1.0]), f"z-konum normalizasyonu bozuk: {p}"
    assert float(D.norm_positions(torch.tensor([0]), 1)[0]) == 0.5, "tek kesitte 0.5 olmalı"


def test_milwrapper_pos():
    """use_pos başlığı konumu görmeli; konum değişince logit değişmeli."""
    enc = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 16))
    m = M.MILWrapper(enc, 16, 3, use_pos=True).eval()
    x = torch.randn(4, 3, 8, 8)
    with torch.no_grad():
        z0 = m.instance_logits(x, torch.zeros(4))
        z1 = m.instance_logits(x, torch.ones(4))
        bag = m(x, 4, torch.zeros(4))
    assert not torch.allclose(z0, z1), "konum logite etki etmiyor"
    assert bag.shape == (1, 3)


def test_mirror_involution():
    x = torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8)
    assert torch.equal(torch.flip(torch.flip(x, dims=[-1]), dims=[-1]), x)


def test_binary_metrics_sanity():
    y = np.array([0, 0, 1, 1, 1, 0])
    m = MET.binary_metrics(y, y.astype(float))
    assert m["auc"] == 1.0 and m["acc"] == 1.0
    mc = MET.mean_ci([0.7, 0.8, 0.9])
    assert abs(mc[0] - 0.8) < 1e-9 and mc[1] < 0.8 < mc[2]


def main() -> int:
    # Bu testler bölme bütünlüğünü GERÇEK kohort tablosu üzerinde doğrular (hasta
    # sızıntısı, kat dengeleri). Tablo hasta verisi olduğu için depoda yoktur;
    # yoksa test çöküp yığın izi basmak yerine temiz biçimde atlanır.
    table = LABELS / "kidney_dataset.csv"
    if not table.exists():
        print(f"ATLANDI: kohort tablosu yok -> {table}")
        print("Bu testler hasta verisi gerektirir; bkz. docs/data-availability.md.")
        return 0

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  OK  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} test geçti")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
