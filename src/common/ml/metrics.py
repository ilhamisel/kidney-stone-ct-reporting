"""Metrics and statistics: kidney and patient level, 95% CI, paired tests.

Protocol: report the mean over runs with a 95% confidence interval, and compare
against the majority baseline with a PAIRED test — each (seed, fold) cell is
matched to its own baseline, because the folds differ in difficulty and an
unpaired comparison would absorb that variance into the error term.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def binary_metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    """y (N,) in {0,1}; prob (N,) probability of the positive class."""
    yhat = (prob >= 0.5).astype(int)
    acc = float((y == yhat).mean())
    lo, hi = wilson_ci(int((y == yhat).sum()), len(y))
    return {
        "n": len(y),
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
        "acc": acc, "acc_ci_lo": lo, "acc_ci_hi": hi,
        "bal_acc": float(balanced_accuracy_score(y, yhat)),
        "f1": float(f1_score(y, yhat, average="macro")),
        "majority": float(max(y.mean(), 1 - y.mean())),
        "degenerate": bool(max(yhat.mean(), 1 - yhat.mean()) >= 0.90),
    }


def ordinal_metrics(y: np.ndarray, probs: np.ndarray) -> dict:
    """y (N,) in {0,1,2}; probs (N,3)."""
    yhat = probs.argmax(1)
    counts = np.bincount(y, minlength=3)
    return {
        "n": len(y),
        "acc": float((y == yhat).mean()),
        "bal_acc": float(balanced_accuracy_score(y, yhat)),
        "f1": float(f1_score(y, yhat, average="macro")),
        "qwk": float(cohen_kappa_score(y, yhat, weights="quadratic")),
        "majority": float(counts.max() / counts.sum()),
        "degenerate": bool(np.bincount(yhat, minlength=3).max() / len(yhat) >= 0.90),
    }


def multilabel_metrics(y: np.ndarray, probs: np.ndarray) -> dict:
    """y, probs (N,3) — the zone task (upper / mid / lower)."""
    aucs = []
    for j in range(y.shape[1]):
        if len(np.unique(y[:, j])) > 1:
            aucs.append(float(roc_auc_score(y[:, j], probs[:, j])))
        else:
            aucs.append(float("nan"))
    yhat = (probs >= 0.5).astype(int)
    return {
        "n": len(y),
        "auc_macro": float(np.nanmean(aucs)),
        "auc_per_label": aucs,
        "f1": float(f1_score(y, yhat, average="macro", zero_division=0)),
        "acc": float((y == yhat).all(axis=1).mean()),  # exact-set accuracy
        "majority": float("nan"),
        "degenerate": False,
    }


def patient_binary(patient_ids, y: np.ndarray, prob: np.ndarray) -> dict:
    """Aggregate kidney predictions to the patient: prob = max (strongest
    evidence over the two kidneys), y = logical OR."""
    ids = np.asarray(patient_ids)
    ys, ps = [], []
    for pid in np.unique(ids):
        m = ids == pid
        ys.append(int(y[m].max()))
        ps.append(float(prob[m].max()))
    return binary_metrics(np.array(ys), np.array(ps))


# ---------------------------------------------------------------- across runs
def mean_ci(values, conf: float = 0.95) -> tuple[float, float, float]:
    """Mean with a t-distribution confidence interval. -> (mean, lower, upper)"""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(v.mean())
    if len(v) < 2:
        return m, m, m
    h = float(stats.t.ppf((1 + conf) / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return m, m - h, m + h


def paired_tests(model_vals, base_vals) -> dict:
    """Model versus baseline on the same (seed, fold) cells — paired t test and
    Wilcoxon signed-rank test. Both are reported: the t test assumes normality of
    the differences, the Wilcoxon does not."""
    a = np.asarray(model_vals, dtype=float)
    b = np.asarray(base_vals, dtype=float)
    assert len(a) == len(b) and len(a) >= 2
    d = a - b
    t_p = float(stats.ttest_rel(a, b).pvalue)
    try:
        w_p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        w_p = float("nan")
    return {"delta_mean": float(d.mean()), "t_p": t_p, "wilcoxon_p": w_p}
