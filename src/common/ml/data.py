"""Kidney-level multiple-instance data layer.

The modelling unit is the KIDNEY and the splitting unit is the PATIENT. Inputs are
the per-kidney crop stacks produced by stage 08; the number of slices ranges from
5 to 164 per kidney, so a bag cannot be a fixed slab.

Bag strategy:
  training      a stratified-uniform sample of K slices: the stack is divided into
                K equal segments and one random slice is drawn from each. Different
                slices appear each epoch, which acts as natural augmentation, and
                no padding or masking is needed for variable-length stacks.
  evaluation    ALL slices, with LSE pooling on the model side. A small stone that
                appears on 2 of 164 slices must not be missed at test time because
                the sampler happened to skip it.

The three-window input is stored on disk as genuine RGB (R = soft 40/400,
G = stone 400/1500, B = wide 500/3000). cv2.imread returns BGR, so the channel
order is reversed on read.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common.paths import LABELS, ROOT
from common.ml.transforms import letterbox

SEEDS = (1337, 2027, 7, 42, 12345)
N_FOLDS = 5
SIZE3 = {"MIKROLITIYAZIS": 0, "KUCUK": 0, "ORTA": 1, "BUYUK": 2, "COK_BUYUK": 2}
SIZE3_NAMES = ("SMALL", "MEDIUM", "LARGE")
ZONE_COLS = ("zone_upper", "zone_mid", "zone_lower")

# stack cache: (kidney_id, input) -> (N,3,224,224) uint8, decoded once per process
_STACKS: dict[tuple[str, str], torch.Tensor] = {}


N_KIDNEYS = 390          # one examination per patient (see stage 18)
N_PATIENTS = 195


def load_table(all_exams: bool = False) -> pd.DataFrame:
    """The modelling cohort: ONE examination per patient, the earliest by date.

    Two patients have two examinations each. Because splitting is at patient level
    they always fall in the same fold, so there was never any train/test leakage.
    But counting the same patient twice in a test fold breaks the independence
    assumption behind the confidence intervals and makes them too narrow, so the
    index examination is kept. The criterion is independent of the outcome, so the
    cohort is not selected on the result.

    `all_exams=True` returns the full cohort, for descriptive statistics only.
    """
    df = pd.read_csv(LABELS / "kidney_dataset.csv")
    assert len(df) == 394 and df["patient_id"].nunique() == N_PATIENTS, \
        "kidney table violates its contract"
    if not all_exams:
        df = df[df["primary_exam"]].reset_index(drop=True)
        assert len(df) == N_KIDNEYS and df["case_id"].nunique() == N_PATIENTS, \
            "the index-examination filter did not yield the expected cohort"
    for col in ("halfcrop_stone_dir", "halfcrop_mw3_dir"):
        df[col] = df[col].map(lambda p: str(ROOT / str(p).replace("\\", "/")))
        missing = [p for p in df[col] if not Path(p).is_dir()]
        assert not missing, f"missing crop directory: {missing[:3]}"
    # Coronal crops (stage 22) are used only by input='cor'. Existence is checked
    # when the stack is read, so the axial tasks run without stage 22 having run.
    df["corcrop_dir"] = [
        str(ROOT / "03_derived" / c / f"corcrop_{s.lower()}_stone")
        for c, s in zip(df["case_id"], df["side"])
    ]
    df["has_stone"] = df["has_stone"].astype(bool)
    df["size3"] = df["max_class"].map(SIZE3)
    for c in ZONE_COLS:
        df[c] = df[c].astype(bool)
    return df


def fold_split(df: pd.DataFrame, seed: int, fold: int):
    """test = fold `fold`, val = fold (fold+1)%5, train = the remaining three.
    Every split is at patient level."""
    col = f"cv5_seed{seed}"
    assert col in df.columns, f"unknown seed: {seed}"
    te = df[df[col] == fold]
    va = df[df[col] == (fold + 1) % N_FOLDS]
    tr = df[~df[col].isin([fold, (fold + 1) % N_FOLDS])]
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not set(a["patient_id"]) & set(b["patient_id"]), "patient leakage"
    assert len(tr) + len(va) + len(te) == len(df)
    # each patient appears in exactly one split, represented by one examination
    assert df.groupby("patient_id")["case_id"].nunique().max() == 1, \
        "more than one examination per patient"
    return tr, va, te


def task_subset(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """The subset of kidneys on which the task is defined.

    Applied AFTER the split, never before: subsetting first would let the fold
    sizes depend on the labels.
    """
    if task == "has_stone":
        return df
    if task == "size3":
        return df[df["has_stone"] & df["size3"].notna()]
    if task == "zone":
        return df[df["has_stone"] & df[list(ZONE_COLS)].any(axis=1)]
    raise ValueError(task)


def targets_of(df: pd.DataFrame, task: str) -> np.ndarray:
    if task == "has_stone":
        return df["has_stone"].to_numpy(dtype=np.int64)
    if task == "size3":
        return df["size3"].to_numpy(dtype=np.int64)
    if task == "zone":
        return df[list(ZONE_COLS)].to_numpy(dtype=np.float32).copy()
    raise ValueError(task)


INPUT_DIR_COL = {"stone": "halfcrop_stone_dir", "mw3": "halfcrop_mw3_dir",
                 "cor": "corcrop_dir"}


def _read_stack(crop_dir: str, input_kind: str) -> torch.Tensor:
    files = sorted(Path(crop_dir).glob("*.png"))
    assert files, f"empty crop directory: {crop_dir}"
    out = []
    for f in files:
        if input_kind == "mw3":
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR -> RGB
        else:  # stone / cor: single channel replicated to three
            g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            img = np.repeat(g[:, :, None], 3, axis=2)
        out.append(torch.from_numpy(letterbox(np.ascontiguousarray(img))))
    return torch.stack(out).permute(0, 3, 1, 2).contiguous()  # (N,3,224,224) uint8


def get_stack(row, input_kind: str) -> torch.Tensor:
    key = (row["kidney_id"], input_kind)
    if key not in _STACKS:
        _STACKS[key] = _read_stack(row[INPUT_DIR_COL[input_kind]], input_kind)
    return _STACKS[key]


def sample_indices(n: int, k: int) -> torch.Tensor:
    """Stratified-uniform sample of K slices. When n <= k the stack is covered
    deterministically with linspace, repeating slices rather than padding."""
    if n <= k:
        return torch.from_numpy(np.round(np.linspace(0, n - 1, k)).astype(np.int64))
    edges = np.linspace(0, n, k + 1).astype(np.int64)
    lo = torch.from_numpy(edges[:-1])
    span = torch.from_numpy(edges[1:] - edges[:-1])
    return lo + (torch.rand(k) * span).long().clamp(max=span - 1)


def norm_positions(idx: torch.Tensor, n: int) -> torch.Tensor:
    """Normalized z position of a slice within the kidney stack, in 0..1.

    Zone is a POSITION along the z axis. LSE pooling is permutation-invariant, so
    unless this is supplied separately the model cannot learn zone at all. This was
    measured rather than assumed: the end-to-end formulation reaches a macro AUC of
    0.55, essentially chance. Because the stack is already cropped to the kidney
    bounding box, this value is the kidney-relative zone frame itself.
    """
    if n <= 1:
        return torch.full((len(idx),), 0.5)
    return idx.to(torch.float32) / float(n - 1)


class KidneyBagDataset(Dataset):
    """Yields a (K,3,224,224) float bag, the (K,) z positions and the target.

    With mirror=True the LEFT kidney is flipped horizontally, which turns two
    mirror-image problems into a single canonical-kidney problem and doubles the
    effective sample size seen by the encoder.
    """

    def __init__(self, df: pd.DataFrame, task: str, input_kind: str, k: int,
                 train: bool, mirror: bool, tfm):
        self.rows = df.reset_index(drop=True)
        self.y = targets_of(self.rows, task)
        self.input_kind, self.k, self.train, self.mirror, self.tfm = \
            input_kind, k, train, mirror, tfm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows.iloc[i]
        stack = get_stack(row, self.input_kind)
        idx = sample_indices(len(stack), self.k) if self.train \
            else torch.from_numpy(np.round(np.linspace(0, len(stack) - 1, self.k)).astype(np.int64))
        xs = stack[idx]
        if self.mirror and row["side"] == "LEFT":
            xs = torch.flip(xs, dims=[-1])          # a horizontal flip leaves z intact
        return self.tfm(xs), norm_positions(idx, len(stack)), self.y[i]


def full_stack(row, input_kind: str, mirror: bool):
    """Evaluation path: EVERY slice of the kidney.
    -> (uint8 (N,3,224,224), (N,) z positions)"""
    xs = get_stack(row, input_kind)
    if mirror and row["side"] == "LEFT":
        xs = torch.flip(xs, dims=[-1])
    return xs, norm_positions(torch.arange(len(xs)), len(xs))


def clear_cache() -> None:
    _STACKS.clear()
