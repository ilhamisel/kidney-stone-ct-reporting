"""Image preparation: letterbox to 224, then bag-level augmentation.

Kidney crops range from 45 to 161 pixels and vary in aspect ratio by up to about
1.5, so resizing straight to a square distorts stone shape. Letterboxing — pad to
a square with zeros, then scale to 224 — preserves the ratio; the pad value of 0
is what air maps to under the window, so the padding is not a spurious structure.

Horizontal flipping is permitted only on kidney crops. On a whole-body image it
would swap the kidneys and invert the laterality label; on a single-kidney crop it
is label-preserving.

Augmentation is applied to the ENTIRE bag with one parameter set rather than per
slice. The slices are neighbouring sections of the same kidney, so this
consistency is intended: the bag should stay a coherent view of one organ.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as T

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
RES = 224


def letterbox(img: np.ndarray, res: int = RES) -> np.ndarray:
    """(H,W,3) uint8 -> (res,res,3) uint8; ratio preserved, zero-padded and centred."""
    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    return cv2.resize(canvas, (res, res), interpolation=cv2.INTER_LINEAR)


def train_transform() -> T.Compose:
    """Augmentation recipe carried over from the earlier pipeline, plus the
    horizontal flip that the kidney crop makes admissible."""
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(MEAN, STD),
    ])


def eval_transform() -> T.Compose:
    return T.Compose([T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])
