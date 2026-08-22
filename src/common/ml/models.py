"""Models: a ResNet-18 encoder, the SmallSpatialTransformer, LSE pooling and the
multiple-instance wrapper.

The multiple-instance formulation: a slice encoder and a head produce one logit
per slice, and the bag logit is pooled over slices with log-sum-exp (a soft max).
For the binary task the single logit is the score for "is there a stone on at
least one slice". For the ordinal task each threshold is pooled separately, and
for the multi-label zone task each zone logit is pooled separately — LSEPool
operates per channel.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class SmallSpatialTransformer(nn.Module):
    """A ResNet-18 trunk truncated before pooling, projected to a narrow embedding
    and read by a small Transformer encoder with a class token.

    The embedding is deliberately narrow (64 dimensions). The task is a small,
    high-contrast focus inside a large field, and the global average pooling of a
    standard ResNet dilutes exactly that; keeping spatial tokens and attending
    over them suits the problem better than widening the representation.
    """

    def __init__(self, num_classes=2, embed_dim=64, depth=2, num_heads=4):
        super().__init__()
        cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(cnn.children())[:-2])
        self.project_features = nn.Conv2d(512, embed_dim, kernel_size=1)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.2, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.num_patches = 49
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.2)
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.project_features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        seq = self.project_features(self.cnn(x)).flatten(2).transpose(1, 2)
        seq = torch.cat((self.cls_token.expand(B, -1, -1), seq), dim=1) + self.pos_embedding
        return self.mlp_head(self.transformer(self.dropout(seq))[:, 0, :])


def build_encoder(name: str) -> tuple[nn.Module, int]:
    """Return (encoder, feature dimension). The head lives in MILWrapper, so no
    logit is produced here."""
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        return m, 512
    if name == "sst":
        m = SmallSpatialTransformer(num_classes=2)
        m.mlp_head = m.mlp_head[0]  # keep only the LayerNorm -> 64-d embedding
        return m, 64
    raise ValueError(name)


class LSEPool(nn.Module):
    """Log-sum-exp pooling (a soft max).

    As r grows this converges on the hard maximum, but unlike the hard maximum it
    passes gradient to every slice rather than only to the arg-max, which matters
    when the bag holds several slices through the same stone. For a bag of one
    slice the output equals that slice's logit; this identity is asserted in the
    unit tests.
    """

    def __init__(self, r: float = 4.0):
        super().__init__()
        self.r = r

    def forward(self, inst_logits):                 # (B, K, C) -> (B, C)
        return torch.logsumexp(self.r * inst_logits, dim=1) / self.r - \
            torch.log(torch.tensor(float(inst_logits.shape[1]),
                                   device=inst_logits.device)) / self.r


def numpy_lse(inst_logits: np.ndarray, r: float = 4.0) -> np.ndarray:
    """(N, C) slice logits -> (C,) bag logit. Used in full-stack evaluation, where
    the bag is every slice of the kidney rather than a sampled subset."""
    z = r * inst_logits.astype(np.float64)
    m = z.max(axis=0)
    return (m + np.log(np.exp(z - m).sum(axis=0))) / r - np.log(len(inst_logits)) / r


class MILWrapper(nn.Module):
    """Wraps an encoder with a head and LSE pooling over the bag.

    With use_pos=True the normalized within-kidney z position of the slice is
    appended to the features. The head is then a small MLP rather than a linear
    layer, because zone requires an INTERACTION between appearance and position —
    "a stone is present AND z is high". A single linear layer over [features, z]
    can only express an additive effect, never a multiplicative one.
    """

    def __init__(self, encoder: nn.Module, feat_dim: int, n_out: int,
                 r: float = 4.0, use_pos: bool = False):
        super().__init__()
        self.encoder, self.pool, self.use_pos = encoder, LSEPool(r), use_pos
        if use_pos:
            self.head = nn.Sequential(nn.Linear(feat_dim + 1, 64), nn.ReLU(),
                                      nn.Linear(64, n_out))
        else:
            self.head = nn.Linear(feat_dim, n_out)

    def instance_logits(self, x, pos=None):         # (B*K, 3, H, W) -> (B*K, C)
        f = self.encoder(x)
        if self.use_pos:
            f = torch.cat([f, pos.reshape(-1, 1)], dim=1)
        return self.head(f)

    def forward(self, x, bag_size: int, pos=None):  # -> (B, C) bag logits
        z = self.instance_logits(x, pos)
        return self.pool(z.view(-1, bag_size, z.shape[-1]))


# ------------------------------------------------------------------ ordinal (size3)
ORDINAL_THRESHOLDS = (1, 2)  # P(size>=MEDIUM), P(size>=LARGE)


def ordinal_targets(y: torch.Tensor) -> torch.Tensor:
    """y in {0,1,2} -> float targets [(y>=1), (y>=2)]."""
    return torch.stack([(y >= t).float() for t in ORDINAL_THRESHOLDS], dim=1)


def ordinal_to_class(bag_logits: torch.Tensor):
    """Class from the two thresholds: 0 + 1[>=MEDIUM] + 1[>=LARGE].

    Monotonicity is enforced rather than assumed: P(>=LARGE) is clipped to
    P(>=MEDIUM), so the two heads cannot produce the incoherent ordering
    P(>=LARGE) > P(>=MEDIUM). Returns (class, (B,3) class probabilities).
    """
    p = torch.sigmoid(bag_logits)
    p_m = p[:, 0]
    p_l = torch.minimum(p[:, 1], p_m)
    cls = (p_m > 0.5).long() + (p_l > 0.5).long()
    return cls, torch.stack([1 - p_m, p_m - p_l, p_l], dim=1)
