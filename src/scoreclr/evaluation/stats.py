"""Statistics of the representation (geometry and rank), from extracted features.

The functions are those that produced the paper's tables:
  RankMe            exp(entropy) of the normalized singular values of the raw feature matrix
  align / unif      Wang & Isola (2020) on L2-normalized z: alignment over the augmented pairs,
                    uniformity over 4000 random validation codes
Uniformity is random; it is seeded here, which it was not for the paper's tables, so its last digit can
differ.
"""

import numpy as np


def rankme(X):
    s = np.linalg.svd(X - 0.0, compute_uv=False)  # RankMe: singular values of the raw feature matrix
    p = s / s.sum() + 1e-12
    return float(np.exp(-(p * np.log(p)).sum()))


def align_unif(z1, z2, zv, rng):
    z1 = z1 / np.linalg.norm(z1, axis=1, keepdims=True)
    z2 = z2 / np.linalg.norm(z2, axis=1, keepdims=True)
    align = float(((z1 - z2) ** 2).sum(1).mean())
    zv = zv / np.linalg.norm(zv, axis=1, keepdims=True)
    n = min(len(zv), 4000)
    zs = zv[rng.choice(len(zv), n, replace=False)]
    d2 = ((zs[:, None, :] - zs[None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices(n, 1)
    unif = float(np.log(np.exp(-2 * d2[iu]).mean()))
    return align, unif


def representation_stats(feats: dict, seed: int = 0) -> dict:
    """All statistics of one run from its features (h_val, z_val, z_p1, z_p2; see evaluation.features)."""
    rng = np.random.RandomState(seed)
    z = feats["z_val"].astype(np.float64)
    h = feats["h_val"].astype(np.float64)
    al, un = align_unif(feats["z_p1"].astype(np.float64), feats["z_p2"].astype(np.float64), z, rng)
    return {"n": len(z), "rankme_h": rankme(h), "rankme_z": rankme(z), "align": al, "unif": un}
