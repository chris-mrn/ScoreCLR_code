"""Statistics of the representation (geometry, rank, normality), from extracted features.

The functions are those that produced the paper's tables:
  RankMe            exp(entropy) of the normalized singular values of the raw feature matrix
  top-1 share       variance share of the first principal component
  kurtosis          mean per-coordinate excess kurtosis of z
  Epps-Pulley (EP)  LeJEPA's SIGReg statistic (17 knots on [0, 3], 256 random unit slices, N-scaled,
                    mean of 5 draws) on ZCA-whitened z, with an N(0, I) sample of the same size as the
                    reference
  align / unif      Wang & Isola (2020) on L2-normalized z: alignment over the augmented pairs,
                    uniformity over 4000 random validation codes
EP and uniformity are random; they are seeded here, which they were not for the paper's tables, so
their last digits can differ (by about the sd over the 5 EP draws).
"""

import numpy as np
import torch

from scoreclr.methods.lejepa import SIGReg


def rankme(X):
    s = np.linalg.svd(X - 0.0, compute_uv=False)  # RankMe: singular values of the raw feature matrix
    p = s / s.sum() + 1e-12
    return float(np.exp(-(p * np.log(p)).sum()))


def top1_share(X):
    C = np.cov(X, rowvar=False)
    ev = np.linalg.eigvalsh(C)[::-1]
    return float(ev[0] / ev.sum())


def excess_kurtosis(X):
    Xc = X - X.mean(0)
    m2 = (Xc ** 2).mean(0)
    m4 = (Xc ** 4).mean(0)
    k = m4 / (m2 ** 2 + 1e-12) - 3.0
    return float(k.mean()), float(np.median(k)), float(k.max())


def ep(sig, Xt, reps=5):
    vals = [float(sig(Xt)) for _ in range(reps)]
    return float(np.mean(vals)), float(np.std(vals))


def whiten(X):
    Xc = X - X.mean(0)
    C = np.cov(Xc, rowvar=False)
    ev, U = np.linalg.eigh(C)
    ev = np.clip(ev, 1e-8, None)
    return Xc @ U @ np.diag(ev ** -0.5) @ U.T


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
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    sig = SIGReg(knots=17, t_max=3.0, num_slices=256)
    z = feats["z_val"].astype(np.float64)
    h = feats["h_val"].astype(np.float64)
    n, d = z.shape
    km, kmed, kmax = excess_kurtosis(z)
    ep_wh, ep_wh_sd = ep(sig, torch.tensor(whiten(z), dtype=torch.float32))
    ep_ref, ep_ref_sd = ep(sig, torch.randn(n, d))
    al, un = align_unif(feats["z_p1"].astype(np.float64), feats["z_p2"].astype(np.float64), z, rng)
    return {"n": n, "rankme_h": rankme(h), "rankme_z": rankme(z), "top1_share_z": top1_share(z),
            "kurtosis_mean": km, "kurtosis_median": kmed, "kurtosis_max": kmax,
            "ep_whitened": ep_wh, "ep_whitened_sd": ep_wh_sd, "ep_gaussian_ref": ep_ref, "ep_gaussian_ref_sd": ep_ref_sd,
            "align": al, "unif": un}
