"""Estimate of the Poincare constant of the projector output z (Table 3, Appendix C.8 of the paper).

    Var_p(f) <= C_P E_p |grad f|^2   for every f          (C_P = 1 for N(0, I))

C_P is the supremum of Var(f) / E|grad f|^2 over test functions f; we maximise the ratio over
f in span(z, cos(w_k^T z + b_k)), k = 1..M (M = 1024 random Fourier features, Pillaud-Vivien et al., 2020),
with w_k ~ N(0, (ell * m)^-2 I), b_k ~ U[0, 2 pi] and m the median distance between codes. The ratio is
a generalised Rayleigh quotient between the covariance of the features and the Gram matrix of their
gradients, maximised by a generalised eigenvector. The codes are centred and PCA-whitened (directions
with eigenvalue below 1e-6 of the largest are dropped), then split at random: the eigenvectors are
fitted on 50%, the bandwidth factor ell in {0.25, 0.5, 1, 2}, the ridge and the eigenvector are selected
on 25%, and the ratio is reported on the last 25% (the ratio on the fitting codes overestimates C_P).
Each run is estimated with its own seed (0, 1, 2, ... in the order given); the table prints mean +- std.

    python analysis/poincare_constant.py --runs simclr=RUN1,RUN2,RUN3 lejepa=... scoreclr=...

RUN is a run directory containing features.npz (key z_val by default), as written by
python -m scoreclr.evaluation.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def whiten(z):
    z = z - z.mean(0)
    ev, u = np.linalg.eigh(z.T @ z / len(z))
    keep = ev > 1e-6 * ev.max()
    return (z @ u[:, keep]) / np.sqrt(ev[keep])


def split(z, rng):
    idx = rng.permutation(len(z))
    a, b = int(0.5 * len(z)), int(0.75 * len(z))
    return z[idx[:a]], z[idx[a:b]], z[idx[b:]]


def moments(z, W, b):
    """Covariance of phi = [z, cos(zW + b)] and the gradient Gram matrix E[J J^T] (J = d phi / d z)."""
    n, d = z.shape
    P = z @ W + b
    phi = np.concatenate([z, np.cos(P)], 1)
    S = np.sin(P)
    C = np.cov(phi, rowvar=False, bias=True)
    G = np.empty_like(C)
    G[:d, :d] = np.eye(d)
    G[:d, d:] = -S.mean(0)[None, :] * W          # E[grad z_j . grad cos_k] = -E[sin_k] W_jk
    G[d:, :d] = G[:d, d:].T
    G[d:, d:] = (S.T @ S / n) * (W.T @ W)        # E[sin_k sin_l] w_k^T w_l
    return C, G


def top_generalized(C, B, top):
    """Top `top` eigenvectors of C v = lambda B v (B positive definite), via the Cholesky factor of B."""
    Li = np.linalg.inv(np.linalg.cholesky(B))
    _, U = np.linalg.eigh(Li @ C @ Li.T)
    return Li.T @ U[:, -top:]


def ratio(v, C, G):
    return float((v @ C @ v) / (v @ G @ v))


def poincare(z, features=1024, seed=0, top=20):
    """Estimate of C_P for the codes z (N, D); returns (estimate, selected hyper-parameters)."""
    rng = np.random.default_rng(seed)
    tr, va, te = split(whiten(z.astype(np.float64)), rng)
    d = tr.shape[1]
    med = np.median(np.linalg.norm(tr[rng.choice(len(tr), 500)][:, None] - tr[rng.choice(len(tr), 500)][None], axis=-1))
    best = (-1.0, None)
    for ell in [0.25, 0.5, 1.0, 2.0]:
        W = rng.standard_normal((d, features)) / (ell * med)
        b = rng.uniform(0, 2 * np.pi, features)
        Ctr, Gtr = moments(tr, W, b)
        Cva, Gva = moments(va, W, b)
        for ridge in [1e-4, 1e-3, 1e-2]:
            reg = ridge * np.trace(Gtr) / len(Gtr)
            try:
                V = top_generalized(Ctr, Gtr + reg * np.eye(len(Gtr)), top)
            except np.linalg.LinAlgError:
                continue
            for v in V.T:
                r = ratio(v, Cva, Gva)
                if r > best[0]:
                    best = (r, (W, b, v, ell, ridge))
    W, b, v, ell, ridge = best[1]
    Cte, Gte = moments(te, W, b)
    return ratio(v, Cte, Gte), dict(dim=d, ell=ell, ridge=ridge, val=best[0])



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=[], help="method=RUN1,RUN2,...")
    p.add_argument("--key", default="z_val")
    p.add_argument("--features", type=int, default=1024)
    p.add_argument("--out", default=None, help="optional JSON file with every estimate")
    args = p.parse_args()

    sets = {}
    for spec in args.runs:
        name, runs = spec.split("=", 1)
        sets[name] = [np.load(Path(r) / "features.npz")[args.key] for r in runs.split(",")]

    results = []
    print(f"{'set':20s} {'C_P estimate':>16s}   whitened dimension of each run")
    for name, zs in sets.items():
        est, dims = [], []
        for i, z in enumerate(zs):
            c, info = poincare(z, args.features, seed=i)
            est.append(c)
            dims.append(str(info["dim"]))
            results.append(dict(set=name, run=i, estimate=c, **info))
        v = np.array(est)
        cell = f"{v.mean():.3g}" + (f" +- {v.std():.2g}" if len(v) > 1 else "")
        print(f"{name:20s} {cell:>16s}   {', '.join(dims)}", flush=True)
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
