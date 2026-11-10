"""Lower-bound estimates of the Poincare and log-Sobolev constants of a latent distribution, from samples.

    Poincare:     Var(f)    <= C_P  E|grad f|^2      for all f
    log-Sobolev:  Ent(f^2)  <= 2 C_LS E|grad f|^2    for all f        (N(0, I): C_P = C_LS = 1)

Both constants are suprema of a ratio over test functions f, so maximising the ratio over a class of f
gives a lower bound. Estimators (each fitted on 50% of the codes, hyper-parameters selected on 25%,
reported on the last 25%, since the in-sample ratio overfits upwards):
  P_lin   f linear: the top eigenvalue of the covariance.
  P_rff   f in span(z, random Fourier features cos(w^T z + b)): top generalised eigenvector of
          (Cov phi, E[grad phi grad phi^T]), as in Pillaud-Vivien et al. (2020); grid over bandwidth and ridge.
  P_mlp   f a small MLP trained to maximise Var(f) / E|grad f|^2.
  LS_lin  f^2 = exp(g), g linear (exponential tilts; exact for a Gaussian).
  LS_mlp  f^2 = exp(g), g a small MLP.   Ratio: 2 [E_w g - log E e^g] / E_w |grad g|^2, w = softmax(g).
Since C_P <= C_LS, max(P_*) is also a lower bound on C_LS (column LS_best).

Both constants scale as the square of the latent scale, so the codes are normalised first:
  iso    centred, divided by sqrt(E|z|^2 / d): anisotropy kept (dimensional collapse shows up).
  white  centred and PCA-whitened: anisotropy removed, only non-Gaussianity / clustering remains.
--l2 projects every code on the unit sphere first (what SimCLR's loss sees).

Smoothing (--sigmas): with finite samples there are no points between separated clusters, so a test function
can jump there at no measured gradient cost and the empirical constants are unbounded. We therefore estimate
the constants of the smoothed latent p_sigma = p * N(0, sigma^2 I) (sigma in normalised units), sampled by
adding noise to the codes; for p = N(0, I), C_P = C_LS = 1 + sigma^2. sigma = 0 is reported but is only
meaningful for the closed-form estimators (P_lin, P_rff, LS_lin), whose smoothness is fixed by the class.
Whitening (--modes white) is available but not the default: random features in high dimension do not see
structure in a small subspace, and whitening a low-rank latent mostly amplifies noise.

    python analysis/functional_constants.py --sanity
    python analysis/functional_constants.py --runs simclr=RUN1,RUN2 scoreclr=RUN3,RUN4 [--l2] [--key z_val]
RUN is a run directory containing features.npz (keys z_val, h_val, y_val).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------- normalisation and splits
def normalizer(z, mode):
    """Affine map fitted on z (the training split) and applied to every split."""
    mu = z.mean(0)
    zc = z - mu
    if mode == "iso":
        s = np.sqrt((zc ** 2).sum(1).mean() / z.shape[1])
        return lambda x: (x - mu) / s
    ev, u = np.linalg.eigh(zc.T @ zc / len(z))
    keep = ev > 1e-6 * ev.max()
    A = u[:, keep] / np.sqrt(ev[keep])
    return lambda x: (x - mu) @ A


def split(z, rng):
    idx = rng.permutation(len(z))
    a, b = int(0.5 * len(z)), int(0.75 * len(z))
    return z[idx[:a]], z[idx[a:b]], z[idx[b:]]


def smooth(z, sigma, copies, rng):
    """Samples of p_sigma = p * N(0, sigma^2 I): `copies` noisy copies of every code."""
    if sigma == 0:
        return z
    return np.concatenate([z + sigma * rng.standard_normal(z.shape) for _ in range(copies)])


# ---------------------------------------------------------------- Poincare, linear and random features
def p_linear(tr, va, te):
    ev, u = np.linalg.eigh(np.cov(tr, rowvar=False, bias=True))
    v = u[:, -1]
    return float(v @ np.cov(te, rowvar=False, bias=True) @ v)


def rff_moments(z, W, b):
    """Cov of phi = [z, cos(zW + b)] and the gradient Gram E[J J^T] (J = d phi / d z)."""
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
    L = np.linalg.cholesky(B)
    Li = np.linalg.inv(L)
    _, U = np.linalg.eigh(Li @ C @ Li.T)
    return Li.T @ U[:, -top:]


def ratio_q(v, C, G):
    return float((v @ C @ v) / (v @ G @ v))


def p_rff(tr, va, te, M, rng, top=20):
    d = tr.shape[1]
    med = np.median(np.linalg.norm(tr[rng.choice(len(tr), 500)][:, None] - tr[rng.choice(len(tr), 500)][None], axis=-1))
    best = (-1.0, None)
    for ell in [0.25, 0.5, 1.0, 2.0]:
        W = rng.standard_normal((d, M)) / (ell * med)          # |w^T (z - z')| ~ |z - z'| / (ell med)
        b = rng.uniform(0, 2 * np.pi, M)
        Ctr, Gtr = rff_moments(tr, W, b)
        Cva, Gva = rff_moments(va, W, b)
        for ridge in [1e-4, 1e-3, 1e-2]:
            reg = ridge * np.trace(Gtr) / len(Gtr)
            try:
                V = top_generalized(Ctr, Gtr + reg * np.eye(len(Gtr)), top)
            except np.linalg.LinAlgError:
                continue
            for v in V.T:
                r = ratio_q(v, Cva, Gva)
                if r > best[0]:
                    best = (r, (W, b, v, ell, ridge))
    W, b, v, ell, ridge = best[1]
    Cte, Gte = rff_moments(te, W, b)
    return ratio_q(v, Cte, Gte), dict(ell=ell, ridge=ridge, val=best[0])


# ---------------------------------------------------------------- MLP / tilt estimators (torch)
DEV = torch.device("cpu")


def mlp(d, hidden, linear):
    if linear:
        return torch.nn.Linear(d, 1)
    return torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden),
                               torch.nn.SiLU(), torch.nn.Linear(hidden, 1))


def value_grad(f, z, create_graph):
    z = z.clone().requires_grad_(True)
    out = f(z).squeeze(-1)
    (g,) = torch.autograd.grad(out.sum(), z, create_graph=create_graph)
    return out, g


def poincare_ratio(out, g):
    return out.var(unbiased=False) / g.pow(2).sum(1).mean()


def lsi_ratio(out, g):
    """f^2 = e^out: Ent(f^2) / (2 E|grad f|^2) = 2 [E_w out - log E e^out] / E_w |grad out|^2."""
    w = torch.softmax(out, 0)
    ent = (w * out).sum() - (torch.logsumexp(out, 0) - math.log(len(out)))
    return 2 * ent / (w * g.pow(2).sum(1)).sum(), 1.0 / w.pow(2).sum() / len(out)   # ratio, relative ESS


def fit(kind, tr, va, te, linear, steps, hidden=256, lr=1e-3, min_ess=0.02, seed=0):
    torch.manual_seed(seed)
    tr, va, te = (torch.as_tensor(x, dtype=torch.float32, device=DEV) for x in (tr, va, te))
    f = mlp(tr.shape[1], hidden, linear).to(DEV)
    if kind == "ls":   # start from a small tilt: the ratio of a constant g is 0/0
        for p in f.parameters():
            torch.nn.init.normal_(p, std=0.01)
    opt = torch.optim.Adam(f.parameters(), lr=lr)

    def score(z, create_graph):
        out, g = value_grad(f, z, create_graph)
        if kind == "p":
            return poincare_ratio(out, g), torch.tensor(1.0)
        return lsi_ratio(out, g)

    best_val, best_state = -1.0, None
    for step in range(steps):
        r, _ = score(tr, True)
        opt.zero_grad()
        (-r).backward()
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            rv, ess = score(va, False)
            rv, ess = float(rv.detach()), float(torch.as_tensor(ess).detach())
            if ess >= min_ess and math.isfinite(rv) and rv > best_val:
                best_val, best_state = rv, {k: v.detach().clone() for k, v in f.state_dict().items()}
    if best_state is None:
        return float("nan"), dict(val=float("nan"))
    f.load_state_dict(best_state)
    rt, ess = score(te, False)
    return float(rt.detach()), dict(val=best_val, test_ess=float(ess))


# ---------------------------------------------------------------- driver
def estimate(z, mode, sigma, args, seed=0):
    rng = np.random.default_rng(seed)
    tr, va, te = split(z.astype(np.float64), rng)
    norm = normalizer(tr, mode)
    tr, va, te = (smooth(norm(x), sigma, args.copies, rng) for x in (tr, va, te))
    out, extra = {"dim": tr.shape[1]}, {}
    out["P_lin"] = p_linear(tr, va, te)
    out["P_rff"], extra["P_rff"] = p_rff(tr, va, te, args.features, rng)
    if not args.no_mlp:
        out["P_mlp"], extra["P_mlp"] = fit("p", tr, va, te, False, args.steps, seed=seed)
    out["LS_lin"], extra["LS_lin"] = fit("ls", tr, va, te, True, args.steps, seed=seed)
    if not args.no_mlp:
        out["LS_mlp"], extra["LS_mlp"] = fit("ls", tr, va, te, False, args.steps, seed=seed)
    finite = [v for k, v in out.items() if k != "dim" and math.isfinite(v)]
    out["LS_best"] = max(finite)
    return out, extra


def load(run, key, l2):
    z = np.load(Path(run) / "features.npz")[key]
    if l2:
        z = z / np.linalg.norm(z, axis=1, keepdims=True)
    return z


def sanity(args):
    rng = np.random.default_rng(0)
    d, n = 256, 10000
    gauss = rng.standard_normal((n, d))
    means = rng.standard_normal((10, d)) * 1.0          # 10 clusters, centre distance ~ sqrt(2d)
    mix = means[rng.integers(10, size=n)] + rng.standard_normal((n, d))
    return {"gaussian (C=1)": gauss, "10-cluster mixture (C>>1)": mix}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=[], help="method=RUN1,RUN2,...")
    p.add_argument("--key", default="z_val")
    p.add_argument("--l2", action="store_true")
    p.add_argument("--modes", nargs="+", default=["iso"])
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.3, 1.0],
                   help="noise std of the smoothed latent p * N(0, sigma^2 I), in normalised units")
    p.add_argument("--copies", type=int, default=2, help="noisy copies per code when sigma > 0")
    p.add_argument("--features", type=int, default=1024)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--no-mlp", action="store_true")
    p.add_argument("--sanity", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    global DEV
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sets = {}
    if args.sanity:
        sets |= {name: [z] for name, z in sanity(args).items()}
    for spec in args.runs:
        name, runs = spec.split("=", 1)
        sets[name] = [load(r, args.key, args.l2) for r in runs.split(",")]

    cols = ["P_lin", "P_rff"] + ([] if args.no_mlp else ["P_mlp"]) + ["LS_lin"] + ([] if args.no_mlp else ["LS_mlp"]) + ["LS_best"]
    results = []
    print(f"key={args.key} l2={args.l2}   lower bounds on C_P and C_LS of p * N(0, sigma^2 I) (N(0,I): 1 + sigma^2)")
    print(f"{'set':28s} {'mode':5s} {'sigma':>5s} {'dim':>4s} " + " ".join(f"{c:>14s}" for c in cols))
    for name, zs in sets.items():
        for mode in args.modes:
          for sigma in args.sigmas:
            per = []
            for i, z in enumerate(zs):
                t = time.time()
                r, extra = estimate(z, mode, sigma, args, seed=i)
                per.append(r)
                results.append(dict(set=name, mode=mode, sigma=sigma, run=i, **r, extra=extra, seconds=time.time() - t))
            cell = []
            for c in cols:
                v = np.array([r[c] for r in per], float)
                cell.append(f"{v.mean():8.3g}" + (f"±{v.std():<5.2g}" if len(v) > 1 else "      "))
            print(f"{name:28s} {mode:5s} {sigma:5.2f} {per[0]['dim']:4d} " + " ".join(f"{c:>14s}" for c in cell), flush=True)
    if args.out:
        json.dump(dict(args=vars(args), results=results), open(args.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
