"""Diagnostics of ScoreCLR's run-to-run variability. Each group of metrics tests one explanation:

  geom/   dimensional collapse: effective rank of the codes z and backbone features h, share of the top
          eigenvalue, per-dimension spread of z
  sched/  noise schedule against the code scale: median sigma / median per-dimension spread of z, and the
          share of sampled sigmas below that spread (noise that resolves the codes' structure)
  dsm/    score-net quality at five noise levels sigma_q = exp(p_mean + q * p_std), q = -2..2: its DSM
          loss, the DSM loss of the exact denoiser of a Gaussian fitted to the codes (ratio > 1: the net
          is worse than a Gaussian fit) and its output energy (~ 0: the net predicts nothing, no repulsion)
  lag/    score net against the encoder: DSM of the net on the current codes before and after its update
          (fixed noise); a growing gap means the net is chasing codes the encoder keeps moving
  rep/    averaging the noise queries: signal-to-noise of the per-code score over 4 queries
          (||E s||^2 / Var s), cosine between the queries' scores
  force/  what each loss term does to the code covariance: rate of change of the log-variance of the top
          10 % and the bottom 50 % eigendirections of z under the attraction, the repulsion (learned
          score and Gaussian-oracle score), the prior and their sum (learned); a negative total on the
          bottom directions means they shrink, i.e. dimensional collapse
  step/   every training step's terms and grad norms (steps.csv) and, per epoch (diag_epochs.jsonl), their
          min / p50 / p99 / max, the spikes (> 3x the epoch median) and the share of steps a clip at 1 or
          2 would cut
  twin    GPU non-determinism and chaos: two copies of one run on the same batches and the same noise,
          with or without deterministic kernels or a tiny initial perturbation, and how far apart they
          drift (parameters, codes on a fixed probe set) step after step

    python analysis/diagnostics.py train --config config/cifar10_scoreclr.yaml [--every 100] [key=value ...]
    python analysis/diagnostics.py twin --config config/cifar10_scoreclr.yaml --out twin.jsonl --epochs 20 \
        [--deterministic] [--perturb 1e-6] [key=value ...]
    python analysis/diagnostics.py report RUN [RUN ...] [--epochs 9,49,99,299,999]

`train` trains exactly as `python -m scoreclr.train`, with the same trajectory: the probes run without
gradient, with the encoder in eval mode (no BatchNorm update), on a fixed probe set (two augmented views
of n training images) and with their own random generator. It writes diagnostics.jsonl (one row every
`every` steps), diag_epochs.jsonl and steps.csv to output_dir, next to the usual log.jsonl.
"""

import argparse
import contextlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

import scoreclr.train.train as train_module
from scoreclr.data import TwoViews, build_dataset, build_loader, pretrain_transform
from scoreclr.methods import METHODS, ScoreCLR
from scoreclr.train.optim import set_lr
from scoreclr.utils import distributed as dist
from scoreclr.utils import load_config, seed_everything

LEVELS = (-2, -1, 0, 1, 2)  # noise levels of dsm/, in standard deviations of log sigma
QUERIES = 4  # queries of rep/


class Probe:
    """A fixed probe set (two augmented views of n training images) and its own random generator."""

    def __init__(self, cfg, device, n: int, seed: int = 0):
        devices = [device] if device.type == "cuda" else []
        py_state, np_state = random.getstate(), np.random.get_state()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            view = pretrain_transform(cfg.data.dataset, cfg.data.crop_size, cfg.data.augment)
            data = build_dataset(cfg.data, train=True, transform=TwoViews(view))
            idx = torch.randperm(len(data))[: min(n, len(data))].tolist()
            pairs = [data[i][0] for i in idx]
        random.setstate(py_state)
        np.random.set_state(np_state)
        self.x0 = torch.stack([p[0] for p in pairs]).to(device)
        self.x1 = torch.stack([p[1] for p in pairs]).to(device)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed + 1)

    @torch.no_grad()
    def codes(self, encoder, chunk: int = 512):
        """(h0, h1, z0, z1) of the probe views, encoder in eval mode (no BatchNorm update)."""
        was_training = encoder.training
        encoder.eval()
        out = [[], [], [], []]
        for i in range(0, self.x0.size(0), chunk):
            (h0, h1), (z0, z1), _ = encoder([self.x0[i : i + chunk], self.x1[i : i + chunk]])
            for lst, t in zip(out, (h0, h1, z0, z1)):
                lst.append(t.float())
        encoder.train(was_training)
        return [torch.cat(lst) for lst in out]

    def randn(self, *shape, device):
        return torch.randn(*shape, generator=self.gen, device=device)

    def queries(self, n: int, dim: int, p_mean: float, p_std: float, device):
        """sigma (n,) from the training schedule and eps (n, dim)."""
        return (p_mean + p_std * self.randn(n, device=device)).exp(), self.randn(n, dim, device=device)


def effective_rank(x: torch.Tensor) -> float:
    """exp(entropy) of the normalized eigenvalues of the covariance of x (N, d) (RankMe-like)."""
    lam = torch.linalg.eigvalsh(torch.cov(x.T)).clamp_min(0)
    p = lam / lam.sum()
    return math.exp(-(p * (p + 1e-12).log()).sum().item())


class GaussianFit:
    """N(mu, C) fitted to codes z (N, d): eigenbasis and exact denoiser / score of N(mu, C + sigma^2 I)."""

    def __init__(self, z: torch.Tensor):
        self.mu = z.mean(0)
        lam, U = torch.linalg.eigh((z - self.mu).T @ (z - self.mu) / z.size(0))
        self.lam, self.U = lam.flip(0).clamp_min(0), U.flip(1)  # descending

    def eps_hat(self, z_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """E[eps | z_t] = sigma (C + sigma^2 I)^-1 (z_t - mu)."""
        p = (z_t - self.mu) @ self.U
        return (sigma[:, None] * p / (self.lam + sigma[:, None] ** 2)) @ self.U.T

    def score(self, z_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return -self.eps_hat(z_t, sigma) / sigma[:, None]

    def groups(self):
        """Indices of the top 10 % and the bottom 50 % eigendirections (numerically zero ones excluded)."""
        d = self.lam.numel()
        top = torch.arange(math.ceil(0.1 * d), device=self.lam.device)
        bottom = torch.arange(d // 2, d, device=self.lam.device)
        return {"top": top, "bottom": bottom[self.lam[bottom] > 1e-6 * self.lam.sum()]}

    def log_var_rates(self, z: torch.Tensor, g: torch.Tensor) -> dict:
        """d log Var / dt of each group's subspace when the codes descend the per-code gradient g (N, d)."""
        p, q = (z - self.mu) @ self.U, g @ self.U
        cross, var = (p * q).mean(0), p.pow(2).mean(0)
        return {k: (-2 * cross[i].sum() / var[i].sum().clamp_min(1e-12)).item() for k, i in self.groups().items()}


def dsm(net, z, sigma, eps) -> float:
    return (net(z + sigma[:, None] * eps, sigma) - eps).pow(2).mean().item()


@torch.no_grad()
def probe_metrics(method: ScoreCLR, probe: Probe, h0, h1, z0, z1) -> dict:
    """Every geom/, sched/, dsm/, rep/ and force/ metric on the probe codes."""
    net, p_mean, p_std = method.score_net, method.p_mean, method.p_std
    z = torch.cat([z0, z1])
    n, d = z.shape
    dev = z.device
    fit = GaussianFit(z)
    std = z.std(0)
    m = {"geom/erank_z": effective_rank(z), "geom/erank_h": effective_rank(torch.cat([h0, h1])),
         "geom/top1_share": (fit.lam[0] / fit.lam.sum()).item(),
         "geom/std_median": std.median().item(), "geom/std_min": std.min().item(),
         "geom/radius": (z.norm(dim=1).mean() / math.sqrt(d)).item()}
    m["sched/sigma_median_over_std"] = math.exp(p_mean) / m["geom/std_median"]
    m["sched/frac_sigma_below_std"] = 0.5 * (1 + math.erf((math.log(m["geom/std_median"]) - p_mean) / (p_std * math.sqrt(2))))

    for q in LEVELS:
        sigma = torch.full((n,), math.exp(p_mean + q * p_std), device=dev)
        eps = probe.randn(n, d, device=dev)
        z_t = z + sigma[:, None] * eps
        e_net = net(z_t, sigma)
        m[f"dsm/net_q{q:+d}"] = (e_net - eps).pow(2).mean().item()
        m[f"dsm/gauss_q{q:+d}"] = (fit.eps_hat(z_t, sigma) - eps).pow(2).mean().item()
        m[f"dsm/ratio_q{q:+d}"] = m[f"dsm/net_q{q:+d}"] / max(m[f"dsm/gauss_q{q:+d}"], 1e-12)
        m[f"dsm/out_q{q:+d}"] = e_net.pow(2).mean().item()

    scores, oracle, z_ts = [], [], []
    for _ in range(QUERIES):
        sigma, eps = probe.queries(n, d, p_mean, p_std, dev)
        z_t = z + sigma[:, None] * eps
        z_ts.append(z_t)
        scores.append(net.score(z_t, sigma))
        oracle.append(fit.score(z_t, sigma))
    s = torch.stack(scores)
    mean = s.mean(0)
    var = (s - mean).pow(2).sum(-1).sum(0) / (QUERIES - 1)  # per code, per query
    signal = mean.pow(2).sum(-1) - var / QUERIES  # unbiased ||E s||^2
    m["rep/snr"] = (signal.mean() / var.mean()).item()
    unit = torch.nn.functional.normalize(s, dim=-1)
    cos = torch.einsum("knd,lnd->kln", unit, unit)
    m["rep/cos_queries"] = ((cos.sum((0, 1)) - QUERIES) / (QUERIES * (QUERIES - 1))).mean().item()
    m["rep/score_norm"] = (s.pow(2).sum(-1).mean() / d).item()

    k = method.num_queries  # the training estimator: mean over its K queries
    forces = {
        "attr": 4 * method.attr_weight * torch.cat([z0 - z1, z1 - z0]),  # all per code, times 2B * d
        "rep": s[:k].mean(0),
        "rep_oracle": torch.stack(oracle[:k]).mean(0),
        "prior": torch.stack(z_ts[:k]).mean(0),
    }
    forces["total"] = forces["attr"] + forces["rep"] + forces["prior"]
    for name, g in forces.items():
        for group, rate in fit.log_var_rates(z, g).items():
            m[f"force/{name}_{group}"] = rate
    for name in ("attr", "rep", "prior"):
        m[f"force/norm_{name}"] = forces[name].norm(dim=1).mean().item()
    return m


class Recorder:
    """Collects the per-step terms and the probe rows of a diagnosed training and writes them (rank 0)."""

    def __init__(self, out: Path, every: int, probe_size: int):
        self.out, self.every, self.probe_size = Path(out), every, probe_size
        self.step, self.epoch, self.due, self.rows, self.probe = 0, 0, False, [], None
        self.main = dist.is_main()

    def write(self, name: str, line: str) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        with open(self.out / name, "a") as f:
            f.write(line + "\n")

    def probe_for(self, cfg, device) -> Probe:
        if self.probe is None:
            self.probe = Probe(cfg, device, self.probe_size)
        return self.probe

    def flush_epoch(self) -> None:
        if not (self.main and self.rows):
            return
        keys = sorted(self.rows[0])
        values = torch.stack([torch.stack([r[k].float() for k in keys]) for r in self.rows]).cpu()
        first = self.step - len(self.rows)
        if not (self.out / "steps.csv").exists():
            self.write("steps.csv", ",".join(["step", "epoch", *keys]))
        self.write("steps.csv", "\n".join(f"{first + i},{self.epoch}," + ",".join(f"{v:.5g}" for v in row.tolist())
                                          for i, row in enumerate(values)))
        summary = {"epoch": self.epoch}
        for j, k in enumerate(keys):
            x = values[:, j]
            summary |= {f"step/{k}_min": x.min().item(), f"step/{k}_p50": x.median().item(),
                        f"step/{k}_p99": x.quantile(0.99).item(), f"step/{k}_max": x.max().item()}
            if k.endswith("grad_norm"):
                summary |= {f"step/{k}_spikes": int((x > 3 * x.median()).sum()),
                            f"step/{k}_frac_gt1": (x > 1).float().mean().item(),
                            f"step/{k}_frac_gt2": (x > 2).float().mean().item()}
        self.write("diag_epochs.jsonl", json.dumps(summary))
        self.rows = []


def diagnosed_scoreclr(recorder: Recorder, cfg):
    class DiagnosedScoreCLR(ScoreCLR):
        """ScoreCLR whose auxiliary step also runs the probes, every `recorder.every` steps on rank 0."""

        def auxiliary_step(self, z, optimizer):
            if not (recorder.due and recorder.main):
                return super().auxiliary_step(z, optimizer)
            probe = recorder.probe_for(cfg, z.device)
            h0, h1, z0, z1 = probe.codes(self.encoder)
            zp = torch.cat([z0, z1])
            sigma, eps = probe.queries(zp.size(0), zp.size(1), self.p_mean, self.p_std, zp.device)
            with torch.no_grad():
                before = dsm(self.score_net, zp, sigma, eps)
            terms = super().auxiliary_step(z, optimizer)
            with torch.no_grad():
                after = dsm(self.score_net, zp, sigma, eps)
            row = {"step": recorder.step, "epoch": recorder.epoch,
                   "lag/dsm_before": before, "lag/dsm_after": after, "lag/gain": before - after}
            row |= probe_metrics(self, probe, h0, h1, z0, z1)
            recorder.write("diagnostics.jsonl", json.dumps(row))
            return terms

    return DiagnosedScoreCLR


@contextlib.contextmanager
def diagnosing(cfg, every: int, probe_size: int):
    """Within the context, scoreclr.train.train.train(cfg) records the diagnostics."""
    recorder = Recorder(cfg.output_dir, every, probe_size)
    orig_method, orig_step, orig_epoch = METHODS["scoreclr"], train_module.train_step, train_module.train_one_epoch

    def train_step(*args, **kwargs):
        recorder.due = recorder.step % recorder.every == 0
        terms = orig_step(*args, **kwargs)
        if recorder.main:
            recorder.rows.append({k: v.detach() for k, v in terms.items()})
        recorder.step += 1
        return terms

    def train_one_epoch(method, encoder, enc_opt, aux_opt, loader, cfg, epoch, device, scaler=None):
        recorder.epoch, recorder.step, recorder.rows = epoch, epoch * len(loader), []
        metrics = orig_epoch(method, encoder, enc_opt, aux_opt, loader, cfg, epoch, device, scaler)
        recorder.flush_epoch()
        return metrics

    METHODS["scoreclr"] = diagnosed_scoreclr(recorder, cfg)
    train_module.train_step, train_module.train_one_epoch = train_step, train_one_epoch
    try:
        yield recorder
    finally:
        METHODS["scoreclr"] = orig_method
        train_module.train_step, train_module.train_one_epoch = orig_step, orig_epoch


def train(cfg, every: int = 100, probe_size: int = 2048) -> dict:
    assert cfg.get("method", "scoreclr") == "scoreclr", "the diagnostics are ScoreCLR's"
    with diagnosing(cfg, every, probe_size):
        return train_module.train(cfg)


def param_distance(a, b) -> float:
    """||theta_a - theta_b|| / ||theta_a|| over the parameters of two copies of a module."""
    num = den = 0.0
    for pa, pb in zip(a.parameters(), b.parameters()):
        num += (pa.detach().double() - pb.detach().double()).pow(2).sum().item()
        den += pa.detach().double().pow(2).sum().item()
    return math.sqrt(num / max(den, 1e-30))


def twin(cfg, out: Path, epochs: int, perturb: float = 0.0, log_every: int = 20, probe_size: int = 1024,
         max_steps: int = None) -> list:
    """Trains two copies of one run on the same batches and noise; returns (and writes) their distance."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)
    view = pretrain_transform(cfg.data.dataset, cfg.data.crop_size, cfg.data.augment)
    loader = build_loader(build_dataset(cfg.data, train=True, transform=TwoViews(view)), cfg.optim.batch_size,
                          train=True, num_workers=cfg.data.num_workers)
    twins = []
    for _ in range(2):
        seed_everything(cfg.seed)
        twins.append(train_module.setup(cfg, device))
    if perturb:
        gen = torch.Generator(device=device)
        gen.manual_seed(12345)
        with torch.no_grad():
            for p in twins[1][0].encoder.parameters():
                p.mul_(1 + perturb * torch.randn(p.shape, generator=gen, device=device))
    fp16 = cfg.optim.get("precision", "fp32") == "fp16"
    scalers = [torch.amp.GradScaler(device.type) if fp16 else None for _ in twins]
    probe = Probe(cfg, device, probe_size)
    rows, step = [], 0
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for epoch in range(epochs):
            for _, encoder, _, _ in twins:
                encoder.train()
            for i, ((x0, x1), y) in enumerate(loader):
                x0, x1, y = x0.to(device), x1.to(device), y.to(device)
                cpu_state = torch.get_rng_state()
                cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                terms = []
                for (method, encoder, enc_opt, aux_opt), scaler in zip(twins, scalers):
                    torch.set_rng_state(cpu_state)
                    if cuda_state is not None:
                        torch.cuda.set_rng_state(cuda_state, device)
                    set_lr([o for o in (enc_opt, aux_opt) if o is not None], cfg.optim, epoch, i, len(loader))
                    terms.append(train_module.train_step(method, encoder, enc_opt, aux_opt, x0, x1, y, cfg, scaler))
                if step % log_every == 0:
                    (a, *_), (b, *_) = twins
                    za, zb = probe.codes(a.encoder)[2], probe.codes(b.encoder)[2]
                    row = {"step": step, "epoch": epoch, "encoder": param_distance(a.encoder, b.encoder),
                           "codes": ((za - zb).norm() / za.norm().clamp_min(1e-30)).item()}
                    if isinstance(a, ScoreCLR):
                        row["score_net"] = param_distance(a.score_net, b.score_net)
                    row |= {f"a/{k}": v.item() for k, v in terms[0].items()}
                    row |= {f"b/{k}": v.item() for k, v in terms[1].items()}
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                step += 1
                if max_steps is not None and step >= max_steps:
                    return rows
    return rows


def load_rows(path: Path) -> list:
    """Rows of a jsonl file, the last occurrence of each step or epoch kept (resumed runs)."""
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[r.get("step", r.get("epoch"))] = r
    return [rows[k] for k in sorted(rows)]


REPORT = ["geom/erank_z", "geom/erank_h", "geom/std_median", "sched/sigma_median_over_std",
          "dsm/ratio_q-2", "dsm/ratio_q+0", "dsm/out_q-2", "lag/gain", "rep/snr",
          "force/rep_bottom", "force/rep_oracle_bottom", "force/attr_bottom", "force/total_bottom", "force/total_top"]


def report(runs, epochs) -> None:
    """Prints, for each run, the probe metrics (averaged within the epoch), the epoch's step statistics
    and the kNN accuracy at the given epochs."""
    for run in map(Path, runs):
        print(f"## {run}")
        probes = load_rows(run / "diagnostics.jsonl") if (run / "diagnostics.jsonl").exists() else []
        steps = {r["epoch"]: r for r in load_rows(run / "diag_epochs.jsonl")} if (run / "diag_epochs.jsonl").exists() else {}
        knn = {r["step"]: r.get("eval/knn_acc1") for r in map(json.loads, open(run / "log.jsonl"))} if (run / "log.jsonl").exists() else {}
        cols = REPORT + ["step/enc_grad_norm_max", "step/enc_grad_norm_spikes", "step/score_grad_norm_p99", "knn"]
        print("epoch " + " ".join(f"{c.split('/')[-1][:12]:>12}" for c in cols))
        for e in epochs:
            sel = [r for r in probes if r["epoch"] == e]
            vals = []
            for c in cols:
                if c == "knn":
                    v = knn.get(e)
                elif c.startswith("step/"):
                    v = steps.get(e, {}).get(c)
                else:
                    v = sum(r[c] for r in sel) / len(sel) if sel else None
                vals.append(f"{v:12.4g}" if v is not None else f"{'-':>12}")
            print(f"{e:5d} " + " ".join(vals))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--every", type=int, default=100, help="probe every this many steps")
    p.add_argument("--probe_size", type=int, default=2048, help="images of the probe set")
    p.add_argument("overrides", nargs="*")
    p = sub.add_parser("twin")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--deterministic", action="store_true", help="deterministic cuDNN / cuBLAS kernels")
    p.add_argument("--perturb", type=float, default=0.0, help="relative perturbation of the second copy")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("overrides", nargs="*")
    p = sub.add_parser("report")
    p.add_argument("runs", nargs="+")
    p.add_argument("--epochs", default="9,19,49,99,199,299,499,999")
    args = parser.parse_args(argv)

    if args.cmd == "train":
        train(load_config(args.config, args.overrides), args.every, args.probe_size)
    elif args.cmd == "twin":
        if args.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # read when cuBLAS starts
            torch.use_deterministic_algorithms(True, warn_only=True)  # the twin distance shows what is left
            torch.backends.cudnn.deterministic = True
        twin(load_config(args.config, args.overrides), Path(args.out), args.epochs, args.perturb, args.log_every)
    else:
        report(args.runs, [int(e) for e in args.epochs.split(",")])


if __name__ == "__main__":
    main()
