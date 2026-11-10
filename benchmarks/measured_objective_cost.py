"""Measured GPU cost against batch size of the ScoreCLR and SimCLR objectives, with the code used for
training (scoreclr.methods, scoreclr.train.train.train_step):

    python benchmarks/measured_objective_cost.py --out results.jsonl --commit $(git rev-parse --short HEAD)
    python benchmarks/plot_measured_cost.py results.jsonl cost_vs_batch_measured.pdf

One GPU, no data loader, cudnn.benchmark off and default TF32 settings as in training. B is the
number of images (2B codes); time is the median over n_iters CUDA-event-timed steps after warm-up;
mem_extra = peak allocated during the steps minus allocated before them (inputs, weights and
optimizer states excluded), mem_peak = absolute peak.

part "objective": codes z0, z1 (B x 256, leaves) on the GPU; one step = the method's auxiliary update
  (ScoreCLR: S DSM steps of the score net) and the backward of its loss to z0, z1.
part "full": synthetic images on the GPU; one step = train_step (R18 + 2048/256 projector, online
  classifier, objective, AdamW step). "none" = the same step without objective.
part "dp_emulated": the per-device step under data parallelism over W devices, emulated on one GPU:
  the device holds B/W images; ScoreCLR's objective is local; SimCLR's gather is replaced by the
  concatenation with the other devices' codes, a tensor that receives gradients.
"""
import argparse
import contextlib
import json
import statistics
from pathlib import Path

import torch

from scoreclr.methods import SSLMethod, build_method
from scoreclr.methods import simclr as simclr_module
from scoreclr.train.optim import build_optimizers
from scoreclr.train.train import train_step
from scoreclr.utils import load_config

dev = torch.device("cuda")
D = 256
CONFIG = Path(__file__).resolve().parents[1] / "config"
RECIPES = {  # name: (config, overrides)
    "scoreclr_S1_K1": ("cifar10_scoreclr", ["loss.num_queries=1"]),  # CIFAR-10 recipe with one query
    "scoreclr_S1_K3": ("cifar10_scoreclr", []),  # CIFAR-10 row
    "scoreclr_S5_K3": ("imagenet100_scoreclr", []),  # ImageNet-100 row
    "simclr": ("cifar10_simclr_adamw", []),
}


class NoObjective(SSLMethod):
    """The shared encoder step without any objective (baseline of part "full")."""

    def loss(self, z0, z1):
        return {"loss": (z0.sum() + z1.sum()) * 0.0}


def recipe(name, res=32):
    cfg_name, overrides = RECIPES.get(name, ("cifar10_simclr_adamw", []))
    data = ["data.dataset=cifar10", "data.crop_size=32"] if res == 32 else ["data.dataset=imagenet100", "data.crop_size=224"]
    cfg = load_config(CONFIG / f"{cfg_name}.yaml", overrides + data)
    method = NoObjective(cfg, 10 if res == 32 else 100, res == 32) if name == "none" else build_method(cfg)
    method.to(dev)
    enc_opt, aux_opt = build_optimizers(method, cfg)
    return cfg, method, enc_opt, aux_opt


@contextlib.contextmanager
def emulated_gather(others):
    """SimCLR's all-gather replaced by the concatenation with `others`, the other devices' codes."""
    original = simclr_module.gather
    def gather(x):  # codes: append the others; image indexes: new ids n.., from shapes (no host sync)
        if x.is_floating_point():
            return torch.cat([x, others])
        n = x.numel() // 2
        return torch.cat([x, torch.arange(n, n + others.size(0) // 2, device=x.device).repeat(2)])

    simclr_module.gather = gather
    try:
        yield
    finally:
        simclr_module.gather = original


def time_steps(step, warmup, iters):
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        step()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts), statistics.stdev(ts)


def run(step, warmup, iters, **meta):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    row = dict(meta, n_iters=iters, gpu=GPU, commit=args.commit, oom=False)
    try:
        ms, sd = time_steps(step, warmup, iters)
        peak = torch.cuda.max_memory_allocated()
        row |= dict(ms=ms, ms_sd=sd, img_per_s=meta["B_device"] / ms * 1e3, mem_extra=peak - base, mem_peak=peak)
    except torch.cuda.OutOfMemoryError:
        row |= dict(ms=None, ms_sd=None, img_per_s=None, mem_extra=None, mem_peak=None, oom=True)
    torch.cuda.empty_cache()
    OUT.write(json.dumps(row) + "\n")
    OUT.flush()
    res = "OOM" if row["oom"] else f"{row['ms']:9.3f} ms (sd {row['ms_sd']:.3f})  extra {row['mem_extra'] / 2**20:9.1f} MiB  peak {row['mem_peak'] / 2**30:6.2f} GiB"
    print(f"{meta['part']:12s} {meta['method']:15s} {meta['precision']:5s} res={meta.get('res')} B={meta['B']:6d} Bdev={meta['B_device']:6d}  {res}", flush=True)


def meta_of(cfg, name):
    if name.startswith("scoreclr"):
        return dict(S=cfg.optim.score_steps, K=cfg.loss.num_queries)
    return dict(S=None, K=None)


p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--commit", required=True)
p.add_argument("--parts", nargs="+", default=["objective", "full", "dp_emulated"])
args = p.parse_args()
torch.manual_seed(0)
GPU = torch.cuda.get_device_name()
OUT = open(args.out, "a")
print(GPU, torch.__version__, flush=True)

# ---- (A) objective alone, fp32
if "objective" in args.parts:
    for B in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        for name in RECIPES:
            cfg, method, _, aux_opt = recipe(name)
            z0 = torch.randn(B, D, device=dev, requires_grad=True)
            z1 = torch.randn(B, D, device=dev, requires_grad=True)

            def step(method=method, aux_opt=aux_opt, z0=z0, z1=z1):
                z0.grad = z1.grad = None
                method.auxiliary_step(torch.cat([z0, z1]).detach(), aux_opt)
                method.loss(z0, z1)["loss"].backward()

            step()  # optimizer states exist before the memory baseline
            run(step, 10, 30, part="objective", method=name, precision="fp32", res=None, B=B, B_device=B, W=1,
                **meta_of(cfg, name))
            del z0, z1, method, aux_opt


def images(B, res):
    return (torch.randn(B, 3, res, res, device=dev), torch.randn(B, 3, res, res, device=dev),
            torch.randint(0, 10 if res == 32 else 100, (B,), device=dev))


def full_runs(part, res, precision, batches, W=1):
    for B in batches:
        n = B // W
        x0, x1, y = images(n, res)
        for name in ["none", *RECIPES]:
            if res == 224 and name in ("scoreclr_S1_K1", "scoreclr_S1_K3"):
                continue  # ImageNet-100 uses the S=5, K=3 recipe
            if res == 32 and name == "scoreclr_S5_K3":
                continue
            cfg, method, enc_opt, aux_opt = recipe(name, res)
            method.train()
            scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
            others = None
            if W > 1 and name == "simclr":
                others = torch.randn(2 * (B - n), D, device=dev, requires_grad=True)

            def step():
                ctx = emulated_gather(others) if others is not None else contextlib.nullcontext()
                with ctx:
                    train_step(method, method.encoder, enc_opt, aux_opt, x0, x1, y, cfg, scaler)

            run(step, 5, 15, part=part, method=name, precision=precision, res=res, B=B, B_device=n, W=W,
                **meta_of(cfg, name))
            del method, enc_opt, aux_opt
        del x0, x1, y


# ---- (B) full training step without the loader: fp32 at 32 and 224 px, fp16 mixed precision at 32 px
if "full" in args.parts:
    full_runs("full", 32, "fp32", [128, 256, 512, 1024, 2048])
    full_runs("full", 32, "fp16", [128, 256, 512, 1024, 2048])
    full_runs("full", 224, "fp32", [64, 128, 256, 512])

# ---- (C) per-device step under data parallelism, W = 8, emulated on one GPU (fp32)
if "dp_emulated" in args.parts:
    full_runs("dp_emulated", 32, "fp32", [1024, 4096, 16384, 32768], W=8)
    full_runs("dp_emulated", 224, "fp32", [1024, 4096], W=8)
OUT.close()
