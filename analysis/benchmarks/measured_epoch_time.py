"""Wall-clock time per epoch of one method, with the training code itself (setup, train_one_epoch) and
its config, without evaluation, checkpointing or logging:

    [torchrun --nproc_per_node=2] analysis/benchmarks/measured_epoch_time.py --config config/cifar10_scoreclr.yaml \
        --epochs 13 --out scoreclr.json [key=value ...]

Writes the time and the peak allocated memory (max over processes) of every epoch; report the median
over the epochs after the first few (warm-up, page cache of the data).
"""
import argparse
import json
import time

import torch
import torch.distributed as tdist
from torch.utils.data.distributed import DistributedSampler

from scoreclr.data import TwoViews, build_dataset, build_loader, pretrain_transform
from scoreclr.train.train import setup, train_one_epoch
from scoreclr.utils import distributed as dist
from scoreclr.utils import load_config, seed_everything

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--epochs", type=int, default=13)
p.add_argument("--out", required=True)
p.add_argument("overrides", nargs="*")
args = p.parse_args()

cfg = load_config(args.config, args.overrides)
device = dist.init_distributed()
seed_everything(cfg.seed + dist.rank())
view = pretrain_transform(cfg.data.dataset, cfg.data.crop_size, cfg.data.augment)
train_set = build_dataset(cfg.data, train=True, transform=TwoViews(view))
sampler = DistributedSampler(train_set, shuffle=True, drop_last=True) if dist.is_distributed() else None
loader = build_loader(train_set, cfg.optim.batch_size, train=True, num_workers=cfg.data.num_workers, sampler=sampler)
method, encoder, enc_opt, aux_opt = setup(cfg, device)
scaler = torch.amp.GradScaler(device.type) if cfg.optim.get("precision", "fp32") == "fp16" else None

times, peaks = [], []
for epoch in range(args.epochs):
    if sampler is not None:
        sampler.set_epoch(epoch)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    dist.barrier()
    t0 = time.perf_counter()
    train_one_epoch(method, encoder, enc_opt, aux_opt, loader, cfg, epoch, device, scaler)
    torch.cuda.synchronize(device)
    dist.barrier()
    times.append(time.perf_counter() - t0)
    peak = torch.tensor(float(torch.cuda.max_memory_allocated(device)), device=device)
    if dist.is_distributed():
        tdist.all_reduce(peak, op=tdist.ReduceOp.MAX)
    peaks.append(peak.item())
    if dist.is_main():
        print(f"{cfg.method} epoch {epoch} {times[-1]:.2f} s  peak {peaks[-1] / 2**30:.2f} GiB", flush=True)

if dist.is_main():
    json.dump({"config": args.config, "method": cfg.method, "iters_per_epoch": len(loader),
               "world_size": dist.world_size(), "gpu": torch.cuda.get_device_name(device),
               "batch_per_gpu": cfg.optim.batch_size, "num_workers": cfg.data.num_workers,
               "epoch_s": times, "peak_bytes": peaks}, open(args.out, "w"), indent=1)
dist.cleanup()
