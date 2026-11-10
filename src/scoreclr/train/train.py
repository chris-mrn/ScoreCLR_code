"""Training entry point, the same for every method (cfg.method: scoreclr, simclr, lejepa, byol or barlow):

    python -m scoreclr.train --config config/cifar10_scoreclr.yaml [key=value ...]
    torchrun --nproc_per_node=2 -m scoreclr.train --config config/imagenet100_scoreclr.yaml   # DDP

Under DDP, optim.batch_size is per process, BatchNorm is synchronized across processes, and
rank 0 logs, checkpoints and monitors (online classifier and kNN every eval.every epochs).
The final evaluation is `python -m scoreclr.evaluation --run <output_dir>`.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from scoreclr.data import TwoViews, build_dataset, build_loader, pretrain_transform
from scoreclr.evaluation.knn import monitor
from scoreclr.methods import build_method
from scoreclr.train.optim import build_optimizers, set_lr
from scoreclr.utils import Logger, load_checkpoint, load_config, save_checkpoint, seed_everything
from scoreclr.utils import distributed as dist


def setup(cfg, device):
    """The method on `device` (SyncBatchNorm under multi-GPU DDP), its possibly DDP-wrapped encoder,
    and the encoder and auxiliary optimizers."""
    method = build_method(cfg)
    if dist.is_distributed() and device.type == "cuda":
        method.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(method.encoder)
    method.to(device)
    encoder = method.encoder
    if dist.is_distributed():
        ids = [device.index] if device.type == "cuda" else None
        encoder = DDP(method.encoder, device_ids=ids)
        method.distribute(ids)
    enc_opt, aux_opt = build_optimizers(method, cfg)
    return method, encoder, enc_opt, aux_opt


def grad_norm(opt, max_norm=None) -> torch.Tensor:
    """Global gradient norm of `opt`'s parameters before clipping; clips them in place to `max_norm` if set."""
    params = [p for group in opt.param_groups for p in group["params"]]
    return torch.nn.utils.clip_grad_norm_(params, float("inf") if max_norm is None else max_norm)


def train_step(method, encoder, enc_opt, aux_opt, x0, x1, y, cfg, scaler=None) -> dict:
    """One training step, the same for every method: forward both views, the method's auxiliary update
    on the detached codes (ScoreCLR: score_steps DSM steps), then one encoder update on the method's
    loss plus the online classifier's. `scaler` (a GradScaler) enables 16-bit mixed precision for the
    encoder. Returns the logged terms (tensors)."""
    device_type = x0.device.type
    with torch.autocast(device_type, dtype=torch.float16, enabled=scaler is not None):
        _, (z0, z1), logits = encoder([x0, x1])
        class_loss = sum(F.cross_entropy(lg, y) for lg in logits) / len(logits)
        targets = method.target_codes([x0, x1])  # BYOL's EMA encoder; None for the other methods
    z = torch.cat([z0, z1]).detach()
    terms = method.auxiliary_step(z, aux_opt)
    with torch.autocast(device_type, dtype=torch.float16, enabled=scaler is not None):
        terms |= method.loss(z0, z1, targets)
    loss = terms.pop("loss") + class_loss

    enc_opt.zero_grad(set_to_none=True)
    if scaler is None:
        loss.backward()
        enc_gnorm = grad_norm(enc_opt, cfg.optim.get("grad_clip"))
        enc_opt.step()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(enc_opt)
        enc_gnorm = grad_norm(enc_opt, cfg.optim.get("grad_clip"))
        scaler.step(enc_opt)
        scaler.update()
    return terms | {"class": class_loss, "z_std": z.float().std(0).mean(), "enc_grad_norm": enc_gnorm}


def train_one_epoch(method, encoder, enc_opt, aux_opt, loader, cfg, epoch, device, scaler=None) -> dict:
    """One epoch of train_step, with the learning-rate schedule applied every step."""
    encoder.train()
    optimizers = [o for o in (enc_opt, aux_opt) if o is not None]
    totals, n = {}, 0
    for i, ((x0, x1), y) in enumerate(loader):
        lr_factor = set_lr(optimizers, cfg.optim, epoch, i, len(loader))
        x0, x1, y = x0.to(device, non_blocking=True), x1.to(device, non_blocking=True), y.to(device)
        step = train_step(method, encoder, enc_opt, aux_opt, x0, x1, y, cfg, scaler)
        method.after_step(epoch * len(loader) + i, cfg.optim.epochs * len(loader))
        for k, v in step.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n += 1
    metrics = dist.all_reduce_mean({f"train/{k}": v / n for k, v in totals.items()}, device)
    return metrics | {"train/lr_factor": lr_factor}


def train(cfg) -> dict:
    device = dist.init_distributed()
    seed_everything(cfg.seed + dist.rank())  # identical init anyway: DDP broadcasts rank 0's weights
    main = dist.is_main()
    out = Path(cfg.output_dir)
    if main:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    view = pretrain_transform(cfg.data.dataset, cfg.data.crop_size, cfg.data.augment)
    train_set = build_dataset(cfg.data, train=True, transform=TwoViews(view))
    sampler = DistributedSampler(train_set, shuffle=True, drop_last=True) if dist.is_distributed() else None
    loader = build_loader(train_set, cfg.optim.batch_size, train=True, num_workers=cfg.data.num_workers, sampler=sampler)

    method, encoder, enc_opt, aux_opt = setup(cfg, device)
    precision = cfg.optim.get("precision", "fp32")
    assert precision in ("fp32", "fp16"), precision
    scaler = torch.amp.GradScaler(device.type) if precision == "fp16" else None
    start_epoch = 0
    ckpt = load_checkpoint(out / "last.pt")
    if ckpt is not None:
        method.load_state_dict(ckpt["model"])
        enc_opt.load_state_dict(ckpt["enc_opt"])
        if aux_opt is not None:
            aux_opt.load_state_dict(ckpt["aux_opt"])
        if scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        if main:
            print(f"resumed from epoch {ckpt['epoch']}")

    logger = Logger(out, cfg) if main else None
    metrics = {}
    for epoch in range(start_epoch, cfg.optim.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        metrics = train_one_epoch(method, encoder, enc_opt, aux_opt, loader, cfg, epoch, device, scaler)
        if main:
            state = {"model": method.state_dict(), "enc_opt": enc_opt.state_dict(), "epoch": epoch,
                     "config": cfg.to_dict()}
            if aux_opt is not None:
                state["aux_opt"] = aux_opt.state_dict()
            if scaler is not None:
                state["scaler"] = scaler.state_dict()
            save_checkpoint(out / "last.pt", state)
            if (epoch + 1) % cfg.eval.every == 0 or epoch + 1 == cfg.optim.epochs:
                metrics |= monitor(method, cfg, device)
            logger.log(metrics, step=epoch)
        dist.barrier()
    dist.cleanup()
    if main:
        logger.finish()
    return metrics


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="key=value, e.g. optim.epochs=10")
    args = parser.parse_args(argv)
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
