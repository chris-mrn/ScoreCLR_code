"""Optimizers and learning-rate schedules.

  encoder    AdamW, or LARS (ported from solo-learn's solo/utils/lars.py, without the bookkeeping line
             that stored the effective lr for logging)
  auxiliary  AdamW on the method's auxiliary network (ScoreCLR's score net), if any
  schedule   cosine_tail: constant, then cosine to 0 over the last optim.cosine_decay_epochs epochs
             warmup_cosine: linear warm-up from optim.warmup_start_lr over optim.warmup_epochs, then
             cosine to 0, per step and for every param group (solo-learn's warmup_cosine)

LARS:

Copyright 2023 solo-learn development team.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import math
from typing import Iterable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer, required


class LARS(Optimizer):
    """SGD with LARS layer-wise trust ratio (You et al., 2017). With `exclude_bias_n_norm`, 1-D
    parameters (biases, norm layers) get neither the trust ratio nor weight decay."""

    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False,
                 eta=1e-3, eps=1e-8, clip_lr=False, exclude_bias_n_norm=False):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                        nesterov=nesterov, eta=eta, eps=eps, clip_lr=clip_lr,
                        exclude_bias_n_norm=exclude_bias_n_norm)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # lars scaling + weight decay part
                if p.ndim != 1 or not group["exclude_bias_n_norm"]:
                    if p_norm != 0 and g_norm != 0:
                        lars_lr = p_norm / (g_norm + p_norm * weight_decay + group["eps"])
                        lars_lr *= group["eta"]

                        # clip lr
                        if group["clip_lr"]:
                            lars_lr = min(lars_lr / group["lr"], 1)

                        d_p = d_p.add(p, alpha=weight_decay)
                        d_p *= lars_lr

                # sgd part
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group["lr"])

        return loss


def build_optimizers(method, cfg) -> Tuple[Optimizer, Optional[Optimizer]]:
    """(encoder optimizer, auxiliary-network optimizer or None) of a method (scoreclr.methods)."""
    o = cfg.optim
    groups = method.encoder.param_groups(o.lr, o.weight_decay, o.classifier_lr)
    groups[0]["params"] += list(method.encoder_extra_parameters())  # e.g. BYOL's predictor
    if o.get("name", "adamw") == "lars":
        enc_opt = LARS(groups, lr=o.lr, weight_decay=o.weight_decay, **o.lars)
    else:
        enc_opt = torch.optim.AdamW(groups)
    aux = list(method.auxiliary_parameters())
    aux_opt = torch.optim.AdamW(aux, lr=o.score_lr, weight_decay=o.score_weight_decay) if aux else None
    return enc_opt, aux_opt


def cosine_tail_factor(epoch: float, total_epochs: int, decay_epochs: int) -> float:
    """1 until the last `decay_epochs` epochs, then cosine to 0 (`epoch` may be fractional)."""
    if decay_epochs <= 0:
        return 1.0
    t = (epoch - (total_epochs - decay_epochs)) / decay_epochs
    return 0.5 * (1 + math.cos(math.pi * min(max(t, 0.0), 1.0)))


def warmup_cosine_lr(step: int, warmup_steps: float, total_steps: float, base_lr: float,
                     start_lr: float, min_lr: float = 0.0) -> float:
    """lr at optimizer step `step` (0-based) of a linear warm-up from `start_lr` to `base_lr` over
    `warmup_steps` steps followed by a cosine to `min_lr` at `total_steps`: the closed form of
    solo-learn's LinearWarmupCosineAnnealingLR stepped every batch."""
    if step < warmup_steps:
        return start_lr + step * (base_lr - start_lr) / (warmup_steps - 1)
    t = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def set_warmup_cosine_lr(optimizer: Optimizer, step: int, warmup_steps: float, total_steps: float,
                         start_lr: float) -> None:
    """Every param group's lr from warmup_cosine_lr of its initial lr (same start lr for all groups)."""
    for group in optimizer.param_groups:
        group.setdefault("base_lr", group["lr"])
        group["lr"] = warmup_cosine_lr(step, warmup_steps, total_steps, group["base_lr"], start_lr)


def scale_lr(optimizers: Iterable[Optimizer], factor: float) -> None:
    """Sets every param group's lr to its initial lr times `factor`."""
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group.setdefault("base_lr", group["lr"])
            group["lr"] = group["base_lr"] * factor


def set_lr(optimizers, optim_cfg, epoch: int, i: int, steps_per_epoch: int) -> float:
    """Applies the schedule at step i of `epoch` to every optimizer; returns the encoder lr factor."""
    if optim_cfg.get("schedule", "cosine_tail") == "warmup_cosine":
        for opt in optimizers:
            set_warmup_cosine_lr(opt, epoch * steps_per_epoch + i, optim_cfg.warmup_epochs * steps_per_epoch,
                                 optim_cfg.epochs * steps_per_epoch, optim_cfg.warmup_start_lr)
        group = optimizers[0].param_groups[0]
        return group["lr"] / group["base_lr"]
    factor = cosine_tail_factor(epoch + i / steps_per_epoch, optim_cfg.epochs, optim_cfg.cosine_decay_epochs)
    scale_lr(optimizers, factor)
    return factor
