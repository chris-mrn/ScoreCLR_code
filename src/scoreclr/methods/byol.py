"""BYOL (Grill et al., 2020): a predictor on the online codes regresses, by cosine similarity, the codes
of the other view from a target network, the exponential moving average of the online backbone and
projector. The EMA rate tau follows a cosine from `loss.base_tau` to `loss.final_tau` over training.

The loss, the momentum update and the tau schedule are solo-learn's (solo/losses/byol.py,
solo/utils/momentum.py, solo/methods/byol.py, v1.0.6), with one change for the shared protocol: the
predictor is Linear-ReLU-Linear without BatchNorm like the shared projector, unless
`model.pred_batchnorm` is set (solo-learn's Linear-BN-ReLU-Linear). Under DDP the predictor is applied
once to both views stacked (DDP allows a single forward per step), so with BatchNorm its statistics pool
the two views.

The target network runs in training mode (batch statistics in its BatchNorm layers), as in solo-learn.

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

import copy
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from scoreclr.methods.base import SSLMethod


def byol_loss_func(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """2 - 2 cos(p, z), averaged over the batch; z (the target) is detached."""
    return 2 - 2 * F.cosine_similarity(p, z.detach(), dim=-1).mean()


def cosine_tau(step: int, total_steps: int, base_tau: float, final_tau: float) -> float:
    """solo-learn's MomentumUpdater.update_tau: cosine from base_tau (step 0) to final_tau (step total_steps)."""
    return final_tau - (final_tau - base_tau) * (math.cos(math.pi * step / total_steps) + 1) / 2


class BYOL(SSLMethod):
    """The shared encoder (online), a predictor and an EMA target copy of the backbone and projector."""

    def __init__(self, cfg, num_classes: int, small_images: bool):
        super().__init__(cfg, num_classes, small_images)
        d, h = self.encoder.code_dim, cfg.model.pred_hidden_dim
        norm = [nn.BatchNorm1d(h)] if cfg.model.get("pred_batchnorm", False) else []
        self.predictor = nn.Sequential(nn.Linear(d, h), *norm, nn.ReLU(), nn.Linear(h, d))
        self.target = nn.ModuleDict({"backbone": copy.deepcopy(self.encoder.backbone),
                                     "projector": copy.deepcopy(self.encoder.projector)})
        self.target.requires_grad_(False)
        self.base_tau, self.final_tau = cfg.loss.base_tau, cfg.loss.final_tau
        self.tau = self.base_tau
        object.__setattr__(self, "_trained_predictor", self.predictor)  # DDP-wrapped by distribute

    def _online(self):
        return [self.encoder.backbone, self.encoder.projector]

    def distribute(self, device_ids) -> None:
        # DDP has broadcast rank 0's encoder by now; start every process's target from it too
        for online, target in zip(self._online(), self.target.values()):
            target.load_state_dict(online.state_dict())
        object.__setattr__(self, "_trained_predictor", DDP(self.predictor, device_ids=device_ids))

    def encoder_extra_parameters(self):
        return self.predictor.parameters()

    @torch.no_grad()
    def target_codes(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        self.target.train()  # batch statistics, as solo-learn's momentum backbone
        return [self.target["projector"](self.target["backbone"](x)) for x in views]

    def loss(self, z0, z1, targets=None) -> Dict[str, torch.Tensor]:
        t0, t1 = targets
        p0, p1 = self._trained_predictor(torch.cat([z0, z1])).chunk(2)
        with torch.autocast(z0.device.type, enabled=False):
            loss = byol_loss_func(p0.float(), t1.float()) + byol_loss_func(p1.float(), t0.float())
        return {"loss": loss, "byol": loss.detach(), "tau": torch.tensor(self.tau)}

    @torch.no_grad()
    def after_step(self, step: int, total_steps: int) -> None:
        """EMA update of the target after update `step`, at tau(step): solo-learn updates the target with
        the current tau, then advances tau to (step + 1). Stateless, so a resumed run continues exactly."""
        self.tau = cosine_tau(step, total_steps, self.base_tau, self.final_tau)
        for online, target in zip(self._online(), self.target.values()):
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.mul_(self.tau).add_(po.detach(), alpha=1 - self.tau)
