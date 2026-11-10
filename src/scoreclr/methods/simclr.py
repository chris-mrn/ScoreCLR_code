"""SimCLR (Chen et al., 2020): NT-Xent over the two views' codes, negatives gathered across processes.

The loss and the gather are solo-learn's (solo/losses/simclr.py, solo/utils/misc.py), unchanged.

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

from typing import Dict

import torch
import torch.distributed as tdist
import torch.nn.functional as F

from scoreclr.methods.base import SSLMethod


def get_rank() -> int:
    return tdist.get_rank() if tdist.is_available() and tdist.is_initialized() else 0


class GatherLayer(torch.autograd.Function):
    """All-gather that supports backward: the gradients of every process are summed and each
    process keeps the slice of its own input."""

    @staticmethod
    def forward(ctx, x):
        if tdist.is_available() and tdist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(tdist.get_world_size())]
            tdist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if tdist.is_available() and tdist.is_initialized():
            all_gradients = torch.stack(grads)
            tdist.all_reduce(all_gradients)
            grad_out = all_gradients[get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.cat(GatherLayer.apply(x), dim=dim)


def simclr_loss_func(z: torch.Tensor, indexes: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """SimCLR (NT-Xent) loss of the codes z ((N * views) x D) of this process against the codes of
    all processes. `indexes` identifies the image of each row: rows with the same index are
    positives (the row itself excluded), all others are negatives."""
    z = F.normalize(z, dim=-1)
    gathered_z = gather(z)

    sim = torch.exp(torch.einsum("if, jf -> ij", z, gathered_z) / temperature)

    gathered_indexes = gather(indexes)

    indexes = indexes.unsqueeze(0)
    gathered_indexes = gathered_indexes.unsqueeze(0)
    # positives
    pos_mask = indexes.t() == gathered_indexes
    pos_mask[:, z.size(0) * get_rank():].fill_diagonal_(0)
    # negatives
    neg_mask = indexes.t() != gathered_indexes

    pos = torch.sum(sim * pos_mask, 1)
    neg = torch.sum(sim * neg_mask, 1)
    return -(torch.mean(torch.log(pos / (pos + neg))))


class SimCLR(SSLMethod):
    """The shared encoder trained with NT-Xent at temperature `loss.temperature`."""

    def __init__(self, cfg, num_classes: int, small_images: bool):
        super().__init__(cfg, num_classes, small_images)
        self.temperature = cfg.loss.temperature

    def loss(self, z0, z1, targets=None) -> Dict[str, torch.Tensor]:
        # rows of the same image share an index, unique across processes (as in solo-learn)
        idx = torch.arange(z0.size(0), device=z0.device) + get_rank() * z0.size(0)
        loss = simclr_loss_func(torch.cat([z0, z1]), idx.repeat(2), self.temperature)
        return {"loss": loss, "nce": loss.detach()}
