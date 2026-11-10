"""Barlow Twins (Zbontar et al., 2021): the cross-correlation matrix of the two views' batch-standardized
codes is pulled to the identity; off-diagonal terms are weighted by `lamb`, the total by `scale_loss`.

The loss is solo-learn's (solo/losses/barlow.py, v1.0.6), unchanged except that it is computed in fp32.
As there, each process standardizes its local batch and the cross-correlation is averaged across processes.

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

from scoreclr.methods.base import SSLMethod


def barlow_loss_func(z1: torch.Tensor, z2: torch.Tensor, lamb: float = 5e-3, scale_loss: float = 0.025) -> torch.Tensor:
    """Barlow Twins loss of the codes z1, z2 (N x D) of the two views."""
    N, D = z1.size()

    # to match the original code (BatchNorm1d without affine, training mode: batch statistics)
    bn = torch.nn.BatchNorm1d(D, affine=False).to(z1.device)
    z1 = bn(z1)
    z2 = bn(z2)

    corr = torch.einsum("bi, bj -> ij", z1, z2) / N

    if tdist.is_available() and tdist.is_initialized():
        tdist.all_reduce(corr)
        world_size = tdist.get_world_size()
        corr /= world_size

    diag = torch.eye(D, device=corr.device)
    cdif = (corr - diag).pow(2)
    cdif[~diag.bool()] *= lamb
    loss = scale_loss * cdif.sum()
    return loss


class BarlowTwins(SSLMethod):
    """The shared encoder trained with the Barlow Twins loss on its codes."""

    def __init__(self, cfg, num_classes: int, small_images: bool):
        super().__init__(cfg, num_classes, small_images)
        self.lamb, self.scale_loss = cfg.loss.lamb, cfg.loss.scale_loss

    def loss(self, z0, z1, targets=None) -> Dict[str, torch.Tensor]:
        with torch.autocast(z0.device.type, enabled=False):
            loss = barlow_loss_func(z0.float(), z1.float(), self.lamb, self.scale_loss)
        return {"loss": loss, "barlow": loss.detach()}
