"""LeJEPA (Balestriero & LeCun, 2025): SIGReg (sliced Epps-Pulley test against N(0, I)) plus an
invariance term pulling every view toward the cross-view centroid, mixed by `lamb`.

SIGReg is computed on each process's local batch (not gathered).

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

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scoreclr.methods.base import SSLMethod


class SIGReg(nn.Module):
    """Sliced Epps-Pulley normality test (LeJEPA's SIGReg), written from its definition.

    For each of `num_slices` random unit directions u (redrawn at every call), the 1-D projections
    x = <u, z> are compared with N(0, 1) through their empirical characteristic function:

        T = N * integral over R of |ecf(t) - phi(t)|^2 * phi(t) dt,   phi(t) = exp(-t^2 / 2),

    evaluated by the trapezoid rule on `knots` points of [0, t_max] and doubled (the integrand is even).
    The result is averaged over the slices; it is ~0 for i.i.d. N(0, I) codes.
    """

    def __init__(self, knots: int = 17, t_max: float = 3.0, num_slices: int = 256):
        super().__init__()
        self.num_slices = num_slices
        t = torch.linspace(0.0, t_max, knots)
        self.register_buffer("t", t)
        self.register_buffer("phi", torch.exp(-0.5 * t * t))

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """proj: (..., N, D) codes; returns a scalar."""
        n = proj.size(-2)
        directions = F.normalize(torch.randn(proj.size(-1), self.num_slices, device=proj.device, dtype=proj.dtype), dim=0)
        angle = (proj @ directions).unsqueeze(-1) * self.t  # (..., N, S, K)
        ecf = torch.complex(angle.cos().mean(dim=-3), angle.sin().mean(dim=-3))  # (..., S, K)
        integrand = (ecf - self.phi).abs().square() * self.phi
        return (2 * n * torch.trapezoid(integrand, self.t, dim=-1)).mean()


def lejepa_loss_func(proj: torch.Tensor, sigreg: SIGReg, lamb: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LeJEPA's objective on the (V, N, D) projected codes: lamb * SIGReg + (1 - lamb) * invariance,
    the invariance term pulling every view toward the live cross-view centroid.
    Returns (total_loss, inv_loss, sigreg_loss); the last two are detached, for logging."""
    inv_loss = (proj.mean(dim=0) - proj).square().mean()
    sigreg_loss = sigreg(proj)
    total_loss = sigreg_loss * lamb + inv_loss * (1 - lamb)
    return total_loss, inv_loss.detach(), sigreg_loss.detach()


class LeJEPA(SSLMethod):
    """The shared encoder trained with lamb * SIGReg + (1 - lamb) * invariance on the stacked codes."""

    def __init__(self, cfg, num_classes: int, small_images: bool):
        super().__init__(cfg, num_classes, small_images)
        self.lamb = cfg.loss.lamb
        self.sigreg = SIGReg(cfg.loss.sigreg_knots, cfg.loss.sigreg_t_max, cfg.loss.sigreg_num_slices)

    def loss(self, z0, z1, targets=None) -> Dict[str, torch.Tensor]:
        loss, inv, sig = lejepa_loss_func(torch.stack([z0, z1]), self.sigreg, self.lamb)
        return {"loss": loss, "inv": inv, "sigreg": sig}
