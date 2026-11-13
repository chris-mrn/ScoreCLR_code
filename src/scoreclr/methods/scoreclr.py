"""ScoreCLR. With codes z of the two views and z_t = z + sigma * eps:

    score net:  ||eps_hat(z_t, sigma) - eps||^2                    (denoising score matching)
    encoder:    attr_weight * ||z0 - z1||^2 + 0.5 * ||s(z_t)||^2 + 0.5 * ||z_t||^2,   all / dim

The repulsion term is ScoreReg in the paper, the score-based estimate of the entropy term of the mutual
information. It is written so that its gradient w.r.t. z is the frozen score s = -eps_hat / sigma:
descending it moves codes towards lower density. Together with the Gaussian prior term it is a
DMD-style estimate of KL(Z || N(0, I)).
"""

from typing import Dict, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from scoreclr.methods.base import SSLMethod
from scoreclr.models.score_net import ScoreNet


def sample_sigma(n: int, p_mean: float, p_std: float, device) -> torch.Tensor:
    """sigma = exp(u), u ~ N(p_mean, p_std^2)."""
    return (p_mean + p_std * torch.randn(n, device=device)).exp()


def dsm_loss(score_net: ScoreNet, z: torch.Tensor, p_mean: float, p_std: float) -> torch.Tensor:
    """Denoising score matching on detached codes z (N, dim)."""
    z = z.detach()
    sigma = sample_sigma(z.size(0), p_mean, p_std, z.device)
    eps = torch.randn_like(z)
    z_t = z + sigma[:, None] * eps
    pred = score_net(z_t, sigma)
    return (pred - eps).pow(2).mean()


def scoreclr_loss(
    score_net: ScoreNet, z0: torch.Tensor, z1: torch.Tensor, attr_weight: float, p_mean: float, p_std: float,
    num_queries: int = 1, prior_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encoder loss against the frozen score net. Returns (attraction, repulsion, prior).

    The repulsion and prior are averaged over `num_queries` independently noised copies of each code.
    """
    z = torch.cat([z0, z1])
    dim = z.size(1)
    scores, priors = [], []
    for _ in range(num_queries):
        sigma = sample_sigma(z.size(0), p_mean, p_std, z.device)
        z_t = z + sigma[:, None] * torch.randn_like(z)
        with torch.no_grad():
            scores.append(score_net.score(z_t, sigma))
        priors.append(0.5 * z_t.pow(2).sum(dim=1).mean() / dim)
    score = torch.stack(scores).mean(0)

    attr = attr_weight * (z0 - z1).pow(2).sum(dim=1).mean() / dim
    rep = 0.5 * (z - (z - score).detach()).pow(2).sum(dim=1).mean() / dim  # ScoreReg
    prior = prior_weight * torch.stack(priors).mean()
    return attr, rep, prior


class ScoreCLR(SSLMethod):
    """The shared encoder and a score net, trained with `score_steps` DSM updates on the detached codes
    before every encoder update against the frozen (updated) score net."""

    def __init__(self, cfg, num_classes: int, small_images: bool):
        super().__init__(cfg, num_classes, small_images)
        self.score_net = ScoreNet(self.encoder.code_dim, cfg.model.score_hidden_dim, cfg.model.score_time_emb_dim)
        self.attr_weight, self.p_mean, self.p_std = cfg.loss.attr_weight, cfg.loss.p_mean, cfg.loss.p_std
        self.num_queries = cfg.loss.get("num_queries", 1)
        self.prior_weight = cfg.loss.get("prior_weight", 1.0)  # 0 removes the L2 (Gaussian prior) term
        self.score_steps = cfg.optim.score_steps
        self.score_grad_clip = cfg.optim.get("score_grad_clip")  # None: no clipping
        object.__setattr__(self, "_trained_score_net", self.score_net)  # DDP-wrapped by distribute, not a submodule

    def distribute(self, device_ids) -> None:
        object.__setattr__(self, "_trained_score_net", DDP(self.score_net, device_ids=device_ids))

    def auxiliary_parameters(self):
        return self.score_net.parameters()

    def auxiliary_step(self, z, optimizer) -> Dict[str, torch.Tensor]:
        params = list(self.score_net.parameters())
        max_norm = float("inf") if self.score_grad_clip is None else self.score_grad_clip
        for _ in range(self.score_steps):
            dsm = dsm_loss(self._trained_score_net, z, self.p_mean, self.p_std)
            optimizer.zero_grad(set_to_none=True)
            dsm.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)  # logged before clipping
            optimizer.step()
        return {"dsm": dsm.detach(), "score_grad_norm": grad_norm}

    def loss(self, z0, z1, targets=None) -> Dict[str, torch.Tensor]:
        attr, rep, prior = scoreclr_loss(self.score_net, z0, z1, self.attr_weight, self.p_mean, self.p_std,
                                         self.num_queries, self.prior_weight)
        return {"loss": attr + rep + prior, "attr": attr.detach(), "rep": rep.detach(), "prior": prior.detach()}
