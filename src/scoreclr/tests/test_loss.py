import torch

from scoreclr.models import ScoreNet
from scoreclr.methods.scoreclr import dsm_loss, sample_sigma, scoreclr_loss


def test_repulsion_gradient_is_the_score():
    torch.manual_seed(0)
    net = ScoreNet(8, hidden_dim=16, time_emb_dim=8)
    z0, z1 = torch.randn(4, 8, requires_grad=True), torch.randn(4, 8, requires_grad=True)
    torch.manual_seed(1)
    _, rep, _ = scoreclr_loss(net, z0, z1, attr_weight=1.0, p_mean=-1.2, p_std=1.2)
    g0, g1 = torch.autograd.grad(rep, (z0, z1))

    torch.manual_seed(1)
    z = torch.cat([z0, z1]).detach()
    sigma = sample_sigma(8, -1.2, 1.2, z.device)
    z_t = z + sigma[:, None] * torch.randn_like(z)
    expected = net.score(z_t, sigma) / (8 * 8)  # batch mean and / dim
    assert torch.allclose(torch.cat([g0, g1]), expected, atol=1e-6)


def test_encoder_loss_leaves_score_net_untouched():
    net = ScoreNet(8, hidden_dim=16, time_emb_dim=8)
    z0, z1 = torch.randn(4, 8, requires_grad=True), torch.randn(4, 8, requires_grad=True)
    sum(scoreclr_loss(net, z0, z1, 5.0, -1.2, 1.2)).backward()
    assert z0.grad is not None and all(p.grad is None for p in net.parameters())


def test_dsm_loss_detaches_codes():
    net = ScoreNet(8, hidden_dim=16, time_emb_dim=8)
    z = torch.randn(16, 8, requires_grad=True)
    dsm_loss(net, z, -1.2, 1.2).backward()
    assert z.grad is None and net.fc3.weight.grad is not None


class _StubScoreNet:
    """score(z_t, sigma) = fn(z_t); records the sigmas it was queried at."""

    def __init__(self, fn):
        self.fn, self.sigmas = fn, []

    def score(self, z_t, sigma):
        self.sigmas.append(sigma)
        return self.fn(z_t)


def test_queries_are_averaged():
    net = _StubScoreNet(lambda z_t: -z_t)
    z0, z1 = torch.randn(4, 8), torch.randn(4, 8)
    torch.manual_seed(2)
    _, rep, prior = scoreclr_loss(net, z0, z1, 1.0, -1.2, 1.2, num_queries=3)
    assert len(net.sigmas) == 3

    torch.manual_seed(2)
    z = torch.cat([z0, z1])
    z_ts = []
    for _ in range(3):
        sigma = sample_sigma(8, -1.2, 1.2, z.device)
        z_ts.append(z + sigma[:, None] * torch.randn_like(z))
    mean_score = -torch.stack(z_ts).mean(0)
    assert torch.allclose(rep, 0.5 * mean_score.pow(2).sum(dim=1).mean() / 8)
    assert torch.allclose(prior, torch.stack([0.5 * zt.pow(2).sum(dim=1).mean() / 8 for zt in z_ts]).mean())


def test_single_query_is_the_default():
    net = ScoreNet(8, hidden_dim=16, time_emb_dim=8)
    z0, z1 = torch.randn(4, 8), torch.randn(4, 8)
    torch.manual_seed(3)
    a = scoreclr_loss(net, z0, z1, 5.0, -1.2, 1.2)
    torch.manual_seed(3)
    b = scoreclr_loss(net, z0, z1, 5.0, -1.2, 1.2, num_queries=1)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
