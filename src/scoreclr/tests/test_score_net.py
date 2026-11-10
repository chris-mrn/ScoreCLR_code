import torch

from scoreclr.models import ScoreNet
from scoreclr.methods.scoreclr import dsm_loss


def test_learns_the_gaussian_score():
    """For z ~ N(0, I) the true score at noise sigma is -z_t / (1 + sigma^2)."""
    torch.manual_seed(0)
    net = ScoreNet(4, hidden_dim=64, time_emb_dim=16)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(1500):
        loss = dsm_loss(net, torch.randn(256, 4), p_mean=-1.2, p_std=1.2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        for s in (0.3, 1.0, 3.0):
            z_t = torch.randn(2048, 4) * (1 + s**2) ** 0.5
            est = net.score(z_t, torch.full((2048,), s))
            cos = torch.nn.functional.cosine_similarity(est, -z_t / (1 + s**2)).mean()
            assert cos > 0.9, (s, cos)


def test_film_starts_as_identity():
    net = ScoreNet(8, hidden_dim=16, time_emb_dim=8)
    assert all((p == 0).all() for m in (net.film1, net.film2) for p in m.parameters())


def test_net_learns_the_gaussian_score():
    """For z ~ N(0, I) the score of z + sigma * eps is -z_t / (1 + sigma^2): the net must recover its direction."""
    torch.manual_seed(0)
    net = ScoreNet(4, hidden_dim=64, time_emb_dim=16)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(1500):
        loss = dsm_loss(net, torch.randn(256, 4), p_mean=-1.2, p_std=1.2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        for s in (0.3, 1.0, 3.0):
            z_t = torch.randn(2048, 4) * (1 + s**2) ** 0.5
            est = net.score(z_t, torch.full((2048,), s))
            cos = torch.nn.functional.cosine_similarity(est, -z_t / (1 + s**2)).mean()
            assert cos > 0.9, (s, cos)
