import math
import torch
import torch.nn as nn


class SinusoidalLogTimeEmbedding(nn.Module):
    """Sinusoidal embedding of log(sigma)."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        self.register_buffer("freqs", torch.exp(-math.log(10000) * torch.arange(half).float() / half))

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        args = torch.log(sigma).unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ScoreNet(nn.Module):
    """Noise-prediction network eps_hat(z_t, sigma), z_t = z + sigma * eps.

    3-layer MLP on [z_t, emb(sigma)], each hidden layer FiLM-modulated by emb(sigma)
    (zero-initialized, so the modulation starts as the identity). score = -eps_hat / sigma.
    """

    def __init__(self, dim: int, hidden_dim: int = 256, time_emb_dim: int = 64):
        super().__init__()
        self.time_embed = SinusoidalLogTimeEmbedding(time_emb_dim)
        self.fc1 = nn.Linear(dim + time_emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dim)
        self.film1 = nn.Linear(time_emb_dim, 2 * hidden_dim)
        self.film2 = nn.Linear(time_emb_dim, 2 * hidden_dim)
        self.act = nn.SiLU()
        for film in (self.film1, self.film2):
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)

    def forward(self, z_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        emb = self.time_embed(sigma)
        g1, b1 = self.film1(emb).chunk(2, dim=-1)
        g2, b2 = self.film2(emb).chunk(2, dim=-1)
        h = (1 + g1) * self.act(self.fc1(torch.cat([z_t, emb], dim=-1))) + b1
        h = (1 + g2) * self.act(self.fc2(h)) + b2
        return self.fc3(h)

    def score(self, z_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return -self.forward(z_t, sigma) / sigma.unsqueeze(-1)
