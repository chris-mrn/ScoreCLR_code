import copy
import json
import math

import torch

import diagnostics
from scoreclr.train.train import train


def test_diagnosed_training_has_the_same_trajectory(tiny_cfg, tmp_path):
    plain, diag = copy.deepcopy(tiny_cfg), copy.deepcopy(tiny_cfg)
    plain.output_dir, diag.output_dir = str(tmp_path / "plain"), str(tmp_path / "diag")
    train(plain)
    diagnostics.train(diag, every=1, probe_size=8)
    a = torch.load(f"{plain.output_dir}/last.pt", weights_only=False)["model"]
    b = torch.load(f"{diag.output_dir}/last.pt", weights_only=False)["model"]
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)

    steps = 2 * (32 // 8)  # epochs * steps per epoch
    rows = [json.loads(line) for line in open(f"{diag.output_dir}/diagnostics.jsonl")]
    assert [r["step"] for r in rows] == list(range(steps))
    assert all(math.isfinite(v) for r in rows for v in r.values())
    assert {"geom/erank_z", "dsm/ratio_q+0", "lag/gain", "rep/snr", "force/total_bottom"} <= rows[0].keys()
    assert len(open(f"{diag.output_dir}/steps.csv").readlines()) == 1 + steps
    epochs = [json.loads(line) for line in open(f"{diag.output_dir}/diag_epochs.jsonl")]
    assert [e["epoch"] for e in epochs] == [0, 1] and "step/enc_grad_norm_spikes" in epochs[0]
    assert diagnostics.METHODS["scoreclr"] is diagnostics.ScoreCLR  # restored after the run
    diagnostics.report([diag.output_dir], [0, 1])


def test_gaussian_fit_denoiser_is_the_posterior_mean():
    torch.manual_seed(0)
    lam = torch.tensor([4.0, 1.0, 0.25, 0.01])
    z = torch.randn(200_000, 4) * lam.sqrt()
    fit = diagnostics.GaussianFit(z)
    sigma = torch.full((z.size(0),), 0.5)
    eps = torch.randn_like(z)
    mse = (fit.eps_hat(z + sigma[:, None] * eps, sigma) - eps).pow(2).mean(0)
    expected = (lam / (lam + 0.25)).flip(0).sort().values  # Var(eps_i | z_t) = lam_i / (lam_i + sigma^2)
    assert torch.allclose(mse.sort().values, expected, atol=0.01)

    # the prior term (gradient z) shrinks every direction's log-variance at rate 2
    rates = fit.log_var_rates(z, z)
    assert all(abs(r + 2) < 1e-3 for r in rates.values())


def test_twin_copies_stay_identical_unless_perturbed(tiny_cfg, tmp_path):
    same = diagnostics.twin(tiny_cfg, tmp_path / "same.jsonl", epochs=1, log_every=1, probe_size=8)
    assert all(r["encoder"] == 0 and r["codes"] == 0 and r["score_net"] == 0 for r in same)
    apart = diagnostics.twin(tiny_cfg, tmp_path / "apart.jsonl", epochs=1, perturb=1e-3, log_every=1, probe_size=8)
    assert all(r["encoder"] > 0 for r in apart)
