import json

import pytest
import torch
import torch.nn.functional as F

from scoreclr.methods import BYOL, METHODS, BarlowTwins, LeJEPA, ScoreCLR, SimCLR, SSLMethod
from scoreclr.methods.barlow import barlow_loss_func
from scoreclr.methods.byol import byol_loss_func, cosine_tau
from scoreclr.methods.lejepa import SIGReg, lejepa_loss_func
from scoreclr.methods.simclr import simclr_loss_func
from scoreclr.train.train import train


def test_simclr_loss_matches_nt_xent():
    torch.manual_seed(0)
    n, d, tau = 8, 16, 0.2
    z0, z1 = torch.randn(n, d), torch.randn(n, d)
    z = F.normalize(torch.cat([z0, z1]), dim=1)
    logits = (z @ z.T / tau).masked_fill(torch.eye(2 * n, dtype=torch.bool), float("-inf"))
    ref = F.cross_entropy(logits, torch.cat([torch.arange(n, 2 * n), torch.arange(n)]))
    idx = torch.arange(n).repeat(2)
    assert torch.allclose(simclr_loss_func(torch.cat([z0, z1]), idx, tau), ref, atol=1e-5)


def test_sigreg_is_small_for_gaussian_codes():
    torch.manual_seed(0)
    sigreg = SIGReg()
    gauss = sigreg(torch.randn(2, 2048, 32))
    shifted = sigreg(torch.randn(2, 2048, 32) * 0.3 + 1.0)
    assert gauss < 2.0 < shifted


def test_lejepa_loss_mixes_invariance_and_sigreg():
    torch.manual_seed(0)
    proj = torch.randn(2, 64, 16, requires_grad=True)
    total, inv, sig = lejepa_loss_func(proj, SIGReg(), lamb=0.02)
    assert torch.isclose(inv, (proj.mean(0) - proj).square().mean().detach())
    total.backward()
    assert proj.grad is not None and torch.isfinite(proj.grad).all()


def test_warmup_cosine_lr_endpoints():
    from scoreclr.train.optim import warmup_cosine_lr
    lr = lambda s: warmup_cosine_lr(s, 10, 100, base_lr=0.4, start_lr=3e-5)  # noqa: E731
    assert lr(0) == pytest.approx(3e-5)
    assert lr(9) == pytest.approx(0.4) and lr(10) == pytest.approx(0.4)
    assert lr(55) == pytest.approx(0.2) and lr(100) == pytest.approx(0.0, abs=1e-12)


def test_lars_step_matches_formula():
    from scoreclr.train.optim import LARS
    torch.manual_seed(0)
    w, b = torch.randn(4, 3, requires_grad=True), torch.randn(4, requires_grad=True)
    w.grad, b.grad = torch.randn(4, 3), torch.randn(4)
    w0, b0, gw, gb = w.detach().clone(), b.detach().clone(), w.grad.clone(), b.grad.clone()
    lr, wd, eta = 0.4, 1e-4, 0.02
    LARS([w, b], lr=lr, weight_decay=wd, momentum=0.9, eta=eta, clip_lr=True, exclude_bias_n_norm=True).step()
    trust = min(eta * w0.norm() / (gw.norm() + w0.norm() * wd + 1e-8) / lr, 1)
    assert torch.allclose(w.detach(), w0 - lr * trust * (gw + wd * w0))
    assert torch.allclose(b.detach(), b0 - lr * gb)  # 1-D: no trust ratio, no weight decay


@pytest.mark.parametrize("method,extra", [
    ("simclr", ["loss.temperature=0.2"]),
    ("lejepa", ["loss.lamb=0.02", "loss.sigreg_knots=17", "loss.sigreg_t_max=3.0", "loss.sigreg_num_slices=16"]),
    ("simclr", ["loss.temperature=0.2", "optim.name=lars", "optim.lr=0.4", "optim.weight_decay=1e-4",
                "optim.lars={momentum: 0.9, eta: 0.02, clip_lr: true, exclude_bias_n_norm: true}",
                "optim.schedule=warmup_cosine", "optim.warmup_epochs=1", "optim.warmup_start_lr=3e-5",
                "optim.grad_clip=null"]),
    ("byol", ["model.pred_hidden_dim=32", "model.pred_batchnorm=false", "loss.base_tau=0.99", "loss.final_tau=1.0"]),
    ("byol", ["model.pred_hidden_dim=32", "model.pred_batchnorm=true", "loss.base_tau=0.99", "loss.final_tau=1.0"]),
    ("barlow", ["loss.lamb=0.0051", "loss.scale_loss=0.1"]),
])
def test_train_baseline(tiny_cfg, method, extra):
    from scoreclr.utils.config import apply_overrides
    cfg = type(tiny_cfg)(apply_overrides(tiny_cfg.to_dict(), [f"method={method}", "optim.grad_clip=1.0", *extra]))
    metrics = train(cfg)
    assert "eval/knn_acc1" in metrics
    logged = [json.loads(line) for line in open(f"{cfg.output_dir}/log.jsonl")]
    key = {"simclr": "train/nce", "lejepa": "train/sigreg", "byol": "train/byol", "barlow": "train/barlow"}[method]
    assert all(key in r for r in logged if "train/class" in r)



def test_every_method_shares_the_encoder_structure():
    assert all(issubclass(m, SSLMethod) for m in METHODS.values())
    assert set(METHODS) == {"scoreclr", "simclr", "lejepa", "byol", "barlow"}
    assert list(ScoreCLR.__mro__).index(SSLMethod) == 1
    assert SimCLR.__bases__ == LeJEPA.__bases__ == BYOL.__bases__ == BarlowTwins.__bases__ == (SSLMethod,)


def test_barlow_loss_matches_formula():
    torch.manual_seed(0)
    n, d, lamb, scale = 64, 8, 0.0051, 0.1
    z0, z1 = torch.randn(n, d) * 3 + 1, torch.randn(n, d)
    std = lambda z: (z - z.mean(0)) / torch.sqrt(z.var(0, unbiased=False) + 1e-5)  # noqa: E731
    c = std(z0).T @ std(z1) / n
    off = ~torch.eye(d, dtype=torch.bool)
    ref = scale * ((torch.diagonal(c) - 1).pow(2).sum() + lamb * c[off].pow(2).sum())
    assert torch.allclose(barlow_loss_func(z0, z1, lamb, scale), ref, atol=1e-5)
    # identical views with decorrelated dimensions: the cross-correlation is the identity
    q = torch.linalg.qr(torch.randn(n, d) - 0)[0]
    q = q - q.mean(0)
    assert barlow_loss_func(q, q, lamb, scale) < 1e-3


def test_byol_loss_and_tau_schedule():
    p = torch.randn(4, 8)
    assert torch.allclose(byol_loss_func(p, 2 * p), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(byol_loss_func(p, -p), torch.tensor(4.0), atol=1e-6)
    assert cosine_tau(0, 100, 0.99, 1.0) == pytest.approx(0.99)
    assert cosine_tau(50, 100, 0.99, 1.0) == pytest.approx(0.995)
    assert cosine_tau(100, 100, 0.99, 1.0) == pytest.approx(1.0)


def test_byol_target_is_an_ema_of_the_online_network(tiny_cfg):
    from scoreclr.train.optim import build_optimizers
    from scoreclr.utils.config import apply_overrides
    cfg = type(tiny_cfg)(apply_overrides(tiny_cfg.to_dict(), ["method=byol", "model.pred_hidden_dim=32",
                                                              "loss.base_tau=0.9", "loss.final_tau=1.0"]))
    m = BYOL(cfg, 10, True)
    online = list(m.encoder.backbone.parameters()) + list(m.encoder.projector.parameters())
    target = list(m.target["backbone"].parameters()) + list(m.target["projector"].parameters())
    assert all(torch.equal(a, b) and not b.requires_grad for a, b in zip(online, target))
    enc_opt, aux_opt = build_optimizers(m, cfg)
    pred = {id(p) for p in m.predictor.parameters()}
    assert pred <= {id(p) for p in enc_opt.param_groups[0]["params"]} and aux_opt is None
    with torch.no_grad():
        for p in online:
            p.add_(1.0)
    before = [t.clone() for t in target]
    m.after_step(0, 10)  # tau(0) = base_tau
    assert all(torch.allclose(t, 0.9 * b + 0.1 * o) for t, b, o in zip(target, before, online))
    x = torch.randn(4, 3, 16, 16)
    t0, t1 = m.target_codes([x, x])
    assert not t0.requires_grad and t0.shape == (4, cfg.model.proj_dim)
    _, (z0, z1), _ = m.encoder([x, x])
    out = m.loss(z0, z1, [t0, t1])
    out["loss"].backward()
    assert all(p.grad is not None for p in m.predictor.parameters())
    assert all(t.grad is None for t in target)
    assert any(k.startswith("target.") for k in m.state_dict())  # checkpointed, so a resumed run keeps it
