import torch

from scoreclr.methods import ScoreCLR
from scoreclr.models import resnet
from scoreclr.utils import Config


def test_backbones():
    assert resnet("resnet18", small_images=True)(torch.randn(2, 3, 32, 32)).shape == (2, 512)
    assert resnet("resnet18", small_images=False)(torch.randn(2, 3, 64, 64)).shape == (2, 512)


def test_encoder_outputs_and_param_groups():
    cfg = Config(dict(model=dict(backbone="resnet18", proj_hidden_dim=64, proj_dim=16, score_hidden_dim=32, score_time_emb_dim=8),
                      loss=dict(attr_weight=5.0, p_mean=-1.2, p_std=1.2), optim=dict(score_steps=1)))
    model = ScoreCLR(cfg, num_classes=10, small_images=True)
    feats, zs, logits = model.encoder([torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)])
    assert [f.shape for f in feats] == [(2, 512)] * 2
    assert [z.shape for z in zs] == [(2, 16)] * 2
    assert [lg.shape for lg in logits] == [(2, 10)] * 2

    sum(lg.sum() for lg in logits).backward()  # the classifier sees detached features
    assert all(p.grad is None for p in model.encoder.backbone.parameters())

    groups = model.encoder.param_groups(1e-4, 1e-6, 1e-2)
    assert sum(len(list(g["params"])) for g in groups) == len(list(model.encoder.parameters()))


def test_no_projector():
    cfg = Config(dict(model=dict(backbone="resnet18", projector=False, score_hidden_dim=32, score_time_emb_dim=8),
                      loss=dict(attr_weight=5.0, p_mean=-1.2, p_std=1.2), optim=dict(score_steps=1)))
    model = ScoreCLR(cfg, num_classes=10, small_images=True)
    feats, zs, _ = model.encoder([torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)])
    assert all(z is f for z, f in zip(zs, feats)) and model.encoder.code_dim == 512
    assert model.score_net(zs[0], torch.ones(2)).shape == (2, 512)


def test_score_grad_clip():
    cfg = Config(dict(model=dict(backbone="resnet18", proj_hidden_dim=32, proj_dim=16, score_hidden_dim=32,
                                 score_time_emb_dim=8),
                      loss=dict(attr_weight=5.0, p_mean=-1.2, p_std=1.2),
                      optim=dict(score_steps=1, score_grad_clip=1e-3)))
    model = ScoreCLR(cfg, num_classes=10, small_images=True)
    before = torch.cat([p.detach().flatten().clone() for p in model.score_net.parameters()])
    opt = torch.optim.SGD(model.score_net.parameters(), lr=1.0)  # the step is the clipped gradient
    terms = model.auxiliary_step(torch.randn(64, 16), opt)
    after = torch.cat([p.detach().flatten() for p in model.score_net.parameters()])
    assert terms["score_grad_norm"] > 1e-3  # logged before clipping
    assert (after - before).norm() <= 1e-3 * (1 + 1e-4)
