import torch

from scoreclr.evaluation import weighted_knn
from scoreclr.train.optim import cosine_tail_factor
from scoreclr.utils.config import apply_overrides


def test_knn_separable():
    centers = torch.eye(3) * 10
    train_y = torch.arange(3).repeat(20)
    test_y = torch.arange(3).repeat(5)
    train_f = centers[train_y] + torch.randn(60, 3) * 0.1
    test_f = centers[test_y] + torch.randn(15, 3) * 0.1
    top1, top5 = weighted_knn(train_f, train_y, test_f, test_y, k=5)
    assert top1 == 100.0 and top5 == 100.0


def test_overrides():
    cfg = apply_overrides({"optim": {"lr": 1.0}}, ["optim.lr=1e-4", "eval.milestones=[60, 80]", "wandb.enabled=true"])
    assert cfg["optim"]["lr"] == 1e-4 and cfg["eval"]["milestones"] == [60, 80] and cfg["wandb"]["enabled"] is True


def test_cosine_tail():
    assert cosine_tail_factor(50, 100, 0) == 1.0
    assert cosine_tail_factor(80, 100, 20) == 1.0
    assert abs(cosine_tail_factor(90, 100, 20) - 0.5) < 1e-9
    assert cosine_tail_factor(100, 100, 20) == 0.0
