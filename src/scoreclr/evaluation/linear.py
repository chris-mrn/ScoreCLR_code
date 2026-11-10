"""Offline linear probe, the same protocol for every method: a linear classifier trained for 100 epochs
with SGD (momentum 0.9, no weight decay, batch 256) on the frozen eval-mode backbone, with
random-resized-crop + flip augmentation, then scored on the test split.

  small images (CIFAR)  raw features, lr 0.1, divided by 10 at epochs 60 and 80
  ImageNet-100          L2-normalized features, lr 10, linear warm-up from 3e-5 over 1 epoch then
                        cosine to 0, per step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scoreclr.data import build_dataset, build_loader, eval_transforms, num_classes
from scoreclr.train.optim import set_warmup_cosine_lr

PROBES = {
    "small": dict(lr=0.1, schedule="step", milestones=[60, 80], feature_norm="none"),
    "imagenet": dict(lr=10.0, schedule="warmup_cosine", warmup_epochs=1, warmup_start_lr=3e-5, feature_norm="l2"),
}


def probe_protocol(data_cfg) -> dict:
    return PROBES["small" if data_cfg.crop_size <= 64 else "imagenet"]


def linear_probe(backbone, data_cfg, device, epochs: int = 100, batch_size: int = 256, num_workers: int = 8) -> dict:
    """Top-1 / top-5 test accuracy (%) of the linear probe on `backbone`."""
    p = probe_protocol(data_cfg)
    train_tf, test_tf = eval_transforms(data_cfg.dataset, data_cfg.crop_size)
    train_loader = build_loader(build_dataset(data_cfg, train=True, transform=train_tf), batch_size, train=True,
                                num_workers=num_workers)
    test_loader = build_loader(build_dataset(data_cfg, train=False, transform=test_tf), batch_size, train=False,
                               num_workers=num_workers)
    backbone.eval()
    head = nn.Linear(backbone.out_dim, num_classes(data_cfg)).to(device)
    opt = torch.optim.SGD(head.parameters(), lr=p["lr"], momentum=0.9, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, p["milestones"], gamma=0.1) if p["schedule"] == "step" else None
    steps = len(train_loader)

    def logits(x):
        with torch.no_grad():
            f = backbone(x.to(device, non_blocking=True))
        return head(F.normalize(f, dim=1) if p["feature_norm"] == "l2" else f)

    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            if sched is None:
                set_warmup_cosine_lr(opt, epoch * steps + i, p["warmup_epochs"] * steps, epochs * steps,
                                     p["warmup_start_lr"])
            loss = F.cross_entropy(logits(x), y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()

    correct1 = correct5 = n = 0
    with torch.no_grad():
        for x, y in test_loader:
            y = y.to(device)
            pred = logits(x).topk(min(5, head.out_features), dim=1).indices
            correct1 += (pred[:, 0] == y).sum().item()
            correct5 += (pred == y[:, None]).any(1).sum().item()
            n += y.size(0)
    return {"linear_acc1": 100.0 * correct1 / n, "linear_acc5": 100.0 * correct5 / n}
