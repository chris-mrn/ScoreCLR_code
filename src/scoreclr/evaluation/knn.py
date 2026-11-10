"""Weighted kNN (k = 20, temperature 0.07, cosine; Wu et al., 2018) on frozen backbone features, the
same protocol for every method: the memory bank is the train set under the test transform (no
augmentation), the queries the test set."""

from typing import Tuple

import torch
import torch.nn.functional as F

from scoreclr.data import build_dataset, build_loader, eval_transforms

KNN_K, KNN_T = 20, 0.07


@torch.no_grad()
def extract_features(backbone, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    feats, labels = [], []
    for x, y in loader:
        feats.append(backbone(x.to(device, non_blocking=True)))
        labels.append(y.to(device))
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def weighted_knn(
    train_f: torch.Tensor, train_y: torch.Tensor, test_f: torch.Tensor, test_y: torch.Tensor,
    k: int = 20, t: float = 0.07, chunk: int = 512,
) -> Tuple[float, float]:
    """Cosine kNN, neighbours weighted by exp(similarity / t) (Wu et al., 2018). Returns top-1/top-5 (%)."""
    train_f, test_f = F.normalize(train_f, dim=1), F.normalize(test_f, dim=1)
    num_classes = int(max(train_y.max(), test_y.max())) + 1
    k = min(k, train_f.size(0))
    top1 = top5 = 0
    for i in range(0, test_f.size(0), chunk):
        sim, idx = (test_f[i : i + chunk] @ train_f.T).topk(k, dim=1)
        votes = torch.zeros(sim.size(0), num_classes, device=sim.device)
        votes.scatter_add_(1, train_y[idx], (sim / t).exp())
        pred = votes.topk(min(5, num_classes), dim=1).indices
        target = test_y[i : i + chunk, None]
        top1 += (pred[:, :1] == target).sum().item()
        top5 += (pred == target).any(dim=1).sum().item()
    n = test_f.size(0)
    return 100.0 * top1 / n, 100.0 * top5 / n


@torch.no_grad()
def monitor(method, cfg, device, batch_size: int = 256) -> dict:
    """Online-classifier accuracy and kNN accuracy (%) of a method's encoder."""
    _, test_tf = eval_transforms(cfg.data.dataset, cfg.data.crop_size)
    loader = lambda train: build_loader(build_dataset(cfg.data, train=train, transform=test_tf), batch_size,  # noqa: E731
                                        train=False, num_workers=cfg.data.num_workers)
    was_training = method.training
    method.eval()
    encoder = method.encoder
    train_f, train_y = extract_features(encoder.backbone, loader(True), device)
    test_f, test_y = extract_features(encoder.backbone, loader(False), device)
    online = (encoder.classifier(test_f).argmax(1) == test_y).float().mean().item() * 100
    knn1, knn5 = weighted_knn(train_f, train_y, test_f, test_y, KNN_K, KNN_T)
    method.train(was_training)
    return {"eval/online_acc1": online, "eval/knn_acc1": knn1, "eval/knn_acc5": knn5}
