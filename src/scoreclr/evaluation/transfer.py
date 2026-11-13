"""Out-of-domain linear transfer of a frozen backbone (Kornblith et al., 2019), the protocol of the
paper's transfer table. Ported from solo-learn's transfer evaluation; changes: no default dataset roots, no timm import (same
ImageNet constants).

  1. Frozen, eval-mode backbone; every downstream image resized (shorter side) and center-cropped to
     the source resolution (32 px with the CIFAR stem, 224 px for ImageNet-100 backbones) and
     normalized with the source dataset's statistics. No augmentation.
  2. Backbone features L2-normalized.
  3. Multinomial logistic regression, full-batch L-BFGS, L2 penalty chosen on a validation split
     (the dataset's own where one exists -- DTD -- else a stratified 20% hold-out of train), refit on
     train + val at that penalty, scored once on the test split.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
# = timm.data.constants.IMAGENET_DEFAULT_MEAN / IMAGENET_DEFAULT_STD (solo-learn imported them from timm)
IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# target -> (num_classes, has its own val split)
TARGET_INFO: Dict[str, Tuple[int, bool]] = {
    "cifar10": (10, False),
    "cifar100": (100, False),
    "stl10": (10, False),
    "dtd": (47, True),
    "aircraft": (100, True),
}

# torchvision's DTD / FGVCAircraft expect the root that *contains* `dtd/` and `fgvc-aircraft-2013b/`.
# Roots are passed explicitly (no default paths).
DEFAULT_ROOTS: Dict[str, str] = {}

# Default target list per pretraining dataset: everything in TARGET_INFO except the
# source itself (in-domain accuracy is measured by the linear probe).
def default_targets(source_dataset: str) -> List[str]:
    return [t for t in TARGET_INFO if t != source_dataset]


def default_resolution(source_dataset: str) -> int:
    return 32 if source_dataset in ("cifar10", "cifar100") else 224


def default_weight_decays() -> List[float]:
    return [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


# --------------------------------------------------------------------------- data
def build_eval_transform(source_dataset: str, resolution: int) -> transforms.Compose:
    """Deterministic resize (shorter side) + center crop to the source resolution, with the
    source dataset's channel statistics."""
    if source_dataset in ("cifar10", "cifar100"):
        mean, std = CIFAR_MEAN, CIFAR_STD
    else:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    return transforms.Compose(
        [
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_target_datasets(
    target: str, root: str, transform, download: bool = False
) -> Tuple[Dataset, Optional[Dataset], Dataset]:
    """(train, val or None, test) datasets of one transfer target, all with `transform`."""
    from torchvision.datasets import CIFAR10, CIFAR100, DTD, STL10, FGVCAircraft

    if target == "cifar10":
        return (
            CIFAR10(root, train=True, download=download, transform=transform),
            None,
            CIFAR10(root, train=False, download=download, transform=transform),
        )
    if target == "cifar100":
        return (
            CIFAR100(root, train=True, download=download, transform=transform),
            None,
            CIFAR100(root, train=False, download=download, transform=transform),
        )
    if target == "stl10":
        return (
            STL10(root, split="train", download=download, transform=transform),
            None,
            STL10(root, split="test", download=download, transform=transform),
        )
    if target == "dtd":
        # partition 1 of the 10 official splits: 1880 / 1880 / 1880 images.
        return tuple(DTD(root, split=s, download=download, transform=transform) for s in ("train", "val", "test"))
    if target == "aircraft":
        # 100 variants, 3334 / 3333 / 3333 images.
        return tuple(
            FGVCAircraft(root, split=s, download=download, transform=transform) for s in ("train", "val", "test")
        )
    raise ValueError(f"unknown transfer target {target!r}; known: {list(TARGET_INFO)}")


@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    dataset: Dataset,
    device,
    batch_size: int = 256,
    num_workers: int = 4,
    l2_normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backbone features (float32, on CPU) and labels for every image in `dataset`.
    The backbone must already be in eval mode."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    feats, labels = [], []
    for x, y in loader:
        f = backbone(x.to(device, non_blocking=True)).float()
        if l2_normalize:
            f = F.normalize(f, dim=1)
        feats.append(f.cpu())
        labels.append(torch.as_tensor(y))
    return torch.cat(feats), torch.cat(labels)


# --------------------------------------------------------------------------- classifier
def stratified_holdout(labels: torch.Tensor, fraction: float = 0.2, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Index tensors (fit, holdout) with `fraction` of every class held out (at least one
    sample per class is held out when the class has >= 2 samples)."""
    g = torch.Generator().manual_seed(seed)
    fit, hold = [], []
    for c in labels.unique():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]
        n_hold = int(round(fraction * len(idx)))
        if len(idx) >= 2:
            n_hold = min(max(n_hold, 1), len(idx) - 1)
        else:
            n_hold = 0
        hold.append(idx[:n_hold])
        fit.append(idx[n_hold:])
    return torch.cat(fit), torch.cat(hold)


def fit_logistic_regression(
    feats: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    weight_decay: float,
    max_iter: int = 500,
    device=None,
) -> nn.Linear:
    """Multinomial logistic regression (cross-entropy + weight_decay/2 * ||W||^2, bias
    unpenalized) fit with full-batch L-BFGS. Returns the classifier on CPU."""
    device = device or feats.device
    X = feats.to(device)
    y = labels.to(device)
    clf = nn.Linear(X.size(1), num_classes).to(device)
    nn.init.zeros_(clf.weight)
    nn.init.zeros_(clf.bias)
    opt = torch.optim.LBFGS(
        clf.parameters(),
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-6,
        tolerance_change=1e-9,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(clf(X), y)
        if weight_decay > 0:
            loss = loss + 0.5 * weight_decay * clf.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return clf.cpu()


@torch.no_grad()
def topk_accuracy(clf: nn.Linear, feats: torch.Tensor, labels: torch.Tensor, ks=(1, 5)) -> Dict[int, float]:
    logits = clf(feats)
    out = {}
    for k in ks:
        kk = min(k, logits.size(1))
        pred = logits.topk(kk, dim=1).indices
        out[k] = 100.0 * (pred == labels[:, None]).any(dim=1).float().mean().item()
    return out


def evaluate_transfer_features(
    train: Tuple[torch.Tensor, torch.Tensor],
    val: Optional[Tuple[torch.Tensor, torch.Tensor]],
    test: Tuple[torch.Tensor, torch.Tensor],
    num_classes: int,
    weight_decays: Sequence[float],
    device=None,
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> Dict[str, float]:
    """Select the L2 penalty on `val` (or a stratified hold-out of `train` when val is None),
    refit on train+val, score on test."""
    Xtr, ytr = train
    if val is None:
        fit_idx, hold_idx = stratified_holdout(ytr, holdout_fraction, seed)
        Xfit, yfit, Xval, yval = Xtr[fit_idx], ytr[fit_idx], Xtr[hold_idx], ytr[hold_idx]
        Xall, yall = Xtr, ytr
    else:
        Xfit, yfit = Xtr, ytr
        Xval, yval = val
        Xall, yall = torch.cat([Xtr, Xval]), torch.cat([ytr, yval])

    sweep = {}
    for wd in weight_decays:
        clf = fit_logistic_regression(Xfit, yfit, num_classes, wd, device=device)
        sweep[wd] = topk_accuracy(clf, Xval, yval)[1]
    best_wd = max(sweep, key=lambda w: (sweep[w], w))  # exact ties -> stronger regularization

    clf = fit_logistic_regression(Xall, yall, num_classes, best_wd, device=device)
    Xte, yte = test
    acc = topk_accuracy(clf, Xte, yte)
    return {
        "acc1": acc[1],
        "acc5": acc[5],
        "weight_decay": best_wd,
        "val_acc1_at_best": sweep[best_wd],
        "sweep": {str(w): a for w, a in sweep.items()},
        "n_train": int(Xall.size(0)),
        "n_val": int(Xval.size(0)),
        "n_test": int(Xte.size(0)),
    }


# --------------------------------------------------------------------------- entry point
def run_transfer_eval(
    backbone: nn.Module,
    source_dataset: str,
    device,
    targets: Optional[Sequence[str]] = None,
    roots: Optional[Dict[str, str]] = None,
    resolution: Optional[int] = None,
    weight_decays: Optional[Sequence[float]] = None,
    batch_size: int = 256,
    num_workers: int = 4,
    download: bool = False,
    strict: bool = True,
    log=print,
) -> Dict[str, Dict[str, float]]:
    """{target: evaluate_transfer_features(...) + resolution/num_classes} for every target.
    With strict=False a target whose dataset cannot be built (e.g. a missing root) is skipped
    with a warning instead of raising."""
    targets = list(targets) if targets else default_targets(source_dataset)
    roots = {**DEFAULT_ROOTS, **(roots or {})}
    resolution = resolution or default_resolution(source_dataset)
    weight_decays = list(weight_decays) if weight_decays else default_weight_decays()
    transform = build_eval_transform(source_dataset, resolution)

    was_training = backbone.training
    backbone.eval()
    results: Dict[str, Dict[str, float]] = {}
    for target in targets:
        num_classes, _ = TARGET_INFO[target]
        log(f"### transfer eval {source_dataset} -> {target} @ {resolution}px (root {roots[target]}) ###")
        try:
            ds_train, ds_val, ds_test = build_target_datasets(target, roots[target], transform, download=download)
        except (RuntimeError, FileNotFoundError, OSError) as e:
            if strict:
                raise
            log(f"### transfer eval: skipping {target} -- could not build the dataset from {roots[target]}: {e} ###")
            continue
        kw = dict(device=device, batch_size=batch_size, num_workers=num_workers)
        train = extract_features(backbone, ds_train, **kw)
        val = extract_features(backbone, ds_val, **kw) if ds_val is not None else None
        test = extract_features(backbone, ds_test, **kw)
        res = evaluate_transfer_features(train, val, test, num_classes, weight_decays, device=device)
        res["resolution"] = resolution
        res["num_classes"] = num_classes
        results[target] = res
        log(f"{target}: acc1={res['acc1']:.2f} acc5={res['acc5']:.2f} wd={res['weight_decay']:g} (val {res['val_acc1_at_best']:.2f})")
    if was_training:
        backbone.train()
    return results


def flatten_for_logging(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """{'transfer_<target>_acc1': ..., 'transfer_<target>_acc5': ...} for W&B summaries."""
    out = {}
    for target, r in results.items():
        out[f"transfer_{target}_acc1"] = r["acc1"]
        out[f"transfer_{target}_acc5"] = r["acc5"]
    return out
