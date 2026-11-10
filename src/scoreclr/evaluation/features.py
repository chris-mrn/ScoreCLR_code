"""Features of a trained encoder, in the format the paper's tables were computed from:

    h_val, z_val, y_val  backbone features and projector codes of the whole validation set
                         (probe test transform, no augmentation)
    z_p1, z_p2           codes of two pretraining-augmented views of 2000 train images
                         (chosen with numpy default_rng(1)), for the alignment
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from scoreclr.data import TwoViews, build_dataset, eval_transforms, pretrain_transform


@torch.no_grad()
def extract(encoder, cfg, device, n_pairs: int = 2000, num_workers: int = 8) -> dict:
    encoder.eval()

    def codes(x):
        h = encoder.backbone(x.to(device, non_blocking=True))
        return h.float().cpu(), encoder.projector(h).float().cpu()

    _, test_tf = eval_transforms(cfg.data.dataset, cfg.data.crop_size)
    hs, zs, ys = [], [], []
    for x, y in DataLoader(build_dataset(cfg.data, train=False, transform=test_tf), batch_size=256,
                           num_workers=num_workers):
        h, z = codes(x)
        hs.append(h)
        zs.append(z)
        ys.append(y)

    view = TwoViews(pretrain_transform(cfg.data.dataset, cfg.data.crop_size, cfg.data.augment))
    pairs = build_dataset(cfg.data, train=True, transform=view)
    sel = np.random.default_rng(1).choice(len(pairs), size=min(n_pairs, len(pairs)), replace=False).tolist()
    torch.manual_seed(0)
    z1s, z2s = [], []
    for (x1, x2), _ in DataLoader(Subset(pairs, sel), batch_size=256, num_workers=num_workers):
        z1s.append(codes(x1)[1])
        z2s.append(codes(x2)[1])
    return {"h_val": torch.cat(hs).numpy(), "z_val": torch.cat(zs).numpy(), "y_val": torch.cat(ys).numpy(),
            "z_p1": torch.cat(z1s).numpy(), "z_p2": torch.cat(z2s).numpy()}
