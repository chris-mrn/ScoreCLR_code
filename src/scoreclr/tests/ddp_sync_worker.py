"""Run with torchrun by test_ddp.py: one epoch under DDP, then checks every parameter is
identical on all ranks while each rank saw a different data shard."""

import sys

import torch
import torch.distributed as torch_dist
from torch.utils.data.distributed import DistributedSampler

from scoreclr.data import TwoViews, build_dataset, build_loader, pretrain_transform
from scoreclr.train.train import setup, train_one_epoch
from scoreclr.utils import distributed as dist
from scoreclr.utils import load_config, seed_everything


def main():
    cfg = load_config(sys.argv[1], sys.argv[2:])
    device = dist.init_distributed()
    seed_everything(cfg.seed + dist.rank())
    train_set = build_dataset(cfg.data, True, TwoViews(pretrain_transform(cfg.data.dataset, cfg.data.crop_size, cfg.data.augment)))
    sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)
    loader = build_loader(train_set, cfg.optim.batch_size, True, 0, sampler=sampler)
    model, encoder, enc_opt, aux_opt = setup(cfg, device)
    train_one_epoch(model, encoder, enc_opt, aux_opt, loader, cfg, 0, device)

    params = torch.stack([p.detach().double().sum() for p in model.parameters()])
    gathered = [torch.empty_like(params) for _ in range(dist.world_size())]
    torch_dist.all_gather(gathered, params)
    indices = torch.tensor(list(sampler), dtype=torch.float64)
    shards = [torch.empty_like(indices) for _ in range(dist.world_size())]
    torch_dist.all_gather(shards, indices)
    if dist.is_main():
        assert all(torch.allclose(gathered[0], g, rtol=0, atol=1e-10) for g in gathered[1:]), "ranks diverged"
        assert not set(shards[0].tolist()) & set(shards[1].tolist()), "ranks saw the same data"
        print("DDP_SYNC_OK")
    dist.cleanup()


if __name__ == "__main__":
    main()
