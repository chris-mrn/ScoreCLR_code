import torch
from torch.utils.data import TensorDataset

from scoreclr.data import build_loader


def test_only_training_loaders_keep_workers_alive():
    """evaluate() builds new loaders every time; persistent workers there piled up until the
    process ran out of file descriptors."""
    data = TensorDataset(torch.zeros(4, 1), torch.zeros(4))
    assert build_loader(data, 2, train=True, num_workers=2).persistent_workers
    assert not build_loader(data, 2, train=False, num_workers=2).persistent_workers
