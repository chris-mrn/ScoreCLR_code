from pathlib import Path

import pytest

from scoreclr.utils import load_config

CONFIG = Path(__file__).resolve().parents[3] / "config" / "cifar10_scoreclr.yaml"


@pytest.fixture
def tiny_cfg(tmp_path):
    """config/cifar10_scoreclr.yaml shrunk to a CPU-sized run on random images."""
    return load_config(
        CONFIG,
        [
            "data.dataset=fake", "data.fake_size=32", "data.crop_size=16", "data.num_workers=0",
            "model.proj_hidden_dim=32", "model.proj_dim=16", "model.score_hidden_dim=32", "model.score_time_emb_dim=8",
            "optim.epochs=2", "optim.batch_size=8", "optim.cosine_decay_epochs=1",
            "eval.every=1",
            f"output_dir={tmp_path / 'run'}",
        ],
    )
