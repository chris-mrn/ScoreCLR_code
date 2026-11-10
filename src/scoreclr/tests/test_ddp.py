import subprocess
import sys
from pathlib import Path

import torch

from scoreclr.tests.conftest import CONFIG

TINY = [
    "data.dataset=fake", "data.fake_size=32", "data.crop_size=16", "data.num_workers=0",
    "model.proj_hidden_dim=32", "model.proj_dim=16", "model.score_hidden_dim=32", "model.score_time_emb_dim=8",
    "optim.batch_size=4", "eval.every=1",
]


def _torchrun(out: Path, epochs: int) -> None:
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "--master_port=29531",
           "-m", "scoreclr.train", "--config", str(CONFIG), *TINY, f"optim.epochs={epochs}", f"output_dir={out}"]
    subprocess.run(cmd, check=True, cwd=CONFIG.parents[1], timeout=600)


def test_ddp_train_and_resume(tmp_path):
    """2 processes (gloo on CPU): trains, checkpoints, resumes, and rank 0 monitors."""
    out = tmp_path / "ddp"
    _torchrun(out, epochs=1)
    ckpt = torch.load(out / "last.pt", weights_only=False)
    # 32 images / (2 processes x batch 4) = 4 steps per epoch and process
    assert ckpt["epoch"] == 0

    _torchrun(out, epochs=2)
    assert torch.load(out / "last.pt", weights_only=False)["epoch"] == 1
    assert len((out / "log.jsonl").read_text().splitlines()) == 2  # one line per epoch


def test_ddp_ranks_stay_in_sync():
    """After an epoch on different shards, every parameter is identical on both ranks."""
    worker = Path(__file__).with_name("ddp_sync_worker.py")
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "--master_port=29532",
           str(worker), str(CONFIG), *TINY, "optim.epochs=1"]
    res = subprocess.run(cmd, check=True, cwd=CONFIG.parents[1], timeout=600, capture_output=True, text=True)
    assert "DDP_SYNC_OK" in res.stdout
