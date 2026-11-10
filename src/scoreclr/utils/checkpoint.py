import os
from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(path: Path, state: dict) -> None:
    """Atomic save, so a preemption never leaves a corrupt checkpoint."""
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def export_weights(src, dst) -> Path:
    """Release file from a training checkpoint: the model weights, the config and the epoch only (no
    optimizer or scaler state), which is all `python -m scoreclr.evaluation --checkpoint` needs."""
    ckpt = load_checkpoint(src)
    assert ckpt is not None, f"no checkpoint at {src}"
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(dst, {k: ckpt[k] for k in ("model", "config", "epoch")})
    return dst


def load_checkpoint(path: Path, device="cpu") -> Optional[dict]:
    path = Path(path)
    return torch.load(path, map_location=device, weights_only=False) if path.exists() else None
