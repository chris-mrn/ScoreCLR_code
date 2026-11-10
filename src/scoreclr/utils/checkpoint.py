import os
from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(path: Path, state: dict) -> None:
    """Atomic save, so a preemption never leaves a corrupt checkpoint."""
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device="cpu") -> Optional[dict]:
    path = Path(path)
    return torch.load(path, map_location=device, weights_only=False) if path.exists() else None
