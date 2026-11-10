from scoreclr.data import num_classes
from scoreclr.methods.barlow import BarlowTwins
from scoreclr.methods.base import SSLMethod
from scoreclr.methods.byol import BYOL
from scoreclr.methods.lejepa import LeJEPA
from scoreclr.methods.scoreclr import ScoreCLR
from scoreclr.methods.simclr import SimCLR

METHODS = {"scoreclr": ScoreCLR, "simclr": SimCLR, "lejepa": LeJEPA, "byol": BYOL, "barlow": BarlowTwins}


def build_method(cfg) -> SSLMethod:
    """The method named by cfg.method (default scoreclr), for cfg.data's dataset."""
    return METHODS[cfg.get("method", "scoreclr")](cfg, num_classes(cfg.data), small_images=cfg.data.crop_size <= 64)


__all__ = ["METHODS", "BYOL", "BarlowTwins", "LeJEPA", "SSLMethod", "ScoreCLR", "SimCLR", "build_method"]
