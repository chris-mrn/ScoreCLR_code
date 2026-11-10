"""Every evaluation of the paper for one trained run, the same protocol for every method:

    python -m scoreclr.evaluation --run runs/cifar10_scoreclr --root stl10=./datasets --root dtd=./datasets

Reads <run>/last.pt (its config and weights) and writes <run>/evaluation.json:
  monitor   online classifier and kNN accuracy (scoreclr.evaluation.knn)
  linear    offline linear probe (scoreclr.evaluation.linear)
  stats     alignment / uniformity, RankMe of h and z, kurtosis, Epps-Pulley (scoreclr.evaluation.stats),
            from <run>/features.npz (scoreclr.evaluation.features)
  transfer  linear transfer to the paper's targets (scoreclr.evaluation.transfer)
Transfer roots are the directories torchvision's CIFAR10 / STL10 / DTD expect (containing
cifar-10-batches-py/, stl10_binary/, dtd/); --download fetches missing datasets there. Trailing
key=value arguments override the run's config, e.g. data.root=... when the data moved.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scoreclr.evaluation.features import extract
from scoreclr.evaluation.knn import monitor
from scoreclr.evaluation.linear import linear_probe
from scoreclr.evaluation.stats import representation_stats
from scoreclr.evaluation.transfer import run_transfer_eval
from scoreclr.methods import build_method
from scoreclr.utils import Config, load_checkpoint
from scoreclr.utils.config import apply_overrides

TARGETS = {"cifar10": ["stl10", "dtd"], "imagenet100": ["cifar10", "stl10", "dtd"],  # the paper's transfer table
           "imagenet1k": ["cifar10", "stl10", "dtd"]}


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--root", action="append", default=[], metavar="TARGET=PATH")
    p.add_argument("--targets", nargs="*", default=None, help="default: the paper's targets; empty to skip")
    p.add_argument("--download", action="store_true")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--probe_epochs", type=int, default=100)
    p.add_argument("overrides", nargs="*")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.run / "last.pt")
    assert ckpt is not None, f"no checkpoint in {args.run}"
    cfg = Config(apply_overrides(ckpt["config"], args.overrides))
    method = build_method(cfg)
    method.load_state_dict(ckpt["model"])
    method.to(device).eval()

    out = {"run": str(args.run), "epoch": ckpt["epoch"], "monitor": monitor(method, cfg, device)}
    out["linear"] = linear_probe(method.encoder.backbone, cfg.data, device, epochs=args.probe_epochs,
                                 num_workers=args.num_workers)
    feats = extract(method.encoder, cfg, device, num_workers=args.num_workers)
    np.savez_compressed(args.run / "features.npz", **feats)
    out["stats"] = representation_stats(feats)
    targets = TARGETS.get(cfg.data.dataset, []) if args.targets is None else args.targets
    if targets:
        roots = dict(kv.split("=", 1) for kv in args.root)
        out["transfer"] = run_transfer_eval(method.encoder.backbone, cfg.data.dataset, device, targets=targets,
                                            roots=roots, num_workers=args.num_workers, download=args.download)
    (args.run / "evaluation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "transfer"} |
                     {"transfer": {t: r["acc1"] for t, r in out.get("transfer", {}).items()}}, indent=1))
    return out


if __name__ == "__main__":
    main()
