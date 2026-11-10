"""Exports the weights of trained runs for a release: one file per run, without optimizer state, and a
SHA256SUMS file next to them.

    python scripts/export_weights.py runs/cifar10_scoreclr runs/imagenet100_scoreclr ... --out weights

Each <out>/<run name>.pt holds {model, config, epoch}; evaluate it with
    python -m scoreclr.evaluation --run runs/<name> --checkpoint weights/<name>.pt
"""

import argparse
import hashlib
from pathlib import Path

from scoreclr.utils import export_weights


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="run directories containing last.pt")
    p.add_argument("--out", type=Path, default=Path("weights"))
    args = p.parse_args()
    sums = []
    for run in args.runs:
        dst = export_weights(run / "last.pt", args.out / f"{run.name}.pt")
        sums.append(f"{hashlib.sha256(dst.read_bytes()).hexdigest()}  {dst.name}")
        print(f"{dst}  {dst.stat().st_size / 2**20:.1f} MiB")
    (args.out / "SHA256SUMS").write_text("\n".join(sums) + "\n")


if __name__ == "__main__":
    main()
