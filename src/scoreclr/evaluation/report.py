"""The paper's tables from reproduced runs, next to the numbers reported in the paper:

    python -m scoreclr.evaluation.report [runs]

Reads <runs>/<row>/evaluation.json (python -m scoreclr.evaluation) for every row (config/<row>.yaml)
and prints each table as "reproduced (paper)"; missing runs print "--".
The paper reports the mean over seeds 1, 2 and 3 for the ScoreCLR, SimCLR (AdamW) and LeJEPA rows, and a
single run for BYOL and Barlow Twins (seed 1) and for SimCLR with LARS (seed 5). Training is not
deterministic (cuDNN), so a reproduced run differs from these numbers by seed-to-seed and run-to-run
noise. Rows marked * in PROTOCOL were evaluated in the paper with another protocol than the one shared
here (README.md).
"""

import json
import sys
from pathlib import Path

# Values reported in the paper: linear probe and kNN (top-1, top-5), alignment and uniformity, RankMe of the
# backbone features and of the projector output, transfer top-1. Rows are config/<name>.yaml. The paper has no
# normality table: `normality` only carries the RankMe of z, and the kurtosis and Epps-Pulley columns print the
# reproduced value alone.
PAPER = {
    "cifar10_simclr_adamw": dict(lin=(89.26, 99.70), knn=(88.73, 98.90), align=0.367, unif=-3.804,
                                 rank=(336, 96), transfer={"stl10": 76.12, "dtd": 30.46}),
    "cifar10_lejepa": dict(lin=(89.67, 99.65), knn=(88.40, 98.65), align=0.316, unif=-3.540, rank=(253, 58),
                           normality=(None, None, 58), transfer={"stl10": 76.01, "dtd": 27.02}),
    "cifar10_byol": dict(lin=(88.90, 99.68), knn=(88.37, 98.98)),
    "cifar10_barlow": dict(lin=(87.89, 99.62), knn=(87.26, 98.61)),
    "cifar10_scoreclr": dict(lin=(88.87, 99.66), knn=(88.27, 98.85), align=0.514, unif=-3.802, rank=(332, 229),
                             normality=(None, None, 229), transfer={"stl10": 75.21, "dtd": 28.01}),
    "cifar10_simclr_lars": dict(lin=(90.79, 99.74), knn=(89.99, 99.02), align=0.349, unif=-3.818, rank=(372, 88),
                                normality=(None, None, 88)),
    "imagenet100_simclr_adamw": dict(lin=(72.39, 92.18), knn=(66.41, 87.32), align=0.235, unif=-3.855,
                                     rank=(329, 143), normality=(None, None, 143),
                                     transfer={"cifar10": 75.64, "stl10": 84.35, "dtd": 59.38}),
    "imagenet100_scoreclr": dict(lin=(71.23, 91.03), knn=(66.05, 86.09), align=0.378, unif=-3.931, rank=(346, 244),
                                 normality=(None, None, 244),
                                 transfer={"cifar10": 71.23, "stl10": 82.87, "dtd": 56.28}),
    "imagenet100_simclr_lars": dict(lin=(75.40, 93.30), knn=(69.98, 88.64), align=0.240, unif=-3.856, rank=(368, 118)),
    "imagenet100_simclr_lars_proj4096": dict(lin=(77.02, 93.84), knn=(71.60, 89.10)),
}


# What the paper's value of a row used instead of the shared protocol: knn = memory bank under
# crop + flip; dali = pretrained with NVIDIA DALI loading.
PROTOCOL = {"cifar10_simclr_lars": "knn", "imagenet100_simclr_lars": "knn, dali",
            "imagenet100_simclr_lars_proj4096": "knn"}


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def _cell(value, paper, fmt):
    v = "--" if value is None else fmt.format(value)
    return f"{v} ({fmt.format(paper)})" if paper is not None else v


def main(argv=None) -> None:
    runs = Path((argv or sys.argv[1:] or ["runs"])[0])
    print(f"reproduced (paper), from {runs}/<row>/evaluation.json; * = evaluated in the paper with another protocol\n")
    tables = {"Linear probe / kNN (top-1, top-5)": [], "Alignment / uniformity": [], "RankMe h / z": [],
              "Normality: kurtosis, EP whitened, RankMe z": [], "Transfer top-1": []}
    for row, p in PAPER.items():
        ev = _load(runs / row / "evaluation.json") or {}
        res = ev.get("linear", {}) | ev.get("monitor", {})
        st, tr = ev.get("stats", {}), ev.get("transfer", {})
        name = row + (f" *{PROTOCOL[row]}" if row in PROTOCOL else "")
        tables["Linear probe / kNN (top-1, top-5)"].append(
            f"{name:44s} " + "  ".join(_cell(res.get(k), v, "{:.2f}") for k, v in zip(
                ["linear_acc1", "linear_acc5", "eval/knn_acc1", "eval/knn_acc5"], [*p["lin"], *p["knn"]])))
        if "align" in p:
            tables["Alignment / uniformity"].append(
                f"{name:44s} {_cell(st.get('align'), p['align'], '{:.3f}')}  {_cell(st.get('unif'), p['unif'], '{:.3f}')}")
        if "rank" in p:
            tables["RankMe h / z"].append(
                f"{name:44s} {_cell(st.get('rankme_h'), p['rank'][0], '{:.0f}')}  {_cell(st.get('rankme_z'), p['rank'][1], '{:.0f}')}")
        if "normality" in p:
            k, e, r = p["normality"]
            tables["Normality: kurtosis, EP whitened, RankMe z"].append(
                f"{name:44s} {_cell(st.get('kurtosis_mean'), k, '{:.2f}')}  {_cell(st.get('ep_whitened'), e, '{:.1f}')}"
                f"  {_cell(st.get('rankme_z'), r, '{:.0f}')}")
        if "transfer" in p:
            tables["Transfer top-1"].append(f"{name:44s} " + "  ".join(
                f"{t}: {_cell(tr.get(t, {}).get('acc1'), v, '{:.2f}')}" for t, v in p["transfer"].items()))
    for title, lines in tables.items():
        print(f"== {title}")
        print("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
