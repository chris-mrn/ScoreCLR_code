"""The paper's tables from reproduced runs, next to the published numbers:

    python -m scoreclr.evaluation.report [runs]

Reads <runs>/<row>/evaluation.json (python -m scoreclr.evaluation) for every row (config/<row>.yaml)
and prints each table as "reproduced (paper)"; missing runs print "--".
Every result in the paper is a single run (seed 5) and training is not deterministic (cuDNN), so
reproduced numbers differ from the published ones by run-to-run noise. Rows marked * in PROTOCOL were
evaluated in the published version with another protocol than the one shared here (README.md).
"""

import json
import sys
from pathlib import Path

# Published values, arXiv paper (arkiv/ScoreCLR_arkiv.tex). Rows are config/<name>.yaml.
PAPER = {
    "cifar10_simclr_adamw": dict(lin=(89.45, 99.70), knn=(88.56, 98.84), align=0.355, unif=-3.803,
                                 rank=(356, 96), transfer={"stl10": 76.05, "dtd": 30.96}),
    "cifar10_lejepa": dict(lin=(90.07, 99.62), knn=(88.35, 98.54), align=0.313, unif=-3.539, rank=(266, 59),
                           normality=(0.36, 8.7, 59), transfer={"stl10": 75.66, "dtd": 26.38}),
    "cifar10_scoreclr": dict(lin=(89.33, 99.59), knn=(88.53, 98.98), align=0.497, unif=-3.802, rank=(339, 229),
                             normality=(0.33, 7.0, 229), transfer={"stl10": 75.20, "dtd": 27.34}),
    "cifar10_simclr_lars": dict(lin=(90.79, 99.74), knn=(89.99, 99.02), align=0.349, unif=-3.818, rank=(372, 88),
                                normality=(2.72, 921, 88)),
    "imagenet100_simclr_adamw": dict(lin=(72.60, 92.58), knn=(65.06, 86.98), align=0.236, unif=-3.853,
                                     rank=(358, 143), normality=(3.66, 80.9, 143),
                                     transfer={"cifar10": 76.46, "stl10": 84.21, "dtd": 60.16}),
    "imagenet100_scoreclr": dict(lin=(71.28, 90.76), knn=(66.02, 85.96), align=0.382, unif=-3.936, rank=(348, 245),
                                 normality=(0.25, 2.1, 245),
                                 transfer={"cifar10": 71.39, "stl10": 81.36, "dtd": 56.86}),
    "imagenet100_simclr_lars": dict(lin=(75.40, 93.30), knn=(69.98, 88.64), align=0.240, unif=-3.856, rank=(368, 118)),
    "imagenet100_simclr_lars_proj4096": dict(lin=(77.02, 93.84), knn=(71.60, 89.10)),
}


# What the published value of a row used instead of the shared protocol: knn = memory bank under
# crop + flip; probe = ImageNet-100 probe without warm-up; dali = pretrained with NVIDIA DALI loading.
PROTOCOL = {"cifar10_simclr_adamw": "knn", "cifar10_lejepa": "knn", "cifar10_simclr_lars": "knn",
            "imagenet100_simclr_adamw": "knn", "imagenet100_simclr_lars": "knn, dali",
            "imagenet100_simclr_lars_proj4096": "knn", "imagenet100_scoreclr": "probe"}


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def _cell(value, paper, fmt):
    v = "--" if value is None else fmt.format(value)
    return f"{v} ({fmt.format(paper)})" if paper is not None else v


def main(argv=None) -> None:
    runs = Path((argv or sys.argv[1:] or ["runs"])[0])
    print(f"reproduced (paper), from {runs}/<row>/evaluation.json; * = published with another protocol\n")
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
