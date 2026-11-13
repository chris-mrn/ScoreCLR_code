"""Measured cost vs batch, from measured_objective_cost.py's results (JSON lines).

    python analysis/benchmarks/plot_measured_cost.py results.jsonl cost_vs_batch_measured.pdf

(a) objective alone, ms per step; (b) objective alone, extra peak memory; (c) full iteration,
time ratio ScoreCLR / SimCLR; (d) full iteration, peak-memory ratio ScoreCLR / SimCLR.
x: global batch B (images), log2. OOM: a cross at the first batch that did not fit.
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rows = [json.loads(line) for line in open(sys.argv[1])]
out = sys.argv[2] if len(sys.argv) > 2 else "cost_vs_batch_measured.pdf"

# categorical slots 1-5 of the reference palette, fixed order; markers/dashes as secondary encoding
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METHODS = [("simclr", "SimCLR (InfoNCE)", C[0], "o", "-"),
           ("scoreclr_S1_K1", "ScoreCLR S=1, K=1", C[1], "s", "-"),
           ("scoreclr_S5_K1", "ScoreCLR S=5, K=1", C[2], "^", "-"),
           ("scoreclr_S5_K3", "ScoreCLR S=5, K=3", C[3], "D", "-"),
           ("scoreclr_S10_K1", "ScoreCLR S=10, K=1", C[4], "v", "-")]
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def select(**kw):
    return sorted([r for r in rows if all(r.get(k) == v for k, v in kw.items())], key=lambda r: r["B"])


def line(ax, pts, key, color, marker, dash, label):
    ok = [r for r in pts if not r["oom"]]
    ax.plot([r["B"] for r in ok], [key(r) for r in ok], color=color, marker=marker, ls=dash, lw=1.5, ms=4,
            label=label)
    oom = [r for r in pts if r["oom"]]
    if ok and oom:  # cross at the first batch that did not fit, at the last measured height
        ax.plot(oom[0]["B"], key(ok[-1]), marker="x", color=color, ms=7, mew=1.8, ls="none")


def style(ax, ylabel, title, logy=True):
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_xlabel("global batch $B$ (images)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")


fig, axes = plt.subplots(1, 4, figsize=(13, 3.1), constrained_layout=True)
for m, label, c, mk, ls in METHODS:
    pts = select(part="objective", method=m)
    line(axes[0], pts, lambda r: r["ms"], c, mk, ls, label)
    line(axes[1], pts, lambda r: r["mem_extra"] / 2**20, c, mk, ls, label)
style(axes[0], "ms per step", "(a) objective alone: time")
style(axes[1], "extra peak memory (MiB)", "(b) objective alone: memory")
axes[0].legend(frameon=False, fontsize=7)

# full iteration ratios: 32 px with S=1, K=1 (the CIFAR-10 recipe) and 224 px with S=5, K=3 (the ImageNet-100 recipe)
SETTINGS = [("full", 32, "fp32", 1, "scoreclr_S1_K1", "32 px fp32, S=1 K=1", C[0], "o", "-"),
            ("full", 32, "fp16", 1, "scoreclr_S1_K1", "32 px fp16, S=1 K=1", C[1], "s", "-"),
            ("full", 224, "fp32", 1, "scoreclr_S5_K3", "224 px fp32, S=5 K=3", C[2], "^", "-"),
            ("dp_emulated", 32, "fp32", 8, "scoreclr_S1_K1", "32 px, W=8 (emulated)", C[3], "D", "--"),
            ("dp_emulated", 224, "fp32", 8, "scoreclr_S5_K3", "224 px, W=8 (emulated)", C[4], "v", "--")]
for part, res, prec, W, m, label, c, mk, ls in SETTINGS:
    sc = {r["B"]: r for r in select(part=part, res=res, precision=prec, W=W, method=m)}
    si = {r["B"]: r for r in select(part=part, res=res, precision=prec, W=W, method="simclr")}
    pts = []
    for B in sorted(set(sc) | set(si)):
        a, b = sc.get(B), si.get(B)
        if a is None or b is None:
            continue
        both = not a["oom"] and not b["oom"]
        pts.append(dict(B=B, oom=not both, t=a["ms"] / b["ms"] if both else None,
                        m=a["mem_peak"] / b["mem_peak"] if both else None))
    line(axes[2], pts, lambda r: r["t"], c, mk, ls, label)
    line(axes[3], pts, lambda r: r["m"], c, mk, ls, label)
for ax, yl, t in [(axes[2], "time ScoreCLR / SimCLR", "(c) full iteration: time ratio"),
                  (axes[3], "peak memory ScoreCLR / SimCLR", "(d) full iteration: memory ratio")]:
    style(ax, yl, t, logy=False)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
axes[2].legend(frameon=False, fontsize=7)
gpu = rows[0]["gpu"] if rows else ""
fig.suptitle(f"Measured on one {gpu}, commit {rows[0]['commit'] if rows else ''}; no data loader",
             fontsize=8, color=MUTED, x=0.01, ha="left")
fig.savefig(out)
fig.savefig(out.rsplit(".", 1)[0] + ".png", dpi=200)
print("saved", out)
