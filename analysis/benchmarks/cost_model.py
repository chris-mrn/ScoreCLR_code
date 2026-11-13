"""Analytical cost vs global batch size B (the cost model of Appendix C.7 of the paper, Table 10)

    python analysis/benchmarks/cost_model.py [out.pdf]

Analytical cost vs global batch size B (two views per image), unconditional recipe.

Cost model (Appendix C.7): N = 2B codes of dimension d = 256.
  SimCLR objective : 6 N^2 d FLOPs, 16 N^2 bytes
  ScoreCLR objective: 2P(6S+2K) = 20.18 MFLOPs per image (score net P = 280,320, S = 5, K = 3)
                      and 39,424 bytes per image saved for backward in fp32:
                      30,208 (one DSM step of the score net; the S steps are sequential and the
                      repulsion queries run without gradient) + 9,216 (loss), counted with
                      torch.autograd.graph.saved_tensors_hooks on the implemented modules.
Full-iteration per-device memory ratio (W devices):
  (a B/W + c B/W) / (a B/W + 16 (2B/W)(2B))  =  (a + c) / (a + 64 B)
with c = 39,424 bytes and a = saved-activation bytes per image of the encoder (mixed precision),
fitted at B = 32768 with an earlier estimate of c (30.8 KB per image).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED, BLUE = "#cc0000", "#1a6fd4"
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

d = 256
B = np.logspace(np.log10(1024), np.log10(65536), 200)
N = 2 * B
simclr_tflops = 6 * N**2 * d / 1e12
simclr_gb = 16 * N**2 / 1e9
C_NEW = 39424.0
score_tflops = 20.18304e6 * B / 1e12
score_gb = C_NEW * B / 1e9

# Table 10 points
Bt = np.array([4096, 8192, 16384, 32768, 65536])
tab = {
    "simclr_tflops": [0.10, 0.41, 1.65, 6.60, 26.4],
    "score_tflops": [0.08, 0.17, 0.33, 0.66, 1.32],
    "simclr_gb": [1.1, 4.3, 17.2, 68.7, 275],
    "score_gb": [0.16, 0.32, 0.65, 1.29, 2.58],
    "r18_32": [0.96, 0.92, 0.85, 0.73, 0.57],
    "r18_224": [0.99, 0.98, 0.96, 0.92, 0.85],
}

# encoder activation bytes per image, fitted at B = 32768 with an earlier estimate of c (30.8 KB per image)
def fit_a(r):
    return (r * 64 * 32768 - 30.8e3) / (1 - r)

ratio = lambda a, B: (a + C_NEW) / (a + 64 * B)
configs = [
    ("ResNet-18, 32 px", fit_a(0.73), "-"),
    ("ResNet-18, 224 px", fit_a(0.92), "--"),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
mk = dict(marker="o", ms=5, mec="black", mew=0.8, linestyle="none", zorder=5)

ax = axes[0]
ax.plot(B, simclr_tflops, color=BLUE, lw=2.2, label="SimCLR (InfoNCE)")
ax.plot(B, score_tflops, color=RED, lw=2.2, label="ScoreCLR")
ax.plot(Bt, tab["simclr_tflops"], color=BLUE, **mk)
ax.plot(Bt, tab["score_tflops"], color=RED, **mk)
ax.set_ylabel("Objective FLOPs per iteration (TFLOPs)")
ax.legend(frameon=False, loc="upper left")

ax = axes[1]
ax.plot(B, simclr_gb, color=BLUE, lw=2.2, label="SimCLR (InfoNCE)")
ax.plot(B, score_gb, color=RED, lw=2.2, label="ScoreCLR")
ax.plot(Bt, tab["simclr_gb"], color=BLUE, **mk)
ax.plot(Bt, tab["score_gb"], color=RED, **mk)
ax.axhline(80, color="black", lw=1.0, ls="--", alpha=0.6)
ax.text(1150, 100, "80 GB GPU", fontsize=9, color="black", alpha=0.8, va="bottom")
ax.set_ylabel("Objective memory (GB)")
ax.legend(frameon=False, loc="lower right")

for ax in axes[:2]:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Global batch size $B$")
    ax.grid(True, which="major", color="0.85", lw=0.6)

ax = axes[2]
for (lab, a, ls), key in zip(configs, ["r18_32", "r18_224"]):
    ax.plot(B, ratio(a, B), color=RED, lw=2.0, ls=ls, label=lab)
    ax.plot(Bt, tab[key], color=RED, **mk)
ax.axhline(1.0, color=BLUE, lw=1.6, alpha=0.9)
ax.text(1100, 1.012, "SimCLR", fontsize=10, color=BLUE)
ax.set_xscale("log", base=2)
ax.set_xlabel("Global batch size $B$")
ax.set_ylabel("ScoreCLR / SimCLR per-device memory")
ax.set_ylim(0.5, 1.05)
ax.grid(True, which="major", color="0.85", lw=0.6)
ax.legend(frameon=False, loc="lower left")

for ax in axes:
    ax.set_xlim(1024, 65536)
    ax.set_xticks([1024, 4096, 16384, 65536])
    ax.set_xticklabels(["1k", "4k", "16k", "64k"])
    for s in ax.spines.values():
        s.set_color("black")

fig.tight_layout(w_pad=2.0)
out = sys.argv[1] if len(sys.argv) > 1 else "cost_vs_batch.pdf"
fig.savefig(out, dpi=180, bbox_inches="tight")
print("wrote", out)
for (lab, a, _), key in zip(configs, ["r18_32", "r18_224"]):
    print(f"{lab}: a = {a/1e6:.1f} MB/image; model vs table:",
          " ".join(f"{ratio(a, b):.2f}/{t:.2f}" for b, t in zip(Bt, tab[key])))
