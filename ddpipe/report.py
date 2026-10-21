"""Turn results/*.csv into PNG figures (for slides / README).   python -m ddpipe.report"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("results")
ORDER = ["Baseline", "Qiskit DD", "DD uniform", "DD staggered (ours)", "DD auto (ours)", "ZNE only",
         "Auto DD + ZNE (ours)", "Auto DD + ZNE + REM (ours)"]
COLORS = dict(zip(ORDER, ["#888888", "#aec7e8", "#6baed6", "#3182bd", "#08519c", "#fdae6b", "#e6550d", "#a63603"]))


def ablation():
    d = pd.read_csv(R / "ablation.csv")
    d["err"] = (1 - d["fidelity"]).abs()
    wls = list(dict.fromkeys(d.workload))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.8 / len(ORDER)
    for i, c in enumerate(ORDER):
        v = [d[(d.workload == k) & (d.config == c)].err.mean() for k in wls]
        ax.bar(np.arange(len(wls)) + i * w, v, w, label=c, color=COLORS[c])
    ax.set_xticks(np.arange(len(wls)) + 0.4 - w / 2)
    ax.set_xticklabels(wls)
    ax.set_ylabel("state-fidelity error  |1 - F|  (lower is better)")
    ax.set_title("Ablation: baseline vs DD vs ZNE vs combined (simulated noisy device)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(R / "fig_ablation.png", dpi=150); plt.close(fig)


def scaling():
    d = pd.read_csv(R / "scaling.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in ["Baseline", "Qiskit DD", "DD uniform", "DD auto (ours)"]:
        s = d[d.config == c].sort_values("n")
        ax.plot(s.n, s.fidelity, "o-", label=c, color=COLORS[c])
    ax.set_xlabel("GHZ size (qubits)"); ax.set_ylabel("state fidelity"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("GHZ fidelity vs size")
    fig.tight_layout(); fig.savefig(R / "fig_scaling.png", dpi=150); plt.close(fig)


def tau():
    d = pd.read_csv(R / "tau_sweep.csv")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, wl in zip(axs, list(dict.fromkeys(d.workload))):
        s = d[d.workload == wl]
        base = s[s.family == "-"].fidelity.iloc[0]
        for fam in ["CPMG", "XpXm", "XY4", "XY8"]:
            t = s[s.family == fam].sort_values("tau_ns")
            ax.plot(t.tau_ns, t.fidelity, "o-", label=fam)
        ax.axhline(base, ls="--", color="k", label="no DD")
        ax.set_xscale("log"); ax.set_xlabel("target pulse spacing tau [ns]"); ax.set_title(wl); ax.grid(alpha=.3)
    axs[0].set_ylabel("state fidelity"); axs[0].legend(fontsize=8)
    fig.suptitle("Pulse-spacing sweep: best spacing is neither the smallest nor the largest")
    fig.tight_layout(); fig.savefig(R / "fig_tau.png", dpi=150); plt.close(fig)


def noise():
    d = pd.read_csv(R / "noise_sweep.csv")
    axes = list(dict.fromkeys(d.axis))
    fig, axs = plt.subplots(1, len(axes), figsize=(10, 3.8))
    for ax, a in zip(np.atleast_1d(axs), axes):
        for c in ["Baseline", "Qiskit DD", "DD uniform", "DD staggered (ours)", "DD auto (ours)"]:
            s = d[(d.axis == a) & (d.config == c)].sort_values("value")
            ax.plot(s.value, s.fidelity, "o-", label=c, color=COLORS[c])
        ax.set_xlabel(a); ax.grid(alpha=.3)
    np.atleast_1d(axs)[0].set_ylabel("state fidelity"); np.atleast_1d(axs)[0].legend(fontsize=7)
    fig.suptitle("GHZ-wait-4: robustness to crosstalk and slow drift")
    fig.tight_layout(); fig.savefig(R / "fig_noise.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    for f in (ablation, scaling, tau, noise):
        f()
    print("figures written to", R.resolve())
