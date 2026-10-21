"""Matplotlib figures shared by the experiments script and the Streamlit app."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .idle import idle_windows
from .program import TimedProgram

COL = {"cx": "#4c78a8", "sx": "#9ecae9", "x": "#9ecae9", "y": "#9ecae9", "dd": "#e45756",
       "idle-idle": "#fff2cc", "crosstalk-idle": "#fde0dd"}


def timeline_figure(prog: TimedProgram, edges, title: str = "", show_windows: bool = True, figsize=(10, 3.2)):
    n = prog.n
    fig, ax = plt.subplots(figsize=figsize)
    used = sorted({q for e in prog.events for q in e.qubits})
    row = {q: i for i, q in enumerate(used)}
    if show_windows:
        for w in idle_windows(prog, edges):
            if w.q in row:
                ax.add_patch(Rectangle((w.t0, row[w.q] - 0.38), w.L, 0.76, color=COL[w.kind], zorder=0))
    for e in prog.events:
        if e.name == "rz":
            continue
        key = "dd" if e.tag == "dd" else e.name
        if e.name == "cx":
            a, b = (row[q] for q in e.qubits)
            lo, hi = min(a, b), max(a, b)
            ax.add_patch(Rectangle((e.t, lo - 0.3), e.dur, hi - lo + 0.6, color=COL["cx"], alpha=0.85, zorder=2))
        else:
            r = row[e.qubits[0]]
            ax.add_patch(Rectangle((e.t, r - 0.3), max(e.dur, 14), 0.6, color=COL[key], zorder=3))
    ax.set_yticks(list(row.values()))
    ax.set_yticklabels([f"q{q}" for q in used])
    ax.set_xlim(0, prog.t_end)
    ax.set_ylim(len(used) - 0.4, -0.6)
    ax.set_xlabel("time [ns]")
    ax.set_title(title)
    handles = [Rectangle((0, 0), 1, 1, color=COL[k]) for k in ("cx", "sx", "dd", "idle-idle", "crosstalk-idle")]
    ax.legend(handles, ["CX", "1q gate", "DD pulse", "idle-idle window", "crosstalk-idle window"],
              ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8, frameon=False)
    fig.tight_layout()
    return fig
