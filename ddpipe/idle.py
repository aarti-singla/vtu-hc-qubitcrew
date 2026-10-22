"""Idle-window analyzer.

For every qubit we find the gaps between consecutive operations (inside its
active lifetime, i.e. after its first and before the final time) and classify
each gap following the idle-idle / crosstalk-idle distinction used in the DD
literature:

  * ``idle-idle``       - no neighbouring qubit is running a two-qubit gate
  * ``crosstalk-idle``  - at least one coupled neighbour is driven by a CX
                          while this qubit waits (strongest crosstalk)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .program import TimedProgram

EPS = 1e-6


@dataclass
class Window:
    q: int
    t0: float
    t1: float
    kind: str = "idle-idle"
    busy_frac: float = 0.0      # fraction of the window a neighbour is in a CX

    @property
    def L(self) -> float:
        return self.t1 - self.t0


def _neighbors(edges):
    nb: dict[int, set] = {}
    for a, b in edges:
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)
    return nb


def idle_windows(prog: TimedProgram, edges, min_len: float = 0.0) -> list[Window]:
    per_q: dict[int, list] = {}
    for e in prog.sorted_events():
        for q in e.qubits:
            per_q.setdefault(q, []).append(e)

    cx_by_q: dict[int, list] = {}
    for e in prog.events:
        if len(e.qubits) == 2:
            for q in e.qubits:
                cx_by_q.setdefault(q, []).append((e.t, e.end))
    nb = _neighbors(edges)

    wins: list[Window] = []
    for q, evs in per_q.items():
        cursor = evs[0].end
        spans = [(e.t, e.end) for e in evs[1:]] + [(prog.t_end, prog.t_end)]
        for s, e_end in spans:
            if s - cursor > max(EPS, min_len):
                wins.append(Window(q, cursor, s))
            cursor = max(cursor, e_end)
    # classify: busy time = length of the union of neighbours' two-qubit-gate intervals inside the window
    for w in wins:
        spans = []
        for nq in nb.get(w.q, ()):
            for (a, b) in cx_by_q.get(nq, ()):
                lo, hi = max(w.t0, a), min(w.t1, b)
                if hi > lo:
                    spans.append((lo, hi))
        busy, cur_lo, cur_hi = 0.0, None, None
        for lo, hi in sorted(spans):
            if cur_hi is None or lo > cur_hi:
                if cur_hi is not None:
                    busy += cur_hi - cur_lo
                cur_lo, cur_hi = lo, hi
            else:
                cur_hi = max(cur_hi, hi)
        if cur_hi is not None:
            busy += cur_hi - cur_lo
        w.busy_frac = busy / w.L if w.L > 0 else 0.0
        w.kind = "crosstalk-idle" if w.busy_frac > 0.05 else "idle-idle"
    return wins


def idle_report(prog: TimedProgram, edges) -> dict:
    wins = idle_windows(prog, edges)
    active = {}
    for e in prog.events:
        for q in e.qubits:
            active.setdefault(q, [e.t, e.end])
            active[q][0] = min(active[q][0], e.t)
            active[q][1] = max(active[q][1], e.end)
    rows = []
    for q in sorted(active):
        qw = [w for w in wins if w.q == q]
        life = prog.t_end - active[q][0]
        idle = sum(w.L for w in qw)
        rows.append({
            "qubit": q,
            "lifetime_ns": life,
            "idle_ns": idle,
            "idle_pct": 100 * idle / life if life > 0 else 0.0,
            "n_windows": len(qw),
            "longest_ns": max((w.L for w in qw), default=0.0),
        })
    tot_idle = sum(r["idle_ns"] for r in rows)
    tot_life = sum(r["lifetime_ns"] for r in rows)
    return {
        "per_qubit": rows,
        "total_idle_ns": tot_idle,
        "idle_pct": 100 * tot_idle / tot_life if tot_life else 0.0,
        "n_windows": len(wins),
        "n_crosstalk_idle": sum(1 for w in wins if w.kind == "crosstalk-idle"),
        "t_end_ns": prog.t_end,
        "windows": wins,
    }
