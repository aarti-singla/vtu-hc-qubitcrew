"""Context-aware dynamical-decoupling injector.

Strategies
----------
``uniform``    every idle window independently gets a symmetric CPMG-style train
               whose pulse count follows ``tau_ns``.  This is what a context-blind
               DD pass does (it is also close to Qiskit's PadDynamicalDecoupling).
``staggered``  (our contribution) crosstalk-aware DD.  A pair of coupled idle
               qubits with *identical* DD trains keeps its ZZ crosstalk, because
               X/Y pulses commute with ZZ:  the ZZ phase between qubits a and b is
               J * integral( s_a(t) s_b(t) dt ), where s(t) = +-1 is each qubit's
               toggling frame.  Windows are processed longest-first; for every
               window we search a small candidate set (CPMG / Uhrig placement,
               several even pulse counts) and keep the one minimising

                   cost = pulse_cost * n_pulses
                        + sum_over_overlapping_neighbours ( J_ab * overlap_integral )^2 / 4

               so ZZ is refocused wherever the overlap is long enough to matter,
               while short windows stay cheap (imperfect pulses cost fidelity).
``none``       leave the schedule untouched.

Every candidate is a symmetric even-count train, so single-qubit quasi-static
dephasing is cancelled exactly regardless of the choice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .idle import Window, _neighbors, idle_windows
from .program import Event, TimedProgram

FAMILIES = ("CPMG", "XpXm", "XY4", "XY8")
DEFAULT_ZZ_RAD_NS = 2 * np.pi * 70e3 * 1e-9      # used if no calibration is supplied


def pulse_train(family: str, n: int) -> list[str]:
    """Pulse names for ``n`` (even) pulses; ideal product is the identity."""
    if n % 2:
        raise ValueError("pulse count must be even")
    if family == "CPMG":
        return ["x"] * n
    if family == "XpXm" or n == 2:
        return ["x", "xm"] * (n // 2)
    if family == "XY8" and n % 8 == 0:
        return ["x", "y", "x", "y", "y", "x", "y", "x"] * (n // 8)
    if n % 4 == 0:                                   # XY4 (also XY8 fallback)
        return ["x", "y", "x", "y"] * (n // 4)
    return ["x", "xm"] * (n // 2)                    # n = 2 mod 4


@dataclass
class DDInfo:
    strategy: str
    family: str
    n_windows: int = 0
    n_protected: int = 0
    n_pulses: int = 0
    n_colors: int = 0          # kept for API compatibility: number of distinct pulse counts used


def _fractions(kind: str, n: int) -> list[float]:
    if kind == "cpmg":
        return [(j + 0.5) / n for j in range(n)]
    return [math.sin(math.pi * (j + 1) / (2 * n + 2)) ** 2 for j in range(n)]   # Uhrig


def _frame_cost(tg_a, wa, nb_wins, splits) -> float:
    """Sum of squared ZZ-frame integrals over window ``wa`` against one neighbour.

    s_a toggles at ``tg_a``.  s_b is the neighbour's toggling frame: inside each
    already-placed neighbour window ``(t0, t1, toggles)`` it flips at that
    window's pulses; elsewhere (neighbour busy, or idle but not yet protected)
    it is +1.  The neighbour's own gates (sx / cx) do not commute with Z, so the
    ZZ phase accumulated before and after such a gate cannot cancel against each
    other: the integral must vanish *separately* in every sub-interval between
    the neighbour's gates (``splits``).
    """
    lo, hi = wa.t0, wa.t1
    cuts = [t for t in splits if lo < t < hi]
    pts = {lo, hi, *cuts, *[t for t in tg_a if lo < t < hi]}
    for (b0, b1, tb) in nb_wins:
        pts.update(t for t in (b0, b1, *tb) if lo < t < hi)
    pts = sorted(pts)
    seg = {}
    for a, b in zip(pts[:-1], pts[1:]):
        mid = 0.5 * (a + b)
        sa = -1.0 if sum(t < mid for t in tg_a) % 2 else 1.0
        sb = 1.0
        for (b0, b1, tb) in nb_wins:
            if b0 < mid < b1:
                sb = -1.0 if sum(t < mid for t in tb) % 2 else 1.0
        k = sum(t < mid for t in cuts)
        seg[k] = seg.get(k, 0.0) + sa * sb * (b - a)
    return sum(v * v for v in seg.values())


def _feasible(fr, L, tp):
    cs = [f * L for f in fr]
    if cs[0] < tp * 0.55 or L - cs[-1] < tp * 0.55:
        return False
    return all(b - a >= tp * 1.05 for a, b in zip(cs[:-1], cs[1:]))


def inject_dd(prog: TimedProgram, edges, tp_ns: float, strategy: str = "staggered",
              family: str = "XY4", tau_ns: float = 600.0, min_window_ns: float = 250.0,
              zz_rates: dict | None = None, pulse_cost: float = 3e-4,
              max_pulses: int = 16) -> tuple[TimedProgram, DDInfo]:
    """Return a copy of ``prog`` with DD pulses inserted into idle windows.

    tau_ns     : target spacing between pulses.  Surveys of DD on IBM hardware find
                 the best spacing to be well above the hardware minimum, so this is
                 a tunable that the benchmark sweeps.
    zz_rates   : optional {edge: rad/ns}; unknown -> a typical 70 kHz on every edge.
    pulse_cost : infidelity charged per DD pulse in the staggering cost function.
    """
    info = DDInfo(strategy, family)
    out = prog.copy()
    if strategy == "none":
        return out, info
    if strategy not in ("uniform", "staggered"):
        raise ValueError(strategy)

    wins = [w for w in idle_windows(prog, edges) if w.L >= max(min_window_ns, 4 * tp_ns)]
    info.n_windows = len(wins)
    if not wins:
        return out, info
    nb = _neighbors(edges)
    J = {}
    for (a, b) in edges:
        r = (zz_rates or {}).get((a, b), (zz_rates or {}).get((b, a), DEFAULT_ZZ_RAD_NS))
        J[frozenset((a, b))] = r

    def n_nominal(w):
        cap = min(max_pulses, int(w.L // (2 * tp_ns)))
        n = 2 * max(1, round(w.L / tau_ns / 2.0))
        return max(2, min(n, cap - cap % 2)), cap

    gate_times: dict[int, list] = {}
    for e in prog.events:
        if e.tag == "circuit" and e.name not in ("rz", "id", "cz"):   # diagonal gates commute with ZZ
            for q in e.qubits:
                gate_times.setdefault(q, []).append(e.t)

    chosen: dict[int, tuple[str, int]] = {}
    toggles: dict[int, list[float]] = {}
    order = sorted(range(len(wins)), key=lambda i: -wins[i].L)

    for i in order:
        w = wins[i]
        n0, cap = n_nominal(w)
        if cap < 2:
            continue
        if strategy == "uniform":
            fr = _fractions("cpmg", n0)
            if not _feasible(fr, w.L, tp_ns):
                continue
            chosen[i] = ("cpmg", n0)
            toggles[i] = [w.t0 + f * w.L for f in fr]
            continue
        # ---- staggered: search candidates against already-placed neighbours
        nbr_q = [q2 for q2 in nb.get(w.q, ())]
        nb_wins = {q2: [(wins[j].t0, wins[j].t1, toggles[j]) for j in toggles if wins[j].q == q2]
                   for q2 in nbr_q}
        cands = sorted({c for c in (n0 - 2, n0, n0 + 2, n0 + 4, n0 + 6) if 2 <= c <= cap and c % 2 == 0})
        best = None
        for kind in ("cpmg", "udd"):
            for n in cands:
                fr = _fractions(kind, n)
                if not _feasible(fr, w.L, tp_ns):
                    continue
                tg = [w.t0 + f * w.L for f in fr]
                cost = pulse_cost * n
                for q2 in nbr_q:
                    cost += J[frozenset((w.q, q2))] ** 2 * _frame_cost(tg, w, nb_wins[q2], gate_times.get(q2, ()))
                # mild preference for the nominal spacing / plain CPMG on ties
                cost += 1e-9 * abs(n - n0) + (1e-10 if kind == "udd" else 0.0)
                if best is None or cost < best[0]:
                    best = (cost, kind, n, tg)
        if best is None:
            continue
        chosen[i] = (best[1], best[2])
        toggles[i] = best[3]

    order_id = 10_000_000
    for i, (kind, n) in sorted(chosen.items()):
        w = wins[i]
        for name, t_c in zip(pulse_train(family, n), toggles[i]):
            out.events.append(Event(t_c - tp_ns / 2, tp_ns, name, (w.q,), 0.0, "dd", order_id))
            order_id += 1
        info.n_protected += 1
        info.n_pulses += n
    info.n_colors = len({n for _, n in chosen.values()} | {k for k, _ in chosen.values()}) if chosen else 0
    return out, info


# ---------------------------------------------------------------------------
# Baseline: Qiskit's own PadDynamicalDecoupling pass (for honest comparison)
# ---------------------------------------------------------------------------
def qiskit_pad_dd(qc_basis, target, family: str = "XY4"):
    """Qiskit built-in DD on the same compiled circuit -> padded Qiskit circuit."""
    from qiskit.circuit.library import XGate, YGate
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling

    seqs = {"CPMG": [XGate(), XGate()], "XpXm": [XGate(), XGate()],
            "XY4": [XGate(), YGate(), XGate(), YGate()],
            "XY8": [XGate(), YGate(), XGate(), YGate(), YGate(), XGate(), YGate(), XGate()]}
    pm = PassManager([ALAPScheduleAnalysis(target=target),
                      PadDynamicalDecoupling(target=target, dd_sequence=seqs[family])])
    return pm.run(qc_basis)