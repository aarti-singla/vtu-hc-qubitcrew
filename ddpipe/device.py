"""Synthetic superconducting-device model.

The model carries two things:
  * a Qiskit ``Target`` (basis gates, coupling map, gate durations) used for
    transpilation and ALAP scheduling, and
  * a physical noise description used by our timeline simulator.

Why this noise model?  Dynamical decoupling only suppresses *low-frequency /
coherent* noise.  A purely Markovian (Lindblad) T1/T2 model would show **zero**
DD benefit, so we model the three mechanisms that DD is known to fight on
IBM-style fixed-frequency transmons:

  1. quasi-static qubit-frequency noise  (slow detuning drift, refocusable)
  2. always-on ZZ crosstalk between coupled qubits (coherent, refocusable
     only if neighbouring DD sequences are *staggered*)
  3. imperfect DD pulses (amplitude error + depolarising error), which is what
     makes "more pulses" a trade-off instead of a free lunch.

Markovian T1 / white dephasing and gate depolarising errors are included too,
and are *not* suppressed by DD - so gains stay realistic.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from qiskit.circuit import Delay, Parameter
from qiskit.circuit.library import CXGate, RZGate, SXGate, XGate, YGate
from qiskit.transpiler import InstructionProperties, Target

DT_S = 1.0 / 4.5e9            # IBM-like sample time (~0.222 ns)
DT_NS = DT_S * 1e9
T1Q_DT = 160                  # 35.6 ns single-qubit pulse
T2Q_DT = 1344                 # ~298.7 ns CX


def make_edges(topology: str, n: int) -> tuple[tuple[int, int], ...]:
    if topology == "line" or n <= 2:
        return tuple((i, i + 1) for i in range(n - 1))
    if topology == "ring":
        return tuple((i, i + 1) for i in range(n - 1)) + ((n - 1, 0),)
    raise ValueError(f"unknown topology {topology!r}")


@dataclass(frozen=True)
class NoiseParams:
    t1_us: float = 120.0          # mean T1
    t2_us: float = 100.0          # mean T2 (echo); Tphi derived from T1,T2
    sigma_f_khz: float = 25.0     # std of quasi-static detuning (per qubit)
    zz_khz: float = 70.0          # mean always-on ZZ rate on every coupled pair
    zz_spread: float = 0.25       # relative spread of ZZ between pairs
    p1q: float = 3e-4             # depolarising prob. per 1q pulse
    p2q: float = 7e-3             # depolarising prob. per CX
    pulse_overrot: float = 0.004  # fractional amplitude error of x/sx/y pulses
    p_readout: float = 0.015      # symmetric readout flip probability
    spread: float = 0.15          # per-qubit relative spread of T1/T2/sigma


@dataclass(frozen=True)
class Device:
    n: int
    topology: str = "ring"
    noise: NoiseParams = field(default_factory=NoiseParams)
    seed: int = 7

    # ---- derived static properties ------------------------------------
    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return make_edges(self.topology, self.n)

    @property
    def tp_ns(self) -> float:
        return T1Q_DT * DT_NS

    @property
    def t2q_ns(self) -> float:
        return T2Q_DT * DT_NS

    def _rng(self):
        return np.random.default_rng(self.seed)

    def per_qubit(self) -> dict[str, np.ndarray]:
        """Per-qubit T1/T2/sigma (seeded, reproducible)."""
        rng, p = self._rng(), self.noise
        jit = lambda m: np.clip(m * (1 + p.spread * rng.standard_normal(self.n)), 0.3 * m, 2.0 * m)
        t1 = jit(p.t1_us) * 1e3
        t2 = np.minimum(jit(p.t2_us) * 1e3, 1.95 * t1)
        sig = jit(p.sigma_f_khz)
        # frequency (kHz) -> angular rate (rad/ns)
        return {"t1_ns": t1, "t2_ns": t2, "sigma_rad_ns": 2 * np.pi * sig * 1e-6}

    def zz_rates(self) -> dict[tuple[int, int], float]:
        """ZZ conditional-phase rate per edge in rad/ns."""
        rng = np.random.default_rng(self.seed + 1)
        p = self.noise
        out = {}
        for e in self.edges:
            f = p.zz_khz * (1 + p.zz_spread * rng.standard_normal())
            out[e] = 2 * np.pi * max(f, 0.0) * 1e-6
        return out

    def with_noise(self, **kw) -> "Device":
        return replace(self, noise=replace(self.noise, **kw))

    # ---- Qiskit target --------------------------------------------------
    def target(self) -> Target:
        return make_target(self.n, self.edges, self.noise.p1q, self.noise.p2q)


def make_target(n: int, edges, p1q: float = 3e-4, p2q: float = 7e-3) -> Target:
    t = Target(num_qubits=n, dt=DT_S)
    q1 = lambda dur, err: {(q,): InstructionProperties(duration=dur * DT_S, error=err) for q in range(n)}
    t.add_instruction(RZGate(Parameter("theta")), q1(0, 0.0))
    t.add_instruction(SXGate(), q1(T1Q_DT, p1q))
    t.add_instruction(XGate(), q1(T1Q_DT, p1q))
    t.add_instruction(YGate(), q1(T1Q_DT, p1q))
    both = {}
    for a, b in edges:
        both[(a, b)] = InstructionProperties(duration=T2Q_DT * DT_S, error=p2q)
        both[(b, a)] = InstructionProperties(duration=T2Q_DT * DT_S, error=p2q)
    t.add_instruction(CXGate(), both)
    t.add_instruction(Delay(Parameter("t")), {(q,): None for q in range(n)})
    return t
