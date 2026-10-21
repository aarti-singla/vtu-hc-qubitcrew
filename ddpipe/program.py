"""Timed program representation + bridge to Qiskit scheduled circuits.

Everything downstream (idle analysis, DD injection, simulation, plotting)
works on a flat list of timed events.  ``program_from_circuit`` accepts *any*
Qiskit circuit whose idle gaps are explicit ``Delay`` instructions, i.e. the
output of ``transpile(..., scheduling_method="alap")`` **or** of Qiskit's own
``PadDynamicalDecoupling`` pass - which lets us benchmark against Qiskit's
built-in DD with exactly the same simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import Target

from .device import DT_NS

GATE_NAMES = {"rz", "sx", "x", "y", "cx", "ecr", "cz", "id"}
TWO_Q_NAMES = {"cx", "ecr", "cz"}


@dataclass
class Event:
    t: float                    # start time [ns]
    dur: float                  # duration [ns]
    name: str                   # rz | sx | x | y | xm | cx
    qubits: tuple
    param: float = 0.0          # rz angle
    tag: str = "circuit"        # "circuit" | "dd"
    order: int = 0              # stable tie-break

    @property
    def end(self) -> float:
        return self.t + self.dur


@dataclass
class TimedProgram:
    n: int
    events: list
    t_end: float
    label: str = ""
    extra_dd: int = 0            # DD pulses present in the circuit but not tagged (Qiskit's own pass)
    dt_ns: float = DT_NS         # length of one backend `dt` [ns]; read from the target (IBM Heron: 4 ns)

    def copy(self) -> "TimedProgram":
        return TimedProgram(self.n, [replace(e) for e in self.events], self.t_end, self.label, self.extra_dd,
                            self.dt_ns)

    def sorted_events(self):
        return sorted(self.events, key=lambda e: (round(e.t, 6), e.order))

    @property
    def n_cx(self) -> int:
        return sum(1 for e in self.events if e.name == "cx")

    @property
    def n_dd_pulses(self) -> int:
        return sum(1 for e in self.events if e.tag == "dd") + self.extra_dd


# --------------------------------------------------------------------------
def compile_basis(qc: QuantumCircuit, target: Target, optimization_level: int = 1, seed: int = 11) -> QuantumCircuit:
    """Map + translate to the device basis (no scheduling).  Trivial layout."""
    n = target.num_qubits
    tc = transpile(qc, target=target, initial_layout=list(range(qc.num_qubits)),
                   optimization_level=optimization_level, seed_transpiler=seed)
    fl = getattr(tc.layout, "final_layout", None) if tc.layout is not None else None
    if fl is not None:
        perm = [fl[tc.qubits[i]] for i in range(tc.num_qubits)]
        idx = [tc.find_bit(q).index for q in perm]
        if idx != list(range(tc.num_qubits)):
            raise ValueError("routing introduced a qubit permutation; use a ring topology or a "
                             "circuit that respects the coupling map")
    return tc


def schedule_circuit(qc_basis: QuantumCircuit, target: Target) -> QuantumCircuit:
    """ALAP-schedule an already-compiled circuit (explicit Delay padding)."""
    return transpile(qc_basis, target=target, optimization_level=0,
                     scheduling_method="alap", initial_layout=list(range(qc_basis.num_qubits)))


def program_from_circuit(circ: QuantumCircuit, target: Target, label: str = "") -> TimedProgram:
    """Walk a delay-padded circuit with per-qubit clocks -> TimedProgram."""
    n = circ.num_qubits
    dt_ns = float(target.dt) * 1e9 if getattr(target, "dt", None) else DT_NS
    clock = np.zeros(n)
    events: list[Event] = []
    for k, inst in enumerate(circ.data):
        op = inst.operation
        qs = tuple(circ.find_bit(q).index for q in inst.qubits)
        name = op.name
        if name == "delay":
            d = float(op.duration)
            unit = getattr(op, "unit", "dt")
            d_ns = d * dt_ns if unit == "dt" else {"s": d * 1e9, "ns": d, "us": d * 1e3, "ms": d * 1e6}[unit]
            for q in qs:
                clock[q] += d_ns
            continue
        if name == "barrier":
            m = max(clock[q] for q in qs)
            for q in qs:
                clock[q] = m
            continue
        if name not in GATE_NAMES:
            raise ValueError(f"unsupported instruction {name!r} (compile to rz/sx/x/y/cx first)")
        dur = target[name][qs].duration * 1e9
        t0 = max(clock[q] for q in qs)
        events.append(Event(t0, dur, name, qs, float(op.params[0]) if name == "rz" else 0.0, "circuit", k))
        for q in qs:
            clock[q] = t0 + dur
    return TimedProgram(n, events, float(clock.max()) if n else 0.0, label, 0, dt_ns)


def qiskit_circuit_from_program(prog: TimedProgram, granularity: int = 1) -> QuantumCircuit:
    """Export a TimedProgram back to a Qiskit circuit with explicit delays.

    Only IBM-native single-qubit gates are emitted: ``xm`` (pi pulse about -x) becomes
    rz(pi) x rz(-pi) and ``y`` becomes rz(-pi/2) x rz(pi/2) (virtual Z, free on hardware).
    Delay lengths are rounded to multiples of ``granularity`` dt (IBM backends need 16).
    """
    qc = QuantumCircuit(prog.n)
    dt_ns = prog.dt_ns
    clock = np.zeros(prog.n)                 # ns, advanced by the *rounded* delays actually emitted
    for e in prog.sorted_events():
        for q in e.qubits:
            gap = int(round((e.t - clock[q]) / dt_ns)) if e.t > clock[q] else 0
            gap = (gap // granularity) * granularity
            if gap > 0:
                qc.delay(gap, q, unit="dt")
                clock[q] += gap * dt_ns
        q0 = e.qubits[0]
        if e.name == "rz":
            qc.rz(e.param, q0)
        elif e.name == "sx":
            qc.sx(q0)
        elif e.name == "x":
            qc.x(q0)
        elif e.name == "y":
            qc.rz(-np.pi / 2, q0); qc.x(q0); qc.rz(np.pi / 2, q0)
        elif e.name == "xm":
            qc.rz(np.pi, q0); qc.x(q0); qc.rz(-np.pi, q0)
        elif e.name == "cx":
            qc.cx(*e.qubits)
        elif e.name == "cz":
            qc.cz(*e.qubits)
        elif e.name == "ecr":
            qc.ecr(*e.qubits)
        elif e.name == "id":
            qc.id(q0)
        start = max(clock[q] for q in e.qubits)
        for q in e.qubits:
            clock[q] = start + e.dur
    return qc