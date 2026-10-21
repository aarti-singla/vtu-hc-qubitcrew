"""Benchmark workloads: circuits + ideal reference + metric definitions."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize


@dataclass
class Workload:
    name: str
    n: int
    circuit: QuantumCircuit
    hamiltonian: SparsePauliOp | None = None
    opt_level: int = 1
    meta: dict = field(default_factory=dict)

    @property
    def metric(self) -> str:
        return "energy_error" if self.hamiltonian is not None else "fidelity"


# ------------------------------------------------------------------ GHZ
def ghz(n: int) -> Workload:
    qc = QuantumCircuit(n, name=f"ghz{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return Workload(f"GHZ-{n}", n, qc)


# ------------------------------------------------------------ QAOA ring
def _ring_cost(n):
    return SparsePauliOp.from_list([(_zz_label(n, i, (i + 1) % n), 1.0) for i in range(n)])


def _zz_label(n, a, b):
    s = ["I"] * n
    s[n - 1 - a] = "Z"
    s[n - 1 - b] = "Z"
    return "".join(s)


def _qaoa_circuit(n, gamma, beta):
    qc = QuantumCircuit(n, name=f"qaoa{n}")
    qc.h(range(n))
    for i in range(n):
        qc.rzz(2 * gamma, i, (i + 1) % n)
    for i in range(n):
        qc.rx(2 * beta, i)
    return qc


def qaoa_ring(n: int) -> Workload:
    """p=1 QAOA for MaxCut on a ring; cost H = sum Z_i Z_{i+1} (minimise)."""
    H = _ring_cost(n)
    best = (1e9, 0.0, 0.0)
    for g in np.linspace(0, np.pi, 33):
        for b in np.linspace(0, np.pi / 2, 17):
            e = float(np.real(Statevector(_qaoa_circuit(n, g, b)).expectation_value(H)))
            if e < best[0]:
                best = (e, g, b)
    return Workload(f"QAOA-ring-{n}", n, _qaoa_circuit(n, best[1], best[2]), H,
                    meta={"gamma": best[1], "beta": best[2], "ideal_cost": best[0]})


# -------------------------------------------------------- VQE (TFIM chain)
def _tfim(n, h=0.8):
    terms = [(_zz_label(n, i, i + 1), -1.0) for i in range(n - 1)]
    for i in range(n):
        s = ["I"] * n
        s[n - 1 - i] = "X"
        terms.append(("".join(s), -h))
    return SparsePauliOp.from_list(terms)


def _ansatz(n, params, reps):
    qc = QuantumCircuit(n, name=f"vqe{n}")
    k = 0
    for _ in range(reps):
        for q in range(n):
            qc.ry(params[k], q); k += 1
        for q in range(n - 1):
            qc.cx(q, q + 1)
    for q in range(n):
        qc.ry(params[k], q); k += 1
    return qc


def vqe_tfim(n: int, reps: int = 2, seed: int = 3) -> Workload:
    """Hardware-efficient ansatz with parameters optimised *ideally* (statevector)."""
    H = _tfim(n)
    npar = n * (reps + 1)
    rng = np.random.default_rng(seed)
    f = lambda p: float(np.real(Statevector(_ansatz(n, p, reps)).expectation_value(H)))
    res = minimize(f, rng.uniform(-0.5, 0.5, npar), method="L-BFGS-B", options={"maxiter": 200})
    exact = float(np.linalg.eigvalsh(H.to_matrix())[0])
    return Workload(f"VQE-TFIM-{n}", n, _ansatz(n, res.x, reps), H,
                    meta={"ideal_energy": res.fun, "exact_ground": exact, "reps": reps})


# ------------------------------------------------------------ mirror circuit
def mirror(n: int, layers: int = 4, seed: int = 1) -> Workload:
    """Random Clifford-style layers followed by their exact inverse -> |0...0>.

    Each layer applies 1q Cliffords to a random half of the qubits and CX on one
    random neighbouring pair, so the other qubits sit idle (realistic for
    sparse circuits and gives the idle analyzer something to find).
    """
    rng = np.random.default_rng(seed)
    one = ["h", "s", "sdg", "x", "y", "z", "sx"]
    half = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            if rng.random() < 0.5:
                getattr(half, one[rng.integers(len(one))])(q)
        a = int(rng.integers(n - 1))
        half.cx(a, a + 1)
    qc = QuantumCircuit(n, name=f"mirror{n}")
    qc.compose(half, inplace=True)
    qc.barrier()
    qc.compose(half.inverse(), inplace=True)
    return Workload(f"Mirror-{n}x{layers}", n, qc, opt_level=0)


# ----------------------------------------------- GHZ with a mid-circuit wait
def ghz_wait(n: int, wait_ns: float = 1500.0) -> Workload:
    """GHZ -> synchronous wait -> un-compute  (ideal output |0...0>).

    The wait models what dynamic circuits suffer from (mid-circuit measurement
    + feed-forward latency): *all* qubits idle at the same time, so neighbouring
    idle windows coincide and ZZ crosstalk is maximal unless DD is staggered.
    """
    from .device import DT_NS
    half = QuantumCircuit(n)
    half.h(0)
    for i in range(n - 1):
        half.cx(i, i + 1)
    qc = QuantumCircuit(n, name=f"ghzwait{n}")
    qc.compose(half, inplace=True)
    qc.barrier()
    for q in range(n):
        qc.delay(int(round(wait_ns / DT_NS)), q, unit="dt")
    qc.barrier()
    qc.compose(half.inverse(), inplace=True)
    return Workload(f"GHZ-wait-{n}", n, qc, opt_level=0, meta={"wait_ns": wait_ns})


REGISTRY = {"ghz": ghz, "qaoa": qaoa_ring, "vqe": vqe_tfim, "mirror": mirror, "ghzwait": ghz_wait}


def build(kind: str, n: int, **kw) -> Workload:
    return REGISTRY[kind](n, **kw)
