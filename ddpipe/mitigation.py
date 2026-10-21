"""Zero-noise extrapolation by CX unitary folding + extrapolators."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


def fold_cx(circ: QuantumCircuit, scale: float) -> QuantumCircuit:
    """Local unitary folding of CX gates:  CX -> CX (CX CX)^k.

    Noise from the two-qubit layer is amplified by ~``scale``; fractional
    scales fold a subset of the CX gates (chosen evenly).  The circuit stays
    logically identical (CX CX = I).
    """
    if scale < 1:
        raise ValueError("scale must be >= 1")
    cx_idx = [i for i, ins in enumerate(circ.data) if ins.operation.name == "cx"]
    if scale == 1 or not cx_idx:
        return circ.copy()
    extra = (scale - 1.0) / 2.0 * len(cx_idx)               # number of (CX CX) pairs to add
    full, rem = int(extra // len(cx_idx)), extra - (extra // len(cx_idx)) * len(cx_idx)
    n_extra_gates = int(round(rem))
    chosen = set(np.linspace(0, len(cx_idx) - 1, n_extra_gates, dtype=int).tolist()) if n_extra_gates else set()
    out = circ.copy_empty_like()
    k = -1
    for ins in circ.data:
        out.append(ins.operation, ins.qubits, ins.clbits)
        if ins.operation.name == "cx":
            k += 1
            for _ in range(full + (1 if k in chosen else 0)):
                out.append(ins.operation, ins.qubits)
                out.append(ins.operation, ins.qubits)
    return out


def extrapolate(scales, values, method: str = "richardson") -> float:
    """Extrapolate (scale, value) pairs to scale 0."""
    x, y = np.asarray(scales, float), np.asarray(values, float)
    if method == "linear":
        return float(np.polyval(np.polyfit(x, y, 1), 0.0))
    if method == "quadratic" and len(x) >= 3:
        return float(np.polyval(np.polyfit(x, y, 2), 0.0))
    if method == "richardson" or method == "quadratic":
        # exact polynomial interpolation through all points (Lagrange at 0)
        total = 0.0
        for i in range(len(x)):
            li = np.prod([(0 - x[j]) / (x[i] - x[j]) for j in range(len(x)) if j != i])
            total += y[i] * li
        return float(total)
    if method == "exponential":
        # fit y = a * exp(-b x) on log|y| when all values share a sign and are non-zero
        if np.all(y > 1e-9):
            b, a = np.polyfit(x, np.log(y), 1)
            return float(np.exp(a))
        return float(np.polyval(np.polyfit(x, y, 1), 0.0))
    raise ValueError(method)
