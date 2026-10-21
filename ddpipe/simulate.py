"""Exact timeline simulator (density matrix, ensemble over quasi-static noise).

Time is cut into slices at every event boundary.  Gates are applied as
instantaneous unitaries (+ depolarising error) at their start time; between
boundaries every qubit evolves under

   H = sum_q  delta_q/2 Z_q  +  sum_(a,b)  J_ab/2 Z_a Z_b          (coherent)
   + amplitude damping (T1) + pure dephasing (Tphi)                  (Markov)

delta_q is a quasi-static random detuning, sampled per ensemble member with
scrambled-Sobol + antithetic points (low variance, identical across configs
so comparisons are paired).  Because DD pulses are real events in the
timeline, X/Y pulses flip the sign of Z-type terms exactly as on hardware -
refocusing is *emergent*, not assumed.
"""
from __future__ import annotations

import functools

import numpy as np
from scipy.stats import norm, qmc

from .device import Device
from .program import TimedProgram


# ----------------------------------------------------------------- helpers
def _rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _embed1(u2, q, n):
    return np.kron(np.kron(np.eye(2 ** (n - 1 - q)), u2), np.eye(2 ** q))


@functools.lru_cache(maxsize=None)
def _zsigns(n):
    z = np.arange(2 ** n)
    return np.stack([1 - 2 * ((z >> q) & 1) for q in range(n)]).astype(float)   # (n, d)


@functools.lru_cache(maxsize=None)
def _cx_perm(c, t, n):
    z = np.arange(2 ** n)
    return np.where((z >> c) & 1, z ^ (1 << t), z)


def sample_detunings(n, K, sigma, seed):
    """K x n standard-normal quasi-random samples (antithetic) times sigma."""
    half = K // 2
    m = max(1, int(np.ceil(np.log2(max(half, 2)))))
    u = qmc.Sobol(d=n, scramble=True, seed=seed).random_base2(m)[:half]
    z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    z = np.concatenate([z, -z])
    return z * sigma[None, :]


def _depolarize(rho, qubits, lam, n):
    """rho <- (1-lam) rho + lam * Tr_qubits(rho) (x) I/2^m  (batched over axis 0)."""
    if lam <= 0:
        return rho
    K = rho.shape[0]
    t = rho.reshape((K,) + (2,) * (2 * n))
    letters = [chr(ord("a") + i) for i in range(2 * n)]
    row = {q: letters[n - 1 - q] for q in range(n)}
    col = {q: letters[n + n - 1 - q] for q in range(n)}
    full = "Z" + "".join(letters)
    # traced tensor: row letter == col letter for traced qubits
    in_sub = ["Z"] + [letters[i] for i in range(2 * n)]
    for q in qubits:
        in_sub[1 + n + (n - 1 - q)] = row[q]
    out_sub = "Z" + "".join(l for i, l in enumerate(letters) if not (
        (i < n and (n - 1 - i) in qubits) or (i >= n and (n - 1 - (i - n)) in qubits)))
    tr = np.einsum("".join(in_sub) + "->" + out_sub, t)
    ops, subs = [tr], [out_sub]
    for q in qubits:
        ops.append(np.eye(2)); subs.append(row[q] + col[q])
    rep = np.einsum(",".join(subs) + "->" + full, *ops) / (2 ** len(qubits))
    return ((1 - lam) * t + lam * rep).reshape(rho.shape)


def _relax(rho, q, n, dt, t1, t2):
    """T1 amplitude damping + pure dephasing on qubit q (in place)."""
    K = rho.shape[0]
    r = rho.reshape(K, 2 ** (n - 1 - q), 2, 2 ** q, 2 ** (n - 1 - q), 2, 2 ** q)
    g = 1.0 - np.exp(-dt / t1)
    rate_phi = max(1.0 / t2 - 0.5 / t1, 0.0)
    coh = np.sqrt(1.0 - g) * np.exp(-dt * rate_phi)
    r[:, :, 0, :, :, 0, :] += g * r[:, :, 1, :, :, 1, :]
    r[:, :, 1, :, :, 1, :] *= 1.0 - g
    r[:, :, 0, :, :, 1, :] *= coh
    r[:, :, 1, :, :, 0, :] *= coh


# ------------------------------------------------------------------ main
def simulate(prog: TimedProgram, dev: Device, K: int = 32, seed: int = 1234,
             ideal: bool = False) -> np.ndarray:
    """Return the ensemble-averaged density matrix at ``prog.t_end``.

    ideal=True switches every noise source off (used by tests / sanity checks).
    """
    n, d = prog.n, 2 ** prog.n
    pq, p = dev.per_qubit(), dev.noise
    zz = dev.zz_rates()
    S = _zsigns(n)

    if ideal:
        Keff, delta = 1, np.zeros((1, n))
    else:
        Keff = K
        delta = sample_detunings(n, K, pq["sigma_rad_ns"], seed)
    base = 0.5 * delta @ S                                   # (K, d)

    rho = np.zeros((Keff, d, d), dtype=complex)
    rho[:, 0, 0] = 1.0

    events = prog.sorted_events()
    bounds = sorted({0.0, prog.t_end} | {e.t for e in events} | {e.end for e in events})
    ev_i = 0
    running_cx: list = []
    zz_cache: dict = {}
    eps = 0.0 if ideal else p.pulse_overrot

    for bi in range(len(bounds) - 1):
        t0, t1 = bounds[bi], bounds[bi + 1]
        # ---- instantaneous gates starting at t0
        while ev_i < len(events) and events[ev_i].t <= t0 + 1e-9:
            e = events[ev_i]; ev_i += 1
            q = e.qubits[0]
            if e.name == "rz":
                ph = np.exp(-0.5j * e.param * S[q])
                rho *= ph[None, :, None] * np.conj(ph)[None, None, :]
                continue
            if e.name == "cx":
                perm = _cx_perm(e.qubits[0], e.qubits[1], n)
                rho = rho[:, perm][:, :, perm]
                if not ideal:
                    rho = _depolarize(rho, tuple(e.qubits), p.p2q, n)
                if e.dur > 0:
                    running_cx.append((e.end, frozenset(e.qubits)))
                continue
            amp = 1.0 + eps
            u2 = {"x": _rx(np.pi * amp), "xm": _rx(-np.pi * amp), "y": _ry(np.pi * amp),
                  "sx": _rx(np.pi / 2 * amp)}[e.name]
            U = _embed1(u2, q, n)
            rho = U @ rho @ U.conj().T
            if not ideal:
                rho = _depolarize(rho, (q,), p.p1q, n)
        dt = t1 - t0
        if dt <= 1e-9 or ideal:
            running_cx = [(te, qs) for te, qs in running_cx if te > t1 + 1e-9]
            continue
        running_cx = [(te, qs) for te, qs in running_cx if te > t0 + 1e-9]
        skip = frozenset(qs for _, qs in running_cx)
        if skip not in zz_cache:
            w = np.zeros(d)
            for (a, b), J in zz.items():
                if frozenset((a, b)) in skip:
                    continue
                w += 0.5 * J * S[a] * S[b]
            zz_cache[skip] = w
        wv = base + zz_cache[skip][None, :]
        ph = np.exp(-1j * dt * wv)                           # (K, d)
        rho *= ph[:, :, None] * np.conj(ph)[:, None, :]
        for q in range(n):
            _relax(rho, q, n, dt, pq["t1_ns"][q], pq["t2_ns"][q])
    return rho.mean(axis=0)


# ------------------------------------------------------------- observables
def fidelity(rho: np.ndarray, psi: np.ndarray) -> float:
    return float(np.real(np.conj(psi) @ rho @ psi))


def pauli_expectations(rho: np.ndarray, op) -> list[tuple[int, float, float]]:
    """[(weight, coeff, <P>)] for every term of a SparsePauliOp."""
    from qiskit.quantum_info import Pauli
    out = []
    for label, coeff in zip(op.paulis.to_labels(), op.coeffs):
        val = float(np.real(np.trace(Pauli(label).to_matrix() @ rho)))
        out.append((sum(ch != "I" for ch in label), float(np.real(coeff)), val))
    return out


def energy(rho: np.ndarray, op, p_readout: float = 0.0, readout_mitigated: bool = False) -> float:
    """<H> including symmetric readout error (weight-w term damped by (1-2p)^w)."""
    e = 0.0
    for w, c, v in pauli_expectations(rho, op):
        damp = 1.0 if readout_mitigated else (1.0 - 2.0 * p_readout) ** w
        e += c * v * damp
    return e
