"""End-to-end pipeline: workload -> compile -> (fold) -> schedule -> DD -> simulate -> metrics."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
from qiskit.quantum_info import Statevector

from .dd import DDInfo, inject_dd, qiskit_pad_dd
from .device import Device
from .idle import idle_report
from .mitigation import extrapolate, fold_cx
from .program import TimedProgram, compile_basis, program_from_circuit, schedule_circuit
from .simulate import energy, fidelity, pauli_expectations, simulate
from .workloads import Workload


@dataclass(frozen=True)
class Config:
    name: str
    dd: str = "none"              # none | qiskit | uniform | staggered | auto
    family: str = "XY4"           # CPMG | XpXm | XY4 | XY8
    tau_ns: float = 600.0
    min_window_ns: float = 250.0
    taus: tuple = (300.0, 600.0, 1200.0)   # candidate pulse spacings searched by dd="auto"
    zne: bool = False
    scales: tuple = (1.0, 2.0, 3.0)
    extrap: str = "linear"
    rem: bool = False             # calibrated readout-error mitigation (energy workloads)
    shots: int | None = None      # None = exact expectation values; int = emulate shot noise (energy)


@dataclass
class Result:
    workload: str
    n: int
    config: str
    fidelity: float                       # state fidelity vs ideal (exact, infinite shots)
    fidelity_clipped: float = 0.0         # clipped to [0, 1] (physical range)
    energy: float | None = None
    energy_ideal: float | None = None
    energy_error: float | None = None
    energy_error_std: float | None = None
    idle_pct: float = 0.0
    n_cx: int = 0
    n_dd_pulses: int = 0
    t_end_ns: float = 0.0
    seconds: float = 0.0
    raw: dict = field(default_factory=dict)

    def to_row(self):
        d = asdict(self)
        d.pop("raw")
        return d


class Compiled:
    """Compile once per (workload, device); reuse for every config / ZNE scale."""

    def __init__(self, wl: Workload, dev: Device):
        self.wl, self.dev = wl, dev
        self.target = dev.target()
        self.basis = compile_basis(wl.circuit, self.target, wl.opt_level)
        # ideal reference taken from the *compiled* circuit (same logical unitary)
        self.psi = Statevector(self.basis).data
        if wl.hamiltonian is not None:
            self.e_ideal = float(np.real(np.conj(self.psi) @ wl.hamiltonian.to_matrix() @ self.psi))
        else:
            self.e_ideal = None
        self._cache: dict = {}

    def _build(self, circ, scale, cfg, dd, tau):
        if dd == "qiskit":
            padded = qiskit_pad_dd(circ, self.target, cfg.family)
            prog = program_from_circuit(padded, self.target, f"{cfg.name}@{scale}")
            count = lambda c: sum(i.operation.name in ("x", "y") for i in c.data)
            prog.extra_dd = count(padded) - count(circ)
            return prog, None
        sched = schedule_circuit(circ, self.target)
        prog = program_from_circuit(sched, self.target, f"{cfg.name}@{scale}")
        info = None
        if dd in ("uniform", "staggered"):
            prog, info = inject_dd(prog, self.dev.edges, self.dev.tp_ns, dd, cfg.family, tau,
                                   cfg.min_window_ns, zz_rates=self.dev.zz_rates(),
                                   pulse_cost=self.dev.noise.p1q)
        return prog, info

    def choose_auto(self, cfg: Config, K: int = 16, seed: int = 99991) -> tuple[str, float, dict]:
        """Model-in-the-loop DD selection (ADAPT-style).

        Candidates: no DD, plus {uniform, staggered} x tau.  Each is scored with the
        calibrated device model on a *different* quasi-static noise sample (seed) than
        the one used for the final evaluation, and the best one is kept.  'none' is a
        candidate, so the pass cannot make a circuit worse (on the model).
        """
        key = ("auto", cfg.family, cfg.min_window_ns, cfg.taus)
        if key in self._cache:
            return self._cache[key]
        circ = self.basis
        scores = {}
        cands = [("none", 0.0)] + [(d, t) for d in ("uniform", "staggered") for t in cfg.taus]
        for d, t in cands:
            prog, _ = self._build(circ, 1.0, cfg, d, t)
            scores[(d, t)] = fidelity(simulate(prog, self.dev, K=K, seed=seed), self.psi)
        best = max(scores, key=scores.get)
        out = (best[0], best[1], {f"{d}@{t:g}": v for (d, t), v in scores.items()})
        self._cache[key] = out
        return out

    def program(self, scale: float, cfg: Config) -> tuple[TimedProgram, DDInfo | None]:
        dd, tau = cfg.dd, cfg.tau_ns
        if dd == "auto":
            dd, tau, _ = self.choose_auto(cfg)
        key = (scale, dd, cfg.family, tau, cfg.min_window_ns)
        if key in self._cache:
            return self._cache[key]
        out = self._build(fold_cx(self.basis, scale), scale, cfg, dd, tau)
        self._cache[key] = out
        return out


def _noisy_energy(rho, H, p_ro, rem, shots, rng):
    """Energy with emulated shot noise: each Pauli term ~ N(v, (1-v^2)/shots)."""
    e = 0.0
    for w, c, v in pauli_expectations(rho, H):
        damp = 1.0 if rem else (1.0 - 2.0 * p_ro) ** w
        m = v * damp
        m += rng.standard_normal() * np.sqrt(max(1 - m * m, 0.0) / shots)
        e += c * (m / ((1.0 - 2.0 * p_ro) ** w) if rem else m)
    return e


def evaluate(comp: Compiled, cfg: Config, K: int = 32, seed: int = 1234, repeats: int = 40) -> Result:
    t0 = time.time()
    wl, dev = comp.wl, comp.dev
    scales = cfg.scales if cfg.zne else (1.0,)
    fids, ens, progs, rhos = [], [], [], []
    for s in scales:
        prog, _ = comp.program(s, cfg)
        rho = simulate(prog, dev, K=K, seed=seed)
        fids.append(fidelity(rho, comp.psi))
        if wl.hamiltonian is not None:
            ens.append(energy(rho, wl.hamiltonian, dev.noise.p_readout, cfg.rem))
        progs.append(prog)
        rhos.append(rho)
    if cfg.zne:
        fid = extrapolate(scales, fids, cfg.extrap)
        en = extrapolate(scales, ens, cfg.extrap) if ens else None
    else:
        fid, en = fids[0], (ens[0] if ens else None)
    err = abs(en - comp.e_ideal) if en is not None else None
    err_std = None
    if cfg.shots and wl.hamiltonian is not None:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(repeats):
            es = [_noisy_energy(r, wl.hamiltonian, dev.noise.p_readout, cfg.rem, cfg.shots, rng) for r in rhos]
            draws.append(extrapolate(scales, es, cfg.extrap) if cfg.zne else es[0])
        draws = np.asarray(draws)
        en = float(draws.mean())
        err = float(np.sqrt(np.mean((draws - comp.e_ideal) ** 2)))     # RMSE incl. variance
        err_std = float(draws.std())
    base = progs[0]
    rep = idle_report(base, dev.edges)
    return Result(wl.name, wl.n, cfg.name, fid, float(np.clip(fid, 0, 1)), en, comp.e_ideal, err, err_std,
                  rep["idle_pct"], base.n_cx, base.n_dd_pulses, base.t_end, time.time() - t0,
                  {"scales": list(scales), "fid_by_scale": fids,
                   "auto_choice": (lambda c: {"strategy": c[0], "tau_ns": c[1], "scores": c[2]})(comp.choose_auto(cfg))
                   if cfg.dd == "auto" else None})


# ------------------------------------------------------------ standard configs
def standard_configs(tau_ns: float = 600.0, family: str = "XY4", shots: int | None = None) -> list[Config]:
    kw = dict(family=family, shots=shots)
    return [
        Config("Baseline", shots=shots),
        Config("Qiskit DD", dd="qiskit", **kw),
        Config("DD uniform", dd="uniform", tau_ns=tau_ns, **kw),
        Config("DD staggered (ours)", dd="staggered", tau_ns=tau_ns, **kw),
        Config("DD auto (ours)", dd="auto", **kw),
        Config("ZNE only", zne=True, shots=shots),
        Config("Auto DD + ZNE (ours)", dd="auto", zne=True, **kw),
        Config("Auto DD + ZNE + REM (ours)", dd="auto", zne=True, rem=True, **kw),
    ]
