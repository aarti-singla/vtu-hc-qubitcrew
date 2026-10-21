"""Run the GHZ benchmark on a real IBM backend (or a local fake backend for a dry run).

Usage
-----
  # credentials: save once with QiskitRuntimeService.save_account(...)  or  export QISKIT_IBM_TOKEN=...
  python scripts/run_hardware.py --n 4 --shots 4000                    # least-busy real device
  python scripts/run_hardware.py --backend ibm_brisbane --n 4
  python scripts/run_hardware.py --fake --n 4                          # offline plumbing test (Aer + noise model)

Configurations compared (state fidelity of an n-qubit GHZ state, measured through the
Estimator on the GHZ projector, which is 2^n Pauli terms):
  1. baseline                     no suppression, no mitigation
  2. Runtime DD                   Qiskit Runtime built-in DD (XY4)
  3. our DD                       idle-aware staggered DD injected by ddpipe (Runtime DD off)
  4. Runtime ZNE (+twirling, TREX)  resilience_level = 2
  5. our DD + Runtime ZNE         our circuit + resilience_level = 2

NEVER hard-code or commit your API token.
Results -> results/hardware.json (shown by the website's Hardware tab).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector  # noqa: E402
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # noqa: E402

from ddpipe import inject_dd, qiskit_circuit_from_program  # noqa: E402
from ddpipe.idle import idle_report  # noqa: E402
from ddpipe.program import program_from_circuit, schedule_circuit  # noqa: E402


def ghz(n, wait_dt=0):
    """GHZ preparation, optionally followed by an idle wait on every qubit.

    A bare 4-qubit GHZ circuit lasts only ~350 ns on current IBM devices, which leaves no idle window
    long enough for DD. The wait (--wait-ns) models a circuit that holds a state while other work
    happens (mid-circuit feed-forward, slow readout of neighbours, ...), which is where DD matters.
    """
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    if wait_dt:
        qc.barrier()
        for q in range(n):
            qc.delay(int(wait_dt), q)
        qc.barrier()
    return qc


def build_circuits(backend, n, strategy, tau_ns, granularity, wait_ns=0.0, max_pulses=16):
    wait_dt = int(round(wait_ns * 1e-9 / backend.target.dt / granularity)) * granularity
    logical = ghz(n, wait_dt)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend, seed_transpiler=7)
    isa = pm.run(logical)
    proj = SparsePauliOp.from_operator(DensityMatrix(Statevector(ghz(n)))).simplify()
    proj = proj.apply_layout(isa.layout)
    target = backend.target
    sched = schedule_circuit(isa, target)
    prog = program_from_circuit(sched, target, "hw")
    edges = sorted({tuple(sorted(e)) for e in backend.coupling_map.get_edges()})
    used = sorted({q for e in prog.events for q in e.qubits})
    tp = float(np.mean([target["x"][(q,)].duration for q in used])) * 1e9
    dd_prog, info = inject_dd(prog, edges, tp, strategy, "XpXm", tau_ns, 250.0, max_pulses=max_pulses)
    ours = qiskit_circuit_from_program(dd_prog, granularity=granularity)
    return isa, ours, proj, idle_report(prog, edges), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--fake", action="store_true", help="offline dry run on a fake backend (Aer)")
    ap.add_argument("--strategy", default="staggered", choices=["staggered", "uniform"])
    ap.add_argument("--tau", type=float, default=600.0)
    ap.add_argument("--granularity", type=int, default=0, help="delay granularity in dt (0 = read from the backend)")
    ap.add_argument("--wait-ns", type=float, default=0.0,
                    help="idle wait after GHZ prep, in ns (try 5000-20000 so DD has windows to fill)")
    ap.add_argument("--max-pulses", type=int, default=16, help="cap on DD pulses per idle window")
    ap.add_argument("--no-zne", action="store_true",
                    help="skip the two ZNE jobs (they cost ~4x the QPU time of a DD job)")
    ap.add_argument("--out", default="results/hardware.json")
    a = ap.parse_args()

    if a.fake:
        from qiskit_aer import AerSimulator
        from qiskit.primitives import BackendEstimatorV2
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        fb = FakeSherbrooke()
        backend = fb
        sim = AerSimulator.from_backend(fb)
        mk = lambda opts: BackendEstimatorV2(backend=sim, options={"default_precision": 1 / np.sqrt(a.shots)})
        name = "FakeSherbrooke (Aer, dry run)"
    else:
        from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService
        service = QiskitRuntimeService()
        backend = (service.backend(a.backend) if a.backend else
                   service.least_busy(operational=True, simulator=False, min_num_qubits=max(a.n, 5)))
        name = backend.name
        mk = None

    gran = a.granularity or max(16, int(getattr(backend.target, "granularity", 0) or 0))
    print(f"dt={backend.target.dt * 1e9:.3f} ns  delay granularity={gran} dt")
    isa, ours, proj, rep, info = build_circuits(backend, a.n, a.strategy, a.tau, gran, a.wait_ns, a.max_pulses)
    print(f"backend={name}  idle windows={rep['n_windows']}  idle={rep['idle_pct']:.0f}%  DD pulses={info.n_pulses}")

    rows, job_ids = [], []

    def run(label, circuit, options):
        if a.fake:
            est = mk(options)
            res = est.run([(circuit, proj)]).result()[0]
            jid = "local"
        else:
            est = EstimatorV2(mode=backend)
            for k, v in options.items():
                obj = est.options
                *path, last = k.split(".")
                for p in path:
                    obj = getattr(obj, p)
                setattr(obj, last, v)
            est.options.default_shots = a.shots
            job = est.run([(circuit, proj)])
            jid = job.job_id()
            print(f"  submitted {label}: {jid}")
            res = job.result()[0]
        ev, sd = float(np.real(res.data.evs)), float(np.real(res.data.stds))
        rows.append({"config": label, "fidelity": ev, "stderr": sd, "job_id": jid})
        job_ids.append(jid)
        print(f"  {label:28s} F = {ev:.4f} +- {sd:.4f}")

    off = {"dynamical_decoupling.enable": False, "resilience_level": 0, "twirling.enable_gates": False}
    run("baseline", isa, off)
    if a.fake:
        run("our DD (idle-aware)", ours, off)
    else:
        run("Runtime DD (XY4)", isa, {**off, "dynamical_decoupling.enable": True,
                                      "dynamical_decoupling.sequence_type": "XY4"})
        run("our DD (idle-aware)", ours, off)
        if not a.no_zne:
            run("Runtime ZNE (level 2)", isa, {**off, "resilience_level": 2})
            run("our DD + Runtime ZNE", ours, {**off, "resilience_level": 2})

    out = Path(a.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"backend": name, "date": time.strftime("%Y-%m-%d %H:%M"), "n": a.n,
                               "shots": a.shots, "wait_ns": a.wait_ns, "max_pulses": a.max_pulses, "strategy": a.strategy, "tau_ns": a.tau,
                               "rows": rows, "job_ids": job_ids}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()