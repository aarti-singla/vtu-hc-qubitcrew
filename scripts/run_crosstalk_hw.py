"""Crosstalk test on a real IBM backend: a row of qubits in |+>, idle for a chosen wait, measured as <X>.

Why this circuit
----------------
ZZ crosstalk between neighbouring qubits dephases a product state like |+>|+> but cannot dephase a GHZ
state (a ZZ phase is identical on both GHZ branches).  Identical DD trains on neighbouring qubits also
leave the ZZ term intact (the two sign flips cancel in the product), while *staggered* trains refocus it.
So this is the experiment where schedule-aware staggering should separate from plain uniform DD.

Metric: mean <X_i> over the n qubits (ideal = 1), measured through the Estimator in one basis.

Configurations (each run `--repeats` times, interleaved so slow drift is shared):
  baseline            no suppression
  IBM Runtime DD      Runtime's built-in XY4
  our DD uniform      ddpipe injector, identical trains on every qubit
  our DD staggered    ddpipe injector, trains staggered across coupled neighbours

Usage
-----
  python scripts/run_crosstalk_hw.py --fake --n 4 --wait-ns 20000        # offline plumbing test (free)
  python scripts/run_crosstalk_hw.py --n 4 --shots 2000 --wait-ns 20000 --repeats 2
  python scripts/run_crosstalk_hw.py --backend ibm_marrakesh --layout 0,1,2,3 ...

NEVER hard-code or commit your API token.  Output: results/crosstalk_hw.json
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
from qiskit.quantum_info import SparsePauliOp  # noqa: E402
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # noqa: E402

from ddpipe import inject_dd, qiskit_circuit_from_program  # noqa: E402
from ddpipe.idle import idle_report  # noqa: E402
from ddpipe.program import program_from_circuit, schedule_circuit  # noqa: E402


def plus_circuit(n, wait_dt):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.barrier()
    for q in range(n):
        qc.delay(int(wait_dt), q)
    qc.barrier()
    return qc


def mean_x(n):
    return SparsePauliOp.from_sparse_list([("X", [i], 1.0 / n) for i in range(n)], num_qubits=n)


def find_path(backend, n):
    """Connected path of n physical qubits with the lowest total readout error."""
    nb = {}
    for a, b in backend.coupling_map.get_edges():
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)

    def ro(q):
        try:
            return backend.target["measure"][(q,)].error or 0.0
        except Exception:
            return 0.0

    best, best_cost = None, 1e9

    def dfs(path):
        nonlocal best, best_cost
        if len(path) == n:
            cost = sum(ro(q) for q in path)
            if cost < best_cost:
                best, best_cost = list(path), cost
            return
        for v in sorted(nb.get(path[-1], ())):
            if v not in path:
                path.append(v)
                dfs(path)
                path.pop()

    for s in sorted(nb):
        dfs([s])
    return best


def qiskit_pad(isa, backend, qubits, granularity):
    """Qiskit's own PadDynamicalDecoupling (X-X, uniform spacing) on the transpiled circuit."""
    from qiskit.circuit.library import XGate
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
    pm = PassManager([ALAPScheduleAnalysis(target=backend.target),
                      PadDynamicalDecoupling(target=backend.target, dd_sequence=[XGate(), XGate()],
                                             qubits=list(qubits), pulse_alignment=granularity)])
    return pm.run(isa)


def build(backend, n, layout, strategy, tau_ns, granularity, wait_ns, max_pulses=16):
    dt = backend.target.dt
    wait_dt = int(round(wait_ns * 1e-9 / dt / granularity)) * granularity
    logical = plus_circuit(n, wait_dt)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend, seed_transpiler=7,
                                      initial_layout=layout)
    isa = pm.run(logical)
    obs = mean_x(n).apply_layout(isa.layout)
    target = backend.target
    prog = program_from_circuit(schedule_circuit(isa, target), target, "hw")
    edges = sorted({tuple(sorted(e)) for e in backend.coupling_map.get_edges()})
    used = sorted({q for e in prog.events for q in e.qubits})
    tp = float(np.mean([target["x"][(q,)].duration for q in used])) * 1e9
    dd_prog, info = inject_dd(prog, edges, tp, strategy, "XpXm", tau_ns, 250.0, max_pulses=max_pulses)
    ours = qiskit_circuit_from_program(dd_prog, granularity=granularity)
    return isa, ours, obs, idle_report(prog, edges), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--layout", default=None, help="comma-separated physical qubits forming a connected path")
    ap.add_argument("--fake", action="store_true", help="offline dry run on a fake backend (Aer)")
    ap.add_argument("--tau", type=float, default=600.0)
    ap.add_argument("--granularity", type=int, default=0, help="delay granularity in dt (0 = read from the backend)")
    ap.add_argument("--wait-ns", type=float, default=20000.0)
    ap.add_argument("--max-pulses", type=int, default=16, help="cap on DD pulses per idle window")
    ap.add_argument("--extended", action="store_true",
                    help="also run Qiskit's PadDynamicalDecoupling and readout mitigation (TREX) variants")
    ap.add_argument("--repeats", type=int, default=1, help="interleaved repeats of every configuration")
    ap.add_argument("--out", default="results/crosstalk_hw.json")
    a = ap.parse_args()

    if a.fake:
        from qiskit.primitives import BackendEstimatorV2
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        backend = FakeMarrakesh()
        sim = AerSimulator.from_backend(backend)
        name = "FakeMarrakesh (Aer, dry run)"
    else:
        from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService
        service = QiskitRuntimeService()
        backend = (service.backend(a.backend) if a.backend else
                   service.least_busy(operational=True, simulator=False, min_num_qubits=max(a.n, 5)))
        name = backend.name

    layout = [int(q) for q in a.layout.split(",")] if a.layout else find_path(backend, a.n)
    gran = a.granularity or max(16, int(getattr(backend.target, "granularity", 0) or 0))
    print(f"backend={name}  layout={layout}  wait={a.wait_ns:.0f} ns  dt={backend.target.dt * 1e9:.3f} ns  granularity={gran} dt")
    built = {s: build(backend, a.n, layout, s, a.tau, gran, a.wait_ns, a.max_pulses) for s in ("uniform", "staggered")}
    isa, _, obs, rep, _ = built["uniform"]
    for s, (_, _, _, _, info) in built.items():
        print(f"  {s:10s} DD pulses={info.n_pulses}")
    print(f"  idle windows={rep['n_windows']}  idle={rep['idle_pct']:.0f}%")

    off = {"dynamical_decoupling.enable": False, "resilience_level": 0, "twirling.enable_gates": False}
    jobs = [("baseline", isa, off),
            ("IBM Runtime DD", isa, {**off, "dynamical_decoupling.enable": True,
                                      "dynamical_decoupling.sequence_type": "XY4"}),
            ("our DD uniform", built["uniform"][1], off),
            ("our DD staggered", built["staggered"][1], off)]
    if a.extended:
        padded = qiskit_pad(isa, backend, layout, gran)
        n_pad = sum(1 for i in padded.data if i.operation.name == "x") - sum(1 for i in isa.data if i.operation.name == "x")
        print(f"  Qiskit PadDD inserted {n_pad} X pulses")
        rem = {**off, "resilience_level": 1}
        jobs += [("Qiskit PadDD (X-X)", padded, off),
                 ("baseline + readout mitigation", isa, rem),
                 ("our DD staggered + readout mitigation", built["staggered"][1], rem)]
    if a.fake:
        jobs = [j for j in jobs if j[0] != "IBM Runtime DD" and "readout mitigation" not in j[0]]

    rows = []

    def run(label, circuit, options, rep_i):
        if a.fake:
            est = BackendEstimatorV2(backend=sim, options={"default_precision": 1 / np.sqrt(a.shots)})
            res = est.run([(circuit, obs)]).result()[0]
            jid = "local"
        else:
            est = EstimatorV2(mode=backend)
            for k, v in options.items():
                o = est.options
                *path, last = k.split(".")
                for p in path:
                    o = getattr(o, p)
                setattr(o, last, v)
            est.options.default_shots = a.shots
            job = est.run([(circuit, obs)])
            jid = job.job_id()
            print(f"  submitted {label} #{rep_i + 1}: {jid}")
            res = job.result()[0]
        ev, sd = float(np.real(res.data.evs)), float(np.real(res.data.stds))
        rows.append({"config": label, "repeat": rep_i + 1, "x_mean": ev, "stderr": sd, "job_id": jid})
        print(f"  {label:18s} #{rep_i + 1}  <X> = {ev:.4f} +- {sd:.4f}")

    for r in range(a.repeats):
        order = jobs if r % 2 == 0 else list(reversed(jobs))  # alternate order to cancel drift
        for label, circ, opts in order:
            run(label, circ, opts, r)

    print("\nsummary (mean over repeats)")
    summary = []
    for label, _, _ in jobs:
        v = [x["x_mean"] for x in rows if x["config"] == label]
        e = [x["stderr"] for x in rows if x["config"] == label]
        m = float(np.mean(v))
        se = float(np.sqrt(np.sum(np.square(e))) / len(e))
        spread = float(np.std(v, ddof=1)) if len(v) > 1 else None
        summary.append({"config": label, "x_mean": m, "stderr": se, "repeat_spread": spread, "runs": len(v)})
        print(f"  {label:18s} <X> = {m:.4f} +- {se:.4f}" + (f"   (spread between repeats {spread:.4f})" if spread else ""))

    out = Path(a.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"backend": name, "date": time.strftime("%Y-%m-%d %H:%M"), "n": a.n, "layout": layout,
                               "shots": a.shots, "wait_ns": a.wait_ns, "tau_ns": a.tau, "max_pulses": a.max_pulses, "extended": a.extended, "repeats": a.repeats,
                               "pulses": {s: built[s][4].n_pulses for s in built},
                               "rows": rows, "summary": summary}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()