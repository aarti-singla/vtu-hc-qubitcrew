"""Spectator test on a real IBM backend: a real circuit with *natural* idle windows.

Six connected qubits in a row, all prepared in |+>.  The two middle qubits run a train of CZ gates
(an even number, so ideally the identity) while the four outer qubits sit idle next to a busy neighbour.
Those are exactly the "crosstalk-idle" windows the idle analyzer classifies.  No artificial delay is
added: the idle time comes from the gate train.

Metric (ideal = 1 for both):
  spectator <X>   mean <X> over the four idle qubits  (what DD should protect)
  active    <X>   mean <X> over the two busy qubits   (a check that DD does not hurt the gates)

Configurations: baseline, IBM Runtime DD, Qiskit PadDynamicalDecoupling (X-X), our DD uniform, our DD staggered.

Usage
-----
  python scripts/run_spectator_hw.py --fake                         # offline plumbing test (free)
  python scripts/run_spectator_hw.py --pairs 150 --shots 2000 --repeats 2
  python scripts/run_spectator_hw.py --backend ibm_fez --pairs 300

NEVER hard-code or commit your API token.  Output: results/spectator_hw.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import SparsePauliOp  # noqa: E402
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # noqa: E402

from ddpipe import inject_dd, qiskit_circuit_from_program  # noqa: E402
from ddpipe.idle import idle_report  # noqa: E402
from ddpipe.program import program_from_circuit, schedule_circuit  # noqa: E402
from run_crosstalk_hw import find_path, qiskit_pad  # noqa: E402

N = 6
ACTIVE = (2, 3)
SPECTATORS = (0, 1, 4, 5)


def spectator_circuit(pairs):
    qc = QuantumCircuit(N)
    qc.h(range(N))
    qc.barrier()
    for _ in range(pairs):          # two CZ per pair: identity overall
        qc.cz(*ACTIVE)
        qc.barrier(list(ACTIVE))
        qc.cz(*ACTIVE)
        qc.barrier(list(ACTIVE))
    qc.barrier()
    return qc


def mean_x(idx):
    return SparsePauliOp.from_sparse_list([("X", [i], 1.0 / len(idx)) for i in idx], num_qubits=N)


def build(backend, layout, strategy, tau_ns, granularity, pairs, max_pulses, xtalk_tau_scale=1.0):
    pm = generate_preset_pass_manager(optimization_level=0, backend=backend, seed_transpiler=7,
                                      initial_layout=layout)
    isa = pm.run(spectator_circuit(pairs))
    target = backend.target
    prog = program_from_circuit(schedule_circuit(isa, target), target, "hw")
    edges = sorted({tuple(sorted(e)) for e in backend.coupling_map.get_edges()})
    used = sorted({q for e in prog.events for q in e.qubits})
    tp = float(np.mean([target["x"][(q,)].duration for q in used])) * 1e9
    dd_prog, info = inject_dd(prog, edges, tp, strategy, "XpXm", tau_ns, 250.0, max_pulses=max_pulses,
                              xtalk_tau_scale=xtalk_tau_scale)
    ours = qiskit_circuit_from_program(dd_prog, granularity=granularity)
    return isa, ours, idle_report(prog, edges), info, prog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=150, help="pairs of CZ gates on the active qubits (idle time ~ 140 ns x pairs)")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--layout", default=None, help="6 comma-separated physical qubits forming a connected path")
    ap.add_argument("--fake", action="store_true", help="offline dry run on a fake backend (Aer)")
    ap.add_argument("--tau", type=float, default=600.0)
    ap.add_argument("--granularity", type=int, default=0)
    ap.add_argument("--max-pulses", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--xtalk-tau-scale", type=float, default=1.0,
                    help="scale tau in crosstalk-idle windows only (our DD configs); 1.0 = off")
    ap.add_argument("--configs", default=None,
                    help="comma-separated subset of configs to run, e.g. 'baseline,our DD uniform,our DD staggered'")
    ap.add_argument("--out", default="results/spectator_hw.json")
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
                   service.least_busy(operational=True, simulator=False, min_num_qubits=N + 1))
        name = backend.name

    layout = [int(q) for q in a.layout.split(",")] if a.layout else find_path(backend, N)
    gran = a.granularity or max(16, int(getattr(backend.target, "granularity", 0) or 0))
    print(f"backend={name}  layout={layout}  pairs={a.pairs}  dt={backend.target.dt * 1e9:.3f} ns  granularity={gran} dt")
    built = {s: build(backend, layout, s, a.tau, gran, a.pairs, a.max_pulses, a.xtalk_tau_scale) for s in ("uniform", "staggered")}
    isa, _, rep, _, prog = built["uniform"]
    for s, b in built.items():
        print(f"  {s:10s} DD pulses={b[3].n_pulses}")
    kinds = {}
    from ddpipe.idle import idle_windows
    edges = sorted({tuple(sorted(e)) for e in backend.coupling_map.get_edges()})
    for w in idle_windows(prog, edges):
        kinds[w.kind] = kinds.get(w.kind, 0) + 1
    print(f"  circuit length ~ {prog.t_end / 1000:.2f} us   idle={rep['idle_pct']:.0f}%   windows={rep['n_windows']}  kinds={kinds}")

    # physical-qubit observables (the transpiled circuit keeps the layout)
    obs_s = mean_x(SPECTATORS).apply_layout(isa.layout)
    obs_a = mean_x(ACTIVE).apply_layout(isa.layout)
    padded = qiskit_pad(isa, backend, layout, gran)

    off = {"dynamical_decoupling.enable": False, "resilience_level": 0, "twirling.enable_gates": False}
    jobs = [("baseline", isa, off),
            ("IBM Runtime DD", isa, {**off, "dynamical_decoupling.enable": True,
                                      "dynamical_decoupling.sequence_type": "XY4"}),
            ("Qiskit PadDD (X-X)", padded, off),
            ("our DD uniform", built["uniform"][1], off),
            ("our DD staggered", built["staggered"][1], off)]
    if a.fake:
        jobs = [j for j in jobs if j[0] != "IBM Runtime DD"]
    if a.configs:
        keep = {c.strip() for c in a.configs.split(",")}
        jobs = [j for j in jobs if j[0] in keep]

    rows = []

    def run(label, circuit, options, rep_i):
        if a.fake:
            est = BackendEstimatorV2(backend=sim, options={"default_precision": 1 / np.sqrt(a.shots)})
            res = est.run([(circuit, [obs_s, obs_a])]).result()[0]
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
            job = est.run([(circuit, [obs_s, obs_a])])
            jid = job.job_id()
            print(f"  submitted {label} #{rep_i + 1}: {jid}")
            res = job.result()[0]
        ev = np.real(np.asarray(res.data.evs)).ravel()
        sd = np.real(np.asarray(res.data.stds)).ravel()
        rows.append({"config": label, "repeat": rep_i + 1, "spectator_x": float(ev[0]), "spectator_stderr": float(sd[0]),
                     "active_x": float(ev[1]), "active_stderr": float(sd[1]), "job_id": jid})
        print(f"  {label:20s} #{rep_i + 1}  spectator <X> = {ev[0]:.4f} +- {sd[0]:.4f}   active <X> = {ev[1]:.4f} +- {sd[1]:.4f}")

    for r in range(a.repeats):
        for label, circ, opts in (jobs if r % 2 == 0 else list(reversed(jobs))):
            run(label, circ, opts, r)

    print("\nsummary (mean over repeats)")
    summary = []
    for label, _, _ in jobs:
        rr = [x for x in rows if x["config"] == label]
        row = {"config": label, "runs": len(rr)}
        for key in ("spectator", "active"):
            v = [x[f"{key}_x"] for x in rr]
            e = [x[f"{key}_stderr"] for x in rr]
            row[f"{key}_x"] = float(np.mean(v))
            row[f"{key}_stderr"] = float(np.sqrt(np.sum(np.square(e))) / len(e))
        summary.append(row)
        print(f"  {label:20s} spectator = {row['spectator_x']:.4f} +- {row['spectator_stderr']:.4f}   "
              f"active = {row['active_x']:.4f} +- {row['active_stderr']:.4f}")

    out = Path(a.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"backend": name, "date": time.strftime("%Y-%m-%d %H:%M"), "layout": layout,
                               "pairs": a.pairs, "circuit_us": prog.t_end / 1000, "shots": a.shots, "tau_ns": a.tau,
                               "max_pulses": a.max_pulses, "xtalk_tau_scale": a.xtalk_tau_scale, "repeats": a.repeats, "window_kinds": kinds,
                               "pulses": {s: built[s][3].n_pulses for s in built},
                               "rows": rows, "summary": summary}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()