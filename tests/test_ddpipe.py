"""Physics + plumbing sanity tests.   pytest -q"""
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from ddpipe import (Compiled, Config, Device, build, evaluate, extrapolate, fidelity, fold_cx,
                    inject_dd, pulse_train, simulate)
from ddpipe.device import NoiseParams
from ddpipe.dd import _fractions
from ddpipe.program import Event, TimedProgram, qiskit_circuit_from_program

QUIET = dict(t1_us=1e6, t2_us=1e6, sigma_f_khz=0.0, p1q=0.0, p2q=0.0, pulse_overrot=0.0, spread=0.0,
             zz_spread=0.0, zz_khz=0.0)


def two_qubit_idle(L=2000.0, **noise):
    dev = Device(2, topology="line").with_noise(**{**QUIET, **noise})
    prog = TimedProgram(2, [Event(0, 35.56, "sx", (q,), 0, "circuit", q) for q in range(2)], L)
    qc = QuantumCircuit(2); qc.sx(0); qc.sx(1)
    return dev, prog, Statevector(qc).data


@pytest.mark.parametrize("kind,n", [("ghz", 4), ("qaoa", 4), ("vqe", 4), ("mirror", 4), ("ghzwait", 4)])
def test_noise_free_simulation_matches_ideal(kind, n):
    dev = Device(n)
    comp = Compiled(build(kind, n), dev)
    prog, _ = comp.program(1.0, Config("b"))
    assert fidelity(simulate(prog, dev, ideal=True), comp.psi) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("family", ["CPMG", "XpXm", "XY4", "XY8"])
@pytest.mark.parametrize("n", [2, 4, 6, 8, 16])
def test_dd_trains_compose_to_identity(family, n):
    """With perfect pulses a DD window must leave the state untouched."""
    dev, prog, psi = two_qubit_idle(L=3000.0)
    from ddpipe.dd import pulse_train as pt
    from ddpipe.program import Event
    ev = list(prog.events)
    train = pt(family, n)
    for j, name in enumerate(train):
        ev.append(Event(3000 * (j + .5) / n - 17.78, 35.56, name, (0,), 0, "dd", 10 ** 7 + j))
    out = TimedProgram(2, ev, 3000.0)
    assert fidelity(simulate(out, dev, ideal=True), psi) == pytest.approx(1.0, abs=1e-9)


def test_dd_refocuses_quasi_static_dephasing():
    dev, prog, psi = two_qubit_idle(sigma_f_khz=30.0)
    f0 = fidelity(simulate(prog, dev, K=32), psi)
    pd_, _ = inject_dd(prog, dev.edges, dev.tp_ns, "uniform", "XY4", 600, 250)
    f1 = fidelity(simulate(pd_, dev, K=32), psi)
    assert f0 < 0.97 and f1 > 0.999


def test_staggering_is_needed_to_refocus_zz():
    """Identical trains commute with ZZ (no help); staggered trains remove it."""
    dev, prog, psi = two_qubit_idle(zz_khz=70.0)
    f = {}
    for s in ("none", "uniform", "staggered"):
        p, _ = inject_dd(prog, dev.edges, dev.tp_ns, s, "XY4", 600, 250, zz_rates=dev.zz_rates())
        f[s] = fidelity(simulate(p, dev, K=32), psi)
    assert f["uniform"] == pytest.approx(f["none"], abs=1e-3)
    assert f["staggered"] > f["uniform"] + 0.1


def test_idle_windows_found():
    from ddpipe import idle_report
    dev = Device(4)
    comp = Compiled(build("ghz", 4), dev)
    prog, _ = comp.program(1.0, Config("b"))
    rep = idle_report(prog, dev.edges)
    assert rep["n_windows"] > 0 and 0 < rep["idle_pct"] < 100


def test_fold_preserves_unitary_and_scales_cx():
    comp = Compiled(build("ghz", 4), Device(4))
    f3 = fold_cx(comp.basis, 3.0)
    n1 = sum(i.operation.name == "cx" for i in comp.basis.data)
    n3 = sum(i.operation.name == "cx" for i in f3.data)
    assert n3 == 3 * n1
    assert abs(np.vdot(Statevector(f3).data, comp.psi)) == pytest.approx(1.0, abs=1e-9)


def test_extrapolators_recover_known_curves():
    x = [1.0, 2.0, 3.0]
    assert extrapolate(x, [0.9, 0.8, 0.7], "linear") == pytest.approx(1.0)
    assert extrapolate(x, [np.exp(-0.1 * s) for s in x], "exponential") == pytest.approx(1.0, abs=1e-6)
    assert extrapolate(x, [1 - 0.1 * s + 0.01 * s * s for s in x], "richardson") == pytest.approx(1.0, abs=1e-6)


def test_export_roundtrip_has_same_ideal_unitary():
    dev = Device(4)
    comp = Compiled(build("ghz", 4), dev)
    prog, _ = comp.program(1.0, Config("s", dd="staggered"))
    qc = qiskit_circuit_from_program(prog)
    assert abs(np.vdot(Statevector(qc).data, comp.psi)) == pytest.approx(1.0, abs=1e-6)


def test_auto_never_worse_than_no_dd_on_model():
    dev = Device(4)
    comp = Compiled(build("qaoa", 4), dev)
    a = evaluate(comp, Config("auto", dd="auto"), K=32).fidelity
    b = evaluate(comp, Config("base"), K=32).fidelity
    assert a >= b - 1e-6


def test_program_uses_backend_dt_not_a_constant():
    """IBM Heron reports dt = 4 ns; a delay of N dt must become N*4 ns in the timed program."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XGate
    from qiskit.transpiler import InstructionProperties, Target
    from ddpipe.program import program_from_circuit, qiskit_circuit_from_program

    tgt = Target(num_qubits=1, dt=4e-9)
    tgt.add_instruction(XGate(), {(0,): InstructionProperties(duration=36e-9)})
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.delay(2500, 0, unit="dt")
    qc.x(0)
    prog = program_from_circuit(qc, tgt)
    assert abs(prog.dt_ns - 4.0) < 1e-9
    assert abs(prog.t_end - (36 + 2500 * 4 + 36)) < 1e-6          # 10 us wait, not 0.55 us
    back = qiskit_circuit_from_program(prog, granularity=16)
    delays = [i.operation.duration for i in back.data if i.operation.name == "delay"]
    assert abs(sum(delays) * 4 - 10000) <= 16 * 4                  # round-trips to ~10 us
