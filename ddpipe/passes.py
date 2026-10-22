"""Drop-in Qiskit transpiler pass: idle-window analysis + crosstalk-aware dynamical decoupling.

    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay
    from ddpipe import IdleAwareDD

    pm = PassManager([ALAPScheduleAnalysis(target=target), PadDelay(target=target),
                      IdleAwareDD(target=target, strategy="staggered")])
    protected = pm.run(isa_circuit)

The pass reads the explicit ``Delay`` instructions that ``PadDelay`` leaves in a scheduled circuit,
labels every idle window (idle-idle or crosstalk-idle, from the target's coupling map), places DD
pulses inside the windows that are long enough to pay for them, and returns the same circuit with the
pulses inserted.  Trailing measurements (``measure_all`` or per-qubit) are kept; a measurement in the
middle of the circuit is not supported and raises ``TranspilerError``.
"""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import Target
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from .dd import inject_dd
from .program import program_from_circuit, qiskit_circuit_from_program

_TAIL = {"measure", "barrier"}


def _edges(target: Target) -> tuple[tuple[int, int], ...]:
    cm = target.build_coupling_map()
    if cm is None:
        return ()
    return tuple(sorted({tuple(sorted(e)) for e in cm.get_edges()}))


def _x_pulse(target: Target) -> tuple[float, float]:
    """(duration in ns, error) of the X gate on qubit 0; fall back to typical numbers."""
    try:
        p = target["x"][(0,)]
        return float(p.duration) * 1e9, float(p.error or 3e-4)
    except Exception:  # noqa: BLE001
        return 35.6, 3e-4


class IdleAwareDD(TransformationPass):
    """Insert crosstalk-aware dynamical decoupling into the idle windows of a scheduled circuit."""

    def __init__(self, target: Target, strategy: str = "staggered", family: str = "XY4", tau_ns: float = 600.0,
                 min_window_ns: float = 250.0, zz_rates: dict | None = None, max_pulses: int = 16):
        super().__init__()
        if strategy not in ("uniform", "staggered"):
            raise ValueError("strategy must be 'uniform' or 'staggered'")
        self.target, self.strategy, self.family = target, strategy, family
        self.tau_ns, self.min_window_ns, self.zz_rates, self.max_pulses = tau_ns, min_window_ns, zz_rates, max_pulses

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        circ = dag_to_circuit(dag)
        data = list(circ.data)
        cut = len(data)
        while cut > 0 and data[cut - 1].operation.name in _TAIL:
            cut -= 1
        if any(i.operation.name == "measure" for i in data[:cut]):
            raise TranspilerError("IdleAwareDD does not support mid-circuit measurements; "
                                  "apply it to the circuit before the final measurements.")
        body = circ.copy_empty_like()
        for inst in data[:cut]:
            body.append(inst.operation, inst.qubits, inst.clbits)
        prog = program_from_circuit(body, self.target, "IdleAwareDD")
        tp_ns, p1q = _x_pulse(self.target)
        out, info = inject_dd(prog, _edges(self.target), tp_ns, self.strategy, self.family, self.tau_ns,
                              self.min_window_ns, zz_rates=self.zz_rates, pulse_cost=p1q, max_pulses=self.max_pulses)
        granularity = int(getattr(self.target, "granularity", 1) or 1)
        new = qiskit_circuit_from_program(out, granularity=granularity)
        final = circ.copy_empty_like()
        final.compose(new, qubits=list(range(circ.num_qubits)), inplace=True)
        for inst in data[cut:]:
            final.append(inst.operation, inst.qubits, inst.clbits)
        if "node_start_time" in self.property_set:      # node objects change, so old start times are stale
            del self.property_set["node_start_time"]
        self.property_set["idle_aware_dd"] = info
        return circuit_to_dag(final)