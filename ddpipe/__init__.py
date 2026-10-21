"""ddpipe - context-aware dynamical decoupling + error mitigation pipeline for Qiskit."""
from .device import Device, NoiseParams
from .dd import inject_dd, pulse_train, qiskit_pad_dd
from .idle import idle_report, idle_windows
from .mitigation import extrapolate, fold_cx
from .pipeline import Compiled, Config, Result, evaluate, standard_configs
from .program import (TimedProgram, compile_basis, program_from_circuit,
                      qiskit_circuit_from_program, schedule_circuit)
from .simulate import energy, fidelity, simulate
from .workloads import build, ghz, mirror, qaoa_ring, vqe_tfim

__version__ = "0.1.0"
