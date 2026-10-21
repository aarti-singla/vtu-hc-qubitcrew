"""Reproduce every number / plot used by the website and the slides.

    python -m ddpipe.experiments            # full run (~3-6 min on a laptop)
    python -m ddpipe.experiments --quick    # smaller, ~1 min
Outputs go to ./results (JSON + CSV + PNG).
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .device import Device
from .pipeline import Compiled, Config, evaluate, standard_configs
from .plots import timeline_figure
from .workloads import build

OUT = Path("results")


def _save(name: str, rows: list[dict]):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(rows, indent=1))
    pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)


def exp_ablation(K: int, quick: bool):
    specs = [("ghz", 5), ("ghzwait", 4), ("qaoa", 4), ("vqe", 4), ("mirror", 5)] + ([] if quick else [("qaoa", 6)])
    rows = []
    for kind, n in specs:
        comp = Compiled(build(kind, n), Device(n))
        for cfg in standard_configs():
            r = evaluate(comp, cfg, K=K)
            d = r.to_row()
            if cfg.dd == "auto":
                ch = r.raw["auto_choice"]
                d["auto_choice"] = f"{ch['strategy']}@{ch['tau_ns']:g}ns"
            rows.append(d)
        print(f"  ablation {comp.wl.name} done")
    _save("ablation", rows)


def exp_scaling(K: int, quick: bool):
    rows = []
    for n in ([3, 4, 5, 6] if quick else [3, 4, 5, 6, 7]):
        comp = Compiled(build("ghz", n), Device(n))
        for cfg in [c for c in standard_configs() if c.name in ("Baseline", "Qiskit DD", "DD uniform", "DD auto (ours)")]:
            rows.append(evaluate(comp, cfg, K=K).to_row())
        print(f"  scaling GHZ-{n} done")
    _save("scaling", rows)


def exp_tau(K: int, quick: bool):
    taus = [150, 300, 600, 1200, 2400] if quick else [100, 150, 200, 300, 450, 600, 900, 1200, 2000, 3000]
    rows = []
    for kind, n in [("ghzwait", 4), ("qaoa", 4)]:
        comp = Compiled(build(kind, n), Device(n))
        base = evaluate(comp, Config("Baseline"), K=K)
        rows.append({**base.to_row(), "family": "-", "tau_ns": None})
        for fam in ("CPMG", "XpXm", "XY4", "XY8"):
            for tau in taus:
                r = evaluate(comp, Config(f"uniform-{fam}", dd="uniform", family=fam, tau_ns=float(tau)), K=K)
                rows.append({**r.to_row(), "family": fam, "tau_ns": tau})
        print(f"  tau sweep {comp.wl.name} done")
    _save("tau_sweep", rows)


def exp_noise(K: int, quick: bool):
    rows = []
    zz = [0, 40, 80, 120] if quick else [0, 20, 40, 60, 80, 100, 130, 160]
    sg = [0, 25, 50] if quick else [0, 10, 20, 30, 40, 50]
    for kind, n in [("ghzwait", 4)]:
        for axis, vals, key in (("ZZ rate [kHz]", zz, "zz_khz"), ("quasi-static sigma [kHz]", sg, "sigma_f_khz")):
            for v in vals:
                dev = Device(n).with_noise(**{key: float(v)})
                comp = Compiled(build(kind, n), dev)
                for cfg in [c for c in standard_configs() if c.name in
                            ("Baseline", "Qiskit DD", "DD uniform", "DD staggered (ours)", "DD auto (ours)")]:
                    rows.append({**evaluate(comp, cfg, K=K).to_row(), "axis": axis, "value": v})
            print(f"  noise sweep {axis} done")
    _save("noise_sweep", rows)


def exp_shots(K: int, quick: bool):
    """ZNE trades bias for variance: emulate finite shots on the energy workloads."""
    rows = []
    for kind, n in [("qaoa", 4), ("vqe", 4)]:
        comp = Compiled(build(kind, n), Device(n))
        for shots in ([2000] if quick else [1000, 4000, 16000]):
            for cfg in standard_configs(shots=shots):
                if cfg.name in ("Baseline", "DD auto (ours)", "ZNE only", "Auto DD + ZNE (ours)",
                                "Auto DD + ZNE + REM (ours)"):
                    rows.append({**evaluate(comp, cfg, K=K).to_row(), "shots": shots})
        print(f"  shots {comp.wl.name} done")
    _save("shots", rows)


def exp_timeline():
    from .pipeline import Compiled
    OUT.mkdir(exist_ok=True)
    dev = Device(4)
    comp = Compiled(build("ghzwait", 4), dev)
    for name, cfg in [("no_dd", Config("b")), ("uniform", Config("u", dd="uniform")),
                      ("staggered", Config("s", dd="staggered"))]:
        prog, _ = comp.program(1.0, cfg)
        fig = timeline_figure(prog, dev.edges, f"GHZ-wait-4 schedule - {name}")
        fig.savefig(OUT / f"timeline_{name}.png", dpi=140)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--K", type=int, default=32, help="quasi-static noise ensemble size")
    a = ap.parse_args()
    t0 = time.time()
    dev = Device(4)
    meta = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version(),
            "K": a.K, "quick": a.quick, "noise_defaults": dev.noise.__dict__}
    for name, fn in [("ablation", exp_ablation), ("scaling", exp_scaling), ("tau", exp_tau),
                     ("noise", exp_noise), ("shots", exp_shots)]:
        print(f"[{time.time() - t0:5.0f}s] {name}")
        fn(a.K, a.quick)
    exp_timeline()
    meta["seconds"] = time.time() - t0
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {OUT.resolve()}")


if __name__ == "__main__":
    main()
