"""Streamlit dashboard.   streamlit run app/app.py"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddpipe import (Compiled, Config, Device, build, evaluate, idle_report,  # noqa: E402
                    qiskit_circuit_from_program, standard_configs)
from ddpipe.plots import timeline_figure  # noqa: E402

RES = ROOT / "results"
st.set_page_config(page_title="Idle-aware DD + Error Mitigation", page_icon="⚛️", layout="wide")

ORDER = ["Baseline", "Qiskit DD", "DD uniform", "DD staggered (ours)", "DD auto (ours)", "ZNE only",
         "Auto DD + ZNE (ours)", "Auto DD + ZNE + REM (ours)"]


@st.cache_data
def load(name):
    p = RES / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource
def compiled(kind, n, topology, zz, sigma, wait):
    dev = Device(n, topology=topology).with_noise(zz_khz=float(zz), sigma_f_khz=float(sigma))
    kw = {"wait_ns": float(wait)} if kind == "ghzwait" else {}
    return Compiled(build(kind, n, **kw), dev)


st.title("⚛️ Idle-aware Dynamical Decoupling + Error Mitigation for NISQ Workflows")
st.caption("Qiskit Fall Fest 2026 · VTU × IBM Quantum · Integrated Dynamical Decoupling and Error Mitigation Pipeline")

tabs = st.tabs(["Home", "Schedule explorer", "Live benchmark", "Results", "Hardware", "Method", "Team & links"])

# ------------------------------------------------------------------ Home
with tabs[0]:
    st.subheader("The problem")
    st.markdown(
        "In multi-qubit NISQ circuits **idle qubits decohere** while their neighbours run gates, and **gates accumulate "
        "coherent errors**. We built an automated Qiskit pipeline that (1) analyses the scheduled circuit to find every "
        "idle window, (2) injects crosstalk-aware dynamical-decoupling (DD) pulse trains, (3) selects the best DD "
        "configuration with a calibrated noise model in the loop, and (4) layers zero-noise extrapolation (ZNE) and "
        "readout mitigation on top.")
    ab = load("ablation")
    if ab is not None:
        ab = ab.assign(err=(1 - ab.fidelity).abs())
        base = ab[ab.config == "Baseline"].set_index("workload").err
        best = ab[ab.config == "DD auto (ours)"].set_index("workload").err
        zne = ab[ab.config == "Auto DD + ZNE (ours)"].set_index("workload").err
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean infidelity, no mitigation", f"{base.mean():.3f}")
        c2.metric("with auto-DD", f"{best.mean():.3f}", f"{(best.mean() - base.mean()) / base.mean() * 100:.0f}%")
        c3.metric("with auto-DD + ZNE", f"{zne.mean():.3f}", f"{(zne.mean() - base.mean()) / base.mean() * 100:.0f}%")
        st.caption("Simulated noisy device (see Method). Averages over the benchmark workloads in the Results tab. "
                   "ZNE values are extrapolated and can overshoot, so read the per-workload table too.")
    st.info("Everything on this page is **simulated** unless marked otherwise in the Hardware tab.")

# ------------------------------------------------------- Schedule explorer
with tabs[1]:
    c = st.columns(5)
    kind = c[0].selectbox("Circuit", ["ghzwait", "ghz", "qaoa", "vqe", "mirror"], format_func=lambda k: {
        "ghzwait": "GHZ + mid-circuit wait", "ghz": "GHZ", "qaoa": "QAOA ring", "vqe": "VQE (TFIM)", "mirror": "Mirror"}[k])
    n = c[1].slider("Qubits", 3, 6, 4, key="n1")
    strategy = c[2].selectbox("DD strategy", ["none", "uniform", "staggered", "auto"])
    family = c[3].selectbox("Sequence", ["XY4", "XpXm", "CPMG", "XY8"])
    tau = c[4].slider("Pulse spacing τ [ns]", 100, 2400, 600, step=50)
    comp = compiled(kind, n, "ring", 70, 25, 1500)
    cfg = Config("view", dd=strategy, family=family, tau_ns=float(tau))
    prog, _ = comp.program(1.0, cfg)
    rep = idle_report(prog, comp.dev.edges)
    st.pyplot(timeline_figure(prog, comp.dev.edges, f"{comp.wl.name} - DD: {strategy}"), clear_figure=True)
    m = st.columns(5)
    m[0].metric("Duration", f"{prog.t_end:.0f} ns")
    m[1].metric("Idle share", f"{rep['idle_pct']:.0f}%")
    m[2].metric("Idle windows", rep["n_windows"])
    m[3].metric("Crosstalk-idle windows", rep["n_crosstalk_idle"])
    m[4].metric("DD pulses", prog.n_dd_pulses)
    st.dataframe(pd.DataFrame(rep["per_qubit"]).round(1), width="stretch", hide_index=True)
    with st.expander("Export this schedule as a Qiskit circuit (for hardware)"):
        qc = qiskit_circuit_from_program(prog)
        st.code(str(qc.draw("text", fold=140)), language=None)

# ------------------------------------------------------------ Live benchmark
with tabs[2]:
    st.write("Runs the full pipeline live on a small circuit (exact density-matrix simulation, a few seconds).")
    c = st.columns(6)
    kind = c[0].selectbox("Circuit ", ["ghzwait", "ghz", "qaoa", "vqe", "mirror"], key="k2")
    n = c[1].slider("Qubits ", 3, 5, 4, key="n2")
    zz = c[2].slider("ZZ crosstalk [kHz]", 0, 150, 70)
    sg = c[3].slider("Slow-drift σ [kHz]", 0, 60, 25)
    wait = c[4].slider("Wait [ns] (GHZ+wait)", 500, 3000, 1500, step=100)
    use_zne = c[5].checkbox("Include ZNE configs", True)
    if st.button("Run benchmark", type="primary"):
        comp = compiled(kind, n, "ring", zz, sg, wait)
        cfgs = [x for x in standard_configs() if use_zne or not x.zne]
        rows, bar = [], st.progress(0.0)
        for i, cf in enumerate(cfgs):
            r = evaluate(comp, cf, K=24)
            rows.append({"config": cf.name, "|1-F|": abs(1 - r.fidelity), "fidelity": r.fidelity,
                         "energy error": r.energy_error, "DD pulses": r.n_dd_pulses})
            bar.progress((i + 1) / len(cfgs))
        df = pd.DataFrame(rows)
        fig = px.bar(df, x="config", y="|1-F|", color="config", title="State-fidelity error (lower is better)")
        fig.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(df.round(4), width="stretch", hide_index=True)
        st.caption("ZNE rows are extrapolated estimates and may fall slightly outside [0, 1].")

# ------------------------------------------------------------------ Results
with tabs[3]:
    ab, sc, ta, nz, sh = (load(x) for x in ("ablation", "scaling", "tau_sweep", "noise_sweep", "shots"))
    if ab is None:
        st.warning("Run `python -m ddpipe.experiments` first to generate results/.")
    else:
        st.subheader("1 · Ablation on five workloads")
        d = ab.assign(err=(1 - ab.fidelity).abs())
        fig = px.bar(d, x="workload", y="err", color="config", barmode="group",
                     category_orders={"config": ORDER}, labels={"err": "|1 − F|"})
        st.plotly_chart(fig, width="stretch")
        piv = d.pivot(index="config", columns="workload", values="fidelity").reindex(ORDER).round(3)
        st.dataframe(piv, width="stretch")
        ee = d.dropna(subset=["energy_error"]).pivot(index="config", columns="workload", values="energy_error")
        if not ee.empty:
            st.markdown("**Energy error |E − E_ideal|** (QAOA / VQE, exact expectation values)")
            st.dataframe(ee.reindex(ORDER).round(3), width="stretch")
        if "auto_choice" in d:
            st.caption("Auto-DD choices: " + ", ".join(
                f"{w}: {c}" for w, c in d[d.config == "DD auto (ours)"][["workload", "auto_choice"]].values))
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("2 · Scaling with GHZ size")
            st.plotly_chart(px.line(sc, x="n", y="fidelity", color="config", markers=True), width="stretch")
        with c2:
            st.subheader("3 · Pulse spacing & sequence sweep")
            wl = st.selectbox("Workload", ta.workload.unique())
            t = ta[(ta.workload == wl) & (ta.family != "-")]
            fig = px.line(t, x="tau_ns", y="fidelity", color="family", markers=True, log_x=True)
            fig.add_hline(y=float(ta[(ta.workload == wl) & (ta.family == "-")].fidelity.iloc[0]),
                          line_dash="dash", annotation_text="no DD")
            st.plotly_chart(fig, width="stretch")
        st.subheader("4 · Robustness to crosstalk and slow drift (GHZ + wait)")
        fig = px.line(nz, x="value", y="fidelity", color="config", facet_col="axis", markers=True)
        fig.update_xaxes(matches=None)
        st.plotly_chart(fig, width="stretch")
        if sh is not None:
            st.subheader("5 · ZNE trades bias for variance (finite-shot emulation)")
            s = sh.assign(label=sh.workload + " · " + sh.shots.astype(str) + " shots")
            fig = px.bar(s, x="config", y="energy_error", color="config", facet_col="label", facet_col_wrap=3,
                         category_orders={"config": ORDER}, labels={"energy_error": "RMSE of energy"})
            fig.update_xaxes(showticklabels=False)
            st.plotly_chart(fig, width="stretch")
            st.caption("RMSE includes bias and shot-noise variance. Extrapolation amplifies variance: gains shrink "
                       "(or reverse) at low shot counts.")

# ------------------------------------------------------------------ Hardware
with tabs[4]:
    hw = RES / "hardware.json"
    if hw.exists():
        data = json.loads(hw.read_text())
        st.success(f"Backend: {data.get('backend')} · date: {data.get('date')}")
        st.dataframe(pd.DataFrame(data["rows"]), width="stretch", hide_index=True)
        st.caption("Job IDs: " + ", ".join(data.get("job_ids", [])))
    else:
        st.info("No hardware run recorded yet. Run `python scripts/run_hardware.py` with your IBM Quantum credentials; "
                "it writes `results/hardware.json`, which this tab then displays (with job IDs).")
    # Sweep over idle-wait length: every results/hw_w*us.json, hardware_wait10us.json and hardware_no_wait.json
    sweep_rows = []
    for f in sorted(list(RES.glob("hw_w*us.json")) + list(RES.glob("hardware_wait*.json")) +
                    list(RES.glob("hardware_no_wait.json"))):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for r in d.get("rows", []):
            if r["config"] in ("baseline", "Runtime DD (XY4)", "our DD (idle-aware)"):
                sweep_rows.append({"wait [us]": d.get("wait_ns", 10000 if "wait10" in f.name else 0) / 1000,
                                   "config": r["config"], "fidelity": r["fidelity"], "stderr": r["stderr"],
                                   "backend": d.get("backend")})
    if sweep_rows:
        sw = pd.DataFrame(sweep_rows).drop_duplicates(["wait [us]", "config"], keep="last")
        st.markdown("#### Hardware sweep: GHZ-4 fidelity vs idle wait")
        fig = px.line(sw.sort_values("wait [us]"), x="wait [us]", y="fidelity", color="config", markers=True,
                      error_y="stderr")
        st.plotly_chart(fig, width="stretch")
        st.caption("Real-device runs (error bars are shot noise only; each point is one run, so drift between "
                   "jobs is not included). At 0 us the circuit has no idle window long enough for DD, so our DD "
                   "inserted no pulses there.")
        st.dataframe(sw.pivot(index="wait [us]", columns="config", values="fidelity").round(3), width="stretch")
    st.markdown("**Configurations compared on hardware:** no suppression · Runtime DD · our schedule-aware DD "
                "(exported circuit, Runtime DD off) · Runtime ZNE · our DD + Runtime ZNE/TREX.")

# -------------------------------------------------------------------- Method
with tabs[5]:
    st.markdown(
        """
**Pipeline**

`circuit → transpile (device basis) → [fold CX for ZNE] → ALAP schedule → idle-window analysis → DD injection → execution/simulation → extrapolate → metrics`

**Idle analyzer.** For every qubit we find the gaps between its operations and label each *idle-idle* or *crosstalk-idle*
(a coupled neighbour is running a CX) - the distinction used in DD-insertion literature.

**Crosstalk-aware (staggered) DD.** X/Y pulses commute with ZZ, so identical trains on neighbouring idle qubits leave ZZ
crosstalk intact: the ZZ phase is `J∫ s_a(t) s_b(t) dt` where `s` is each qubit's ±1 toggling frame. We search small
candidate sets (CPMG/Uhrig, several pulse counts) per window and keep the one minimising pulse cost plus the squared ZZ
frame-integral against already-placed neighbours, splitting at the neighbour's non-commuting gates.

**Auto-DD.** Candidates (none, uniform, staggered × spacing τ) are scored with the calibrated device model on an
independent noise sample, and the best is kept - `none` is a candidate, so the pass can't hurt on the model.

**Mitigation.** CX unitary folding for ZNE (scales 1,2,3 + linear extrapolation by default; Richardson / exponential
available) and calibrated readout-error mitigation for Pauli-sum observables.

**Noise model (simulation).** Quasi-static detuning (σ), always-on ZZ, T1/Tφ, depolarising gate errors, DD-pulse
amplitude error, readout flips. Pure Markovian T1/T2 would give DD *no* benefit, so slow and coherent errors are modelled
explicitly. Expectation values are exact (infinite shots) except where the Results tab says otherwise.

**Honest limits.** Gains come from a *model* of the device; real hardware will differ. ZNE is sensitive to the
extrapolation choice and amplifies shot noise. DD helps most on long idle windows and can be neutral or slightly negative
for short ones - which is why the pass skips tiny windows and the auto mode includes "no DD".
        """)

# -------------------------------------------------------------- Team & links
with tabs[6]:
    st.markdown("""
* **Track:** Integrated Dynamical Decoupling and Error Mitigation Pipeline for NISQ Workflows
* **GitHub:** _add your team repository link (vtu-qiskit-fall-fest-2026 org)_
* **Demo video:** _add your YouTube link_
* **Team:** _names · roles_
""")