"""Export the series plotted in the manuscript figures to whitespace-
separated .dat files (and constants to a .tex file) for the pgfplots/TikZ
figures of the TDSC manuscript (tdsc/paper/figs/).

Every number comes from tifs/results/*.csv; this script only selects and
aggregates, exactly as make_figs.py does for the matplotlib versions.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import RES
from exp_bandwidth import n_opt_mono
import pcsp2 as P
from sigs import SIG_BYTES

OUT = Path(__file__).resolve().parents[2] / "tdsc" / "paper" / "figdata"
OUT.mkdir(parents=True, exist_ok=True)
CONSTS: list[str] = []


def dat(name, df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df.to_csv(OUT / f"{name}.dat", sep=" ", index=False, na_rep="nan",
              float_format="%.6g")


def const(name, v, fmt="{:.6g}"):
    CONSTS.append(f"\\newcommand{{\\{name}}}{{{fmt.format(v)}}}")


def shift_traces():
    """Cumulative and trailing-window miscoverage (exp_risk.traces)."""
    for sc, tag in [("cic_zd", "cic"), ("nsl_shift", "nsl")]:
        tr = pd.read_csv(RES / f"e3d_trace_{sc}.csv")
        dat(f"shift_{tag}", tr.rename(columns={
            "cum_static": "cs", "cum_aci0.01": "ca", "cum_aci0.05": "cb",
            "win_static": "ws", "win_aci0.01": "wa", "win_aci0.05": "wb"}))


def q_sweep():
    e = pd.read_csv(RES / "e3_shift.csv")
    for sc, tag in [("cic_zd", "cic"), ("nsl_shift", "nsl")]:
        for a, at in [(0.01, "a"), (0.05, "b")]:
            d = e[(e.scenario == sc) & (e.method == "aci-lac") & (e.alpha == a)
                  & (e.gamma == 0.01) & (e.delay == 100)]
            g = d.groupby("q").agg(m=("miscov", "mean"), lo=("miscov", "min"),
                                   hi=("miscov", "max"),
                                   b=("bound_exp", "mean")).reset_index()
            g["ep"] = g.hi - g.m
            g["em"] = g.m - g.lo
            dat(f"qsweep_{tag}_{at}", g[["q", "m", "ep", "em", "b"]])
            s = e[(e.scenario == sc) & (e.method == "static-lac")
                  & (e.alpha == a)]
            const(f"qsweepStatic{tag.upper()}{at}", s.miscov.mean())


def evasion():
    tr = pd.read_csv(RES / "e3b_trace.csv")
    dat("evasion", tr[["t", "static", "aci", "size_static", "size_aci"]])
    const("evasionTzero", tr.t.max() // 3, "{:d}")


def selective():
    e = pd.read_csv(RES / "e4_selective.csv")
    l = pd.read_csv(RES / "e3c_lost_openings.csv")
    m = lambda d: d.groupby("drop").mean(numeric_only=True).reset_index()
    a = m(e[(e.scenario == "cic_iid") & (e.alpha == 0.01)])
    b = m(e[(e.scenario == "cic_iid") & (e.alpha == 0.05)])
    dat("sel_a", pd.DataFrame(dict(drop=a["drop"], meas=a.cond_unsafe,
                                   bnda=a.bound_hp, bndb=b.bound_hp)))
    za = m(e[(e.scenario == "cic_zd") & (e.alpha == 0.01)])
    zb = m(e[(e.scenario == "cic_zd") & (e.alpha == 0.05)])
    dat("sel_b", pd.DataFrame(dict(drop=za["drop"], acia=za.aci_on_accepted,
                                   acib=zb.aci_on_accepted,
                                   static=za.cond_unsafe)))
    cols = {}
    for a_, at in [(0.01, "a"), (0.05, "b")]:
        for pol in ["impute", "skip"]:
            h = l[(l.alpha == a_) & (l.policy == pol)].groupby("rho").mean(
                numeric_only=True)
            cols["rho"] = h.index.values
            cols[f"{pol}{at}"] = h.miscov.values
    dat("sel_c", pd.DataFrame(cols))


def bytes_():
    d = pd.read_csv(RES / "e6_bytes.csv")
    b = pd.read_csv(RES / "e6_baselines.csv")
    d = d[(d.scenario == "nsl_iid") & (d["mode"] == "SET")]
    for sch, tag in [("Falcon-512", "falcon"), ("ML-DSA-44", "mldsaA"),
                     ("ML-DSA-65", "mldsaB"), ("SLH-DSA-SHA2-128s", "slh")]:
        dat(f"bytes_{tag}", d[d.scheme == sch][["N", "measured"]])
    nb = b[b.scenario == "nsl_iid"].iloc[0]
    cb = b[b.scenario == "cic_iid"].iloc[0]
    const("bytesNSLraw", nb.raw_mean)
    const("bytesCICfeat", cb.feat_bytes)
    const("bytesCICwire", cb.raw_median)
    const("bytesNSLzlib", b[(b.scenario == "nsl_iid") & (b.N == 128)]
          .evid_zlib_mean.iloc[0])
    const("bytesBsem", P.claim_bytes(P.MODE_SET, 5), "{:d}")


def channel():
    """Set mode, ML-DSA-65, 200-B frames, eps = 0.1 (exp_bandwidth.e7fig):
    exact i.i.d. expectations, Monte Carlo with 95% intervals."""
    c = pd.read_csv(RES / "e7_fig.csv")
    d = c[(c.b == 3) & (c.eps == 0.1)]
    for mode, mt in [("MONO", "mono"), ("FIV", "fiv"),
                     ("FIV+FEC", "fec"), ("FIV+FEC-IL", "il")]:
        g = d[d["mode"] == mode]
        dat(f"chan_{mt}", g[["N", "exact", "mc_iid", "ci_iid", "mc_ge",
                             "ci_ge"]])
    o = pd.read_csv(RES / "e7_opt.csv")
    o = o[(o.b == 3) & (o.eps == 0.1)].iloc[0]
    const("chanNopt", o.N_relaxed)
    const("chanNexact", int(o.N_exact), "{:d}")


def perf_sweep():
    """Integrated performability against N (exp_integrated.sweep)."""
    d = pd.read_csv(RES / "e10_sweep.csv")
    for ch in ["iid", "bursty"]:
        for des, dt in [("MONO", "mono"), ("FIV+FEC-IL", "il")]:
            for pol, R, pt in [("static", 0, "st"), ("aci", 0, "a0"),
                               ("aci", 3, "a3")]:
                x = d[(d.channel == ch) & (d.design == des)
                      & (d.policy == pol) & (d.R == R)]
                g = x.groupby("N").agg(
                    gam=("gamma_fwd", "mean"), gsd=("gamma_fwd", "std"),
                    haz=("haz_fwd", "mean"), hsd=("haz_fwd", "std"),
                    mis=("miscov", "mean"), msd=("miscov", "std")
                ).reset_index()
                g.loc[g.gam <= 0, ["gam", "gsd"]] = np.nan
                g.loc[g.haz <= 0, ["haz", "hsd"]] = np.nan
                dat(f"perf_{ch}_{dt}_{pt}", g)


def accountability():
    e = pd.read_csv(RES / "e5_accountability.csv")
    e = e[e.kind == "detect"]
    for (beta, q), tag in [((0.01, 0.01), "a"), ((0.01, 0.05), "b"),
                           ((0.05, 0.01), "c"), ((0.001, 0.05), "d")]:
        g = e[(e.beta == beta) & (e.q == q)]
        n = np.logspace(2, math.log10(20000), 80)
        dat(f"acc_{tag}_th", pd.DataFrame(dict(
            n=n, p=1 - (1 - q * g.beta_eff.iloc[0]) ** n)))
        # 95% Wilson intervals for the 2000 simulated sessions per point
        n_t, p = 2000, g.p_detect.to_numpy()
        z = 1.96
        c = (p + z * z / (2 * n_t)) / (1 + z * z / n_t)
        h = z * np.sqrt(p * (1 - p) / n_t + z * z / (4 * n_t * n_t)) \
            / (1 + z * z / n_t)
        dat(f"acc_{tag}_mc", pd.DataFrame(dict(
            n=g.n.to_numpy(), p_detect=p, ep=c + h - p, em=p - (c - h))))


def tolerance():
    tr = pd.read_csv(RES / "e9_trace.csv")
    tol = pd.read_csv(RES / "e9_tolerance.csv")
    for strat, st in [("answer", "ans"), ("withhold", "wh")]:
        for pol, pt in [("static", "static"), ("aci", "aci"),
                        ("aci+key", "key")]:
            d = tr[(tr.strategy == strat) & (tr.policy == pol)]
            dat(f"tol_{st}_{pt}", d[["t", "miscov"]])
        x = tol[(tol.seed == 0) & (tol.aseed == 0) & (tol.q == 0.01)
                & (tol.alpha == 0.01) & (tol.strategy == strat)
                & (tol.policy == "aci+key")].iloc[0]
        te = x.t_effective
        const(f"tolKey{st}", te if te == te else -1, "{:.0f}")


if __name__ == "__main__":
    shift_traces()
    q_sweep()
    evasion()
    selective()
    bytes_()
    channel()
    perf_sweep()
    accountability()
    tolerance()
    (OUT / "consts.tex").write_text("\n".join(CONSTS) + "\n")
    print(len(list(OUT.glob("*.dat"))), "data files")
    print((OUT / "consts.tex").read_text())
