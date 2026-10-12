"""Render all data figures for the journal manuscript (PDF, IEEE column)."""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from common import FIG, RES
from exp_bandwidth import n_opt_mono
import pcsp2 as P
from sigs import SIG_BYTES

FIG.mkdir(exist_ok=True)
# categorical slots (fixed order) + line/marker secondary encoding for print
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times",
                                           "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "lines.linewidth": 1.4, "axes.linewidth": 0.6,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.4, "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
W1, W2 = 3.5, 7.16


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


# --------------------------------------------------------------- E3 traces

def fig_shift_traces():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 1.9))
    for ax, sc, title in [(axs[0], "cic_zd", "(a) CIC-IDS-2017, zero-day stream"),
                          (axs[1], "nsl_shift", "(b) NSL-KDD, KDDTest+ stream")]:
        a = 0.01
        st = pd.read_csv(RES / f"e3_trace_{sc}_a{a}_static.csv")
        ax.plot(st.t, st.run_miscov, color=C2, ls="--", label="static split conformal")
        for q, c, ls in [(0.01, C1, "-"), (0.05, C3, "-.")]:
            tr = pd.read_csv(RES / f"e3_trace_{sc}_a{a}_q{q}.csv")
            ax.plot(tr.t, tr.run_miscov, color=c, ls=ls,
                    label=f"audit-driven ACI, $q={q}$")
        ax.axhline(a, color=INK2, lw=0.8, ls=":", label=r"target $\alpha=0.01$")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 0.5)
        ax.set_xlabel("claims received $T$")
        ax.set_title(title, loc="left")
    axs[0].set_ylabel(r"running miscoverage $\frac{1}{T}\sum_t \mathrm{err}_t$")
    axs[1].legend(loc="upper right", ncol=1)
    save(fig, "fig_shift_traces")


def fig_q_sweep():
    e = pd.read_csv(RES / "e3_shift.csv")
    fig, axs = plt.subplots(1, 2, figsize=(W2, 1.9))
    for ax, sc, title in [(axs[0], "cic_zd", "(a) CIC-IDS-2017 zero-day"),
                          (axs[1], "nsl_shift", "(b) NSL-KDD shift")]:
        for a, c, mk in [(0.01, C1, "o"), (0.05, C2, "s")]:
            d = e[(e.scenario == sc) & (e.method == "aci-lac") & (e.alpha == a)
                  & (e.gamma == 0.01) & (e.delay == 100)]
            g = d.groupby("q").agg(m=("miscov", "mean"), lo=("miscov", "min"),
                                   hi=("miscov", "max"), b=("bound_exp", "mean"))
            ax.errorbar(g.index, g.m, yerr=[g.m - g.lo, g.hi - g.m], color=c,
                        marker=mk, ms=4, capsize=2, lw=1.2,
                        label=rf"ACI, $\alpha={a}$ (mean, min--max)")
            ax.plot(g.index, g.b, color=c, ls="--", lw=1.0,
                    label=rf"Theorem 3 bound, $\alpha={a}$")
            s = e[(e.scenario == sc) & (e.method == "static-lac") & (e.alpha == a)]
            ax.axhline(s.miscov.mean(), color=c, ls=":", lw=1.0,
                       label=rf"static, $\alpha={a}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("audit rate $q$")
        ax.set_title(title, loc="left")
    axs[0].set_ylabel("miscoverage over the stream")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    save(fig, "fig_q_sweep")


def fig_evasion():
    tr = pd.read_csv(RES / "e3b_trace.csv")
    fig, axs = plt.subplots(2, 1, figsize=(W1, 2.6), sharex=True)
    T0 = tr.t.max() // 3
    axs[0].plot(tr.t, tr.static, color=C2, ls="--", label="static")
    axs[0].plot(tr.t, tr.aci, color=C1, label="audit-driven ACI ($q=0.01$)")
    axs[0].axhline(0.01, color=INK2, lw=0.8, ls=":", label=r"$\alpha=0.01$")
    axs[0].set_ylabel("miscoverage\n(2k-claim window)")
    axs[0].set_yscale("log")
    axs[0].set_ylim(1e-4, 1)
    axs[0].legend(loc="lower center", ncol=3, fontsize=6.5,
                  bbox_to_anchor=(0.5, 1.0))
    axs[1].plot(tr.t, tr.size_static, color=C2, ls="--")
    axs[1].plot(tr.t, tr.size_aci, color=C1)
    axs[1].set_ylabel("mean set size\n($K=7$)")
    axs[1].set_xlabel("claim index $t$")
    for ax in axs:
        ax.axvline(T0, color=INK, lw=0.7)
    axs[1].text(T0 * 0.97, 3.5, "evasion\nstarts", fontsize=7, color=INK,
                ha="right")
    save(fig, "fig_evasion")


def fig_selective():
    e = pd.read_csv(RES / "e4_selective.csv")
    l = pd.read_csv(RES / "e3c_lost_openings.csv")
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2))
    fig.subplots_adjust(wspace=0.32)
    ax = axs[0]
    g = e[(e.scenario == "cic_iid") & (e.alpha == 0.01)].groupby("drop").mean(
        numeric_only=True)
    ax.plot(g.index, g.cond_unsafe, color=C1, marker="o", ms=3.5,
            label=r"measured $U/n_{\rm acc}$")
    for a, c, ls in [(0.01, C1, "--"), (0.05, C2, "-.")]:
        h = e[(e.scenario == "cic_iid") & (e.alpha == a)].groupby("drop").mean(
            numeric_only=True)
        ax.plot(h.index, h.bound_hp, color=c, ls=ls, lw=1.0,
                label=rf"Corollary 1 bound, $\alpha={a}$")
    ax.set_title("(a) exchangeable, static sets", loc="left")
    ax.set_ylabel("unsafe / accepted")
    ax.legend(fontsize=6, loc="upper left")
    ax = axs[1]
    for a, c, mk in [(0.01, C1, "o"), (0.05, C2, "s")]:
        h = e[(e.scenario == "cic_zd") & (e.alpha == a)].groupby("drop").mean(
            numeric_only=True)
        ax.plot(h.index, h.aci_on_accepted, color=c, marker=mk, ms=3.5,
                label=rf"ACI on accepted, $\alpha={a}$")
    h = e[(e.scenario == "cic_zd") & (e.alpha == 0.01)].groupby("drop").mean(
        numeric_only=True)
    ax.plot(h.index, h.cond_unsafe, color=INK2, ls="--", lw=1.0,
            label="static")
    ax.set_title("(b) zero-day stream", loc="left")
    ax.legend(fontsize=6, loc="upper left")
    for ax in axs[:2]:
        ax.set_yscale("log")
        ax.set_xlabel("fragments dropped")
    ax = axs[2]
    for (a, pol), c, ls, mk in [((0.01, "impute"), C1, "-", "o"),
                                ((0.01, "skip"), C1, ":", "x"),
                                ((0.05, "impute"), C2, "-", "s"),
                                ((0.05, "skip"), C2, ":", "+")]:
        h = l[(l.alpha == a) & (l.policy == pol)].groupby("rho").mean(
            numeric_only=True)
        ax.plot(h.index, h.miscov, color=c, ls=ls, marker=mk, ms=4,
                label=rf"{pol}, $\alpha={a}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"openings dropped $\rho$")
    ax.set_ylabel("miscoverage")
    ax.set_title("(c) suppressed openings", loc="left")
    ax.legend(fontsize=6, loc="upper left", ncol=1, framealpha=0.9,
              frameon=True, edgecolor="none")
    ax.set_ylim(5e-3, 0.6)
    save(fig, "fig_selective")


def fig_bytes():
    d = pd.read_csv(RES / "e6_bytes.csv")
    b = pd.read_csv(RES / "e6_baselines.csv")
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    d = d[(d.scenario == "nsl_iid") & (d["mode"] == "SET")]
    for sch, c, ls, mk in [("Falcon-512", C3, "-", "^"), ("ML-DSA-44", C5, "-", "v"),
                           ("ML-DSA-65", C1, "-", "o"),
                           ("SLH-DSA-SHA2-128s", C2, "-", "s")]:
        g = d[d.scheme == sch]
        ax.plot(g.N, g.measured, color=c, ls=ls, marker=mk, ms=3, label=sch)
    nb = b[b.scenario == "nsl_iid"].iloc[0]
    cb = b[b.scenario == "cic_iid"].iloc[0]
    for v, lab, ls in [(nb.raw_mean, "NSL-KDD native record (128 B)", "--"),
                       (cb.feat_bytes, f"CIC flow features ({cb.feat_bytes:.0f} B)", "-."),
                       (cb.raw_median, f"CIC flow on-wire, median ({cb.raw_median:.0f} B)", ":"),
                       (b[(b.scenario == "nsl_iid") & (b.N == 128)].evid_zlib_mean.iloc[0],
                        "NSL-KDD record, zlib, N=128 (18.7 B)", (0, (1, 1)))]:
        ax.axhline(v, color=INK2, ls=ls, lw=0.9, label=lab)
    ax.axhline(P.claim_bytes(P.MODE_SET, 5), color=INK, lw=0.6)
    ax.text(1.1, 3.4, r"$B_{\rm sem}=3$ B", fontsize=7)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size $N$")
    ax.set_ylabel("bytes per claim")
    ax.legend(fontsize=6, ncol=2, loc="upper center",
              bbox_to_anchor=(0.45, -0.22))
    ax.set_ylim(2, 2e4)
    save(fig, "fig_bytes")


def fig_channel():
    c = pd.read_csv(RES / "e7_channel.csv")
    fig, axs = plt.subplots(1, 2, figsize=(W2, 1.95), sharey=True)
    for ax, eps in zip(axs, [0.05, 0.10]):
        d = c[(c.scheme == "ML-DSA-65") & (c.M == 200) & (c.eps == eps)]
        for mode, col, mk in [("MONO", C2, "s"), ("FIV", C3, "^"),
                              ("FIV+FEC", C1, "o"), ("FIV+FEC-IL", C5, "D")]:
            g = d[d["mode"] == mode]
            if mode != "FIV+FEC-IL":
                ax.plot(g.N, g.eff_theory, color=col, lw=1.2,
                        label=f"{mode} (Prop. 4)")
            ax.plot(g.N, g.eff_mc, ls="none", marker=mk, ms=3.5, color=col,
                    label=f"{mode} (Monte-Carlo, i.i.d.)")
            ax.plot(g.N, g.eff_mc_ge, ls=":", lw=1, color=col,
                    label=f"{mode} (Gilbert-Elliott)")
        no = n_opt_mono(3, P.batch_overhead(SIG_BYTES["ML-DSA-65"]), 200, eps)
        ax.axvline(no, color=C2, lw=0.7, ls="--")
        ax.text(no * 1.05, 0.15, r"$N_{\rm opt}^{\rm MONO}$", color=INK, fontsize=7)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(0.05, 400)
        ax.set_xlabel("batch size $N$")
        ax.set_title(rf"({'ab'[int(eps == 0.1)]}) frame erasure $\epsilon={eps}$,"
                     " 200-B frames, ML-DSA-65", loc="left")
    axs[0].set_ylabel("verified claims per kB on air")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.02),
               fontsize=6.3)
    save(fig, "fig_channel")


def fig_accountability():
    e = pd.read_csv(RES / "e5_accountability.csv")
    e = e[e.kind == "detect"]
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    for (beta, q), c, mk in [((0.01, 0.01), C1, "o"), ((0.01, 0.05), C2, "s"),
                             ((0.05, 0.01), C3, "^"), ((0.001, 0.05), C5, "v")]:
        g = e[(e.beta == beta) & (e.q == q)]
        n = np.logspace(2, math.log10(20000), 80)
        ax.plot(n, 1 - (1 - q * g.beta_eff.iloc[0]) ** n, color=c, lw=1.1)
        ax.plot(g.n, g.p_detect, ls="none", marker=mk, ms=4, color=c,
                label=rf"$\phi={beta}$, $q={q}$")
    ax.set_xscale("log")
    ax.set_xlabel("claims signed by the compromised sender $n$")
    ax.set_ylabel("detection probability")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.text(2300, 0.05, "lines: $1-(1-q\\phi)^n$\nmarkers: simulation",
            fontsize=6.5)
    save(fig, "fig_accountability")


def fig_tolerance():
    tr = pd.read_csv(RES / "e9_trace.csv")
    tol = pd.read_csv(RES / "e9_tolerance.csv")
    fig, axs = plt.subplots(2, 1, figsize=(W1, 2.7), sharex=True)
    for ax, strat, ttl in [(axs[0], "answer", "insider answers audits"),
                           (axs[1], "withhold", "insider withholds openings")]:
        for pol, col, ls, lab in [("static", C2, "--", "static"),
                                  ("aci", C1, "-", "ACI"),
                                  ("aci+key", C3, "-.",
                                   "ACI + key action")]:
            d = tr[(tr.strategy == strat) & (tr.policy == pol)]
            ax.plot(d.t, d.miscov, color=col, ls=ls, lw=1.1, label=lab)
        tk = tol[(tol.seed == 0) & (tol.aseed == 0) & (tol.q == 0.01)
                 & (tol.alpha == 0.01) & (tol.strategy == strat)
                 & (tol.policy == "aci+key")].t_effective.iloc[0]
        if tk == tk:                      # no key action in this run: NaN
            ax.axvline(tk, color=INK, lw=0.7)
            ax.text(tk, 3e-4, " key\n action", fontsize=6.5, color=INK,
                    va="center")
        ax.axhline(0.01, color=INK2, lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1)
        ax.set_ylabel("miscoverage\n(2k-claim window)")
        ax.set_title(ttl, fontsize=7, pad=2)
    axs[0].legend(loc="lower center", ncol=3, fontsize=6.5,
                  bbox_to_anchor=(0.5, 1.12))
    axs[1].set_xlabel("claim index $t$ (zero-day stream)")
    axs[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v / 1000:.0f}k"))
    save(fig, "fig_tolerance")


if __name__ == "__main__":
    fig_shift_traces()
    fig_q_sweep()
    fig_evasion()
    fig_selective()
    fig_bytes()
    fig_channel()
    fig_accountability()
    fig_tolerance()
    print("ok")
