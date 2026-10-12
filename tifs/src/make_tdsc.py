"""Emit the LaTeX table bodies and quoted numbers of the TDSC manuscript
(tdsc/paper/).  Every number comes from tifs/results/*.csv.

Tables
  tab_conformance, tab_crypto   as in the TIFS version (make_tables.py)
  tab_static    static certificates on the i.i.d. folds, alpha = 0.01
  tab_shift     static vs audit-driven certificates on the shifted streams,
                with decision-risk columns (exp_risk.py, E3d)
  tab_tolerance compromised sender (exp_tolerance.py, E9), q = 0.01
  tab_perf      integrated performability (exp_integrated.py, E10)
Numbers
  num_static.tex, num_tdsc.tex
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import make_tables as MT
from common import RES

OUT = Path(__file__).resolve().parents[2] / "tdsc" / "paper"
MT.OUT = OUT
NUM: dict[str, str] = {}


def num(name, v, fmt="{:.4f}"):
    NUM[name] = fmt.format(v)


def static_compact():
    """Static certificates on the i.i.d. folds at alpha = 0.01 only."""
    e = pd.read_csv(RES / "e2_static.csv")
    e = e[(e.bits == 8) & (e.alpha == 0.01)]
    lines = []
    for sc, lab in [("nsl_iid", "NSL-KDD"), ("cic_iid", "CIC-IDS-2017")]:
        for i, (meth, ml) in enumerate([("lac", "LAC"), ("aps", "APS"),
                                        ("lac-M", "LAC-M"), ("aps-M", "APS-M")]):
            d = e[(e.scenario == sc) & (e.method == meth)]
            row = [(r"\multirow{4}{*}{" + lab + "}") if i == 0 else "", ml,
                   MT.pm(d.coverage.mean(), d.coverage.std(), 4),
                   f"{d.mean_size.mean():.2f}",
                   f"{d.singleton.mean():.4f}",
                   f"{d.min_class_cov.mean():.3f}"]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
    (OUT / "tab_static.tex").write_text("\n".join(lines[:-1]) + "\n")


def shift_risk():
    r = pd.read_csv(RES / "e3d_risk.csv")
    e = pd.read_csv(RES / "e3_shift.csv")
    lines = []
    for sc, lab in [("cic_zd", "CIC zero-day"), ("nsl_shift", "KDDTest+")]:
        first = True
        for a in [0.01, 0.05]:
            for i, (m, ml, q) in enumerate([("static", "static", 0),
                                            ("aci", "ACI, $q=0.01$", 0.01),
                                            ("aci", "ACI, $q=0.05$", 0.05)]):
                d = r[(r.scenario == sc) & (r.alpha == a) & (r.method == m)
                      & (r.q == q)]
                row = [(r"\multirow{6}{*}{\rotatebox{90}{\scriptsize " + lab
                        + "}}") if first else "",
                       f"{a}" if i == 0 else "", ml,
                       MT.pm(d.miscov.mean(), d.miscov.std(), 4),
                       f"{d.mean_size.mean():.2f}",
                       f"{d.autonomous.mean():.3f}",
                       f"{d.wrong_act.mean():.4f}",
                       f"{d.sel_err.mean():.3f}",
                       f"{d.false_benign.mean():.3f}"]
                first = False
                lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
    (OUT / "tab_shift.tex").write_text("\n".join(lines[:-1]) + "\n")
    # consistency: the ACI rows reproduce exp_conformal.e3 exactly
    for sc in ["cic_zd", "nsl_shift"]:
        for a in [0.01, 0.05]:
            for q in [0.01, 0.05]:
                x = r[(r.scenario == sc) & (r.alpha == a) & (r.method == "aci")
                      & (r.q == q)].miscov.mean()
                y = e[(e.scenario == sc) & (e.alpha == a) & (e.q == q)
                      & (e.method == "aci-lac") & (e.gamma == 0.01)
                      & (e.delay == 100)].miscov.mean()
                assert abs(x - y) < 1e-12, (sc, a, q, x, y)
    g = lambda sc, a, m, q, c: r[(r.scenario == sc) & (r.alpha == a)
                                 & (r.method == m) & (r.q == q)][c].mean()
    num("riskSelACIcic", 100 * g("cic_zd", 0.01, "aci", 0.01, "sel_err"), "{:.1f}")
    num("riskSelRandcic", 100 * g("cic_zd", 0.01, "random", 0.01, "sel_err"), "{:.1f}")
    num("riskSelThrcic", 100 * g("cic_zd", 0.01, "conf-thr", 0, "sel_err"), "{:.1f}")
    num("riskFBstatic", 100 * g("cic_zd", 0.01, "static", 0, "false_benign"), "{:.1f}")
    num("riskFBaci", 100 * g("cic_zd", 0.01, "aci", 0.01, "false_benign"), "{:.1f}")
    num("riskFBrand", 100 * g("cic_zd", 0.01, "random", 0.01, "false_benign"), "{:.1f}")
    num("riskWinACI", g("cic_zd", 0.01, "aci", 0.01, "worst_win"), "{:.2f}")
    num("riskWinStatic", g("cic_zd", 0.01, "static", 0, "worst_win"), "{:.2f}")
    num("riskIWcic", g("cic_zd", 0.01, "iw-aci", 0.01, "miscov"))
    num("riskIWsizecic", g("cic_zd", 0.01, "iw-aci", 0.01, "mean_size"), "{:.2f}")
    num("riskIWnsl", g("nsl_shift", 0.01, "iw-aci", 0.01, "miscov"))
    num("riskIWsizensl", g("nsl_shift", 0.01, "iw-aci", 0.01, "mean_size"), "{:.2f}")
    num("riskSelACInsl", 100 * g("nsl_shift", 0.01, "aci", 0.01, "sel_err"), "{:.1f}")
    num("riskSelRandnsl", 100 * g("nsl_shift", 0.01, "random", 0.01, "sel_err"), "{:.1f}")
    lab = pd.read_csv(RES / "e3d_labels.csv").groupby("labeler").mean(
        numeric_only=True)
    for k, n in [("flip 0.05", "Flip"), ("model top-1", "Model")]:
        num(f"lab{n}Mis", lab.loc[k, "miscov"])
        num(f"lab{n}Size", lab.loc[k, "mean_size"], "{:.2f}")


def tolerance():
    d = pd.read_csv(RES / "e9_tolerance.csv")
    d = d[d.alpha == 0.01]
    pols = [("static", "static"), ("aci", "ACI"), ("aci+key", "ACI+key")]
    lines = []
    for strat, sl in [("answer", "answers"), ("withhold", "withholds"),
                      ("withhold-low", r"low-rate")]:
        for i, (pol, pl) in enumerate(pols):
            x = d[(d.q == 0.01) & (d.strategy == strat) & (d.policy == pol)]
            if pol == "aci+key":
                hit = x.t_trigger.notna()
                key = (f"{hit.sum()}/{len(x)}, "
                       f"{x.t_effective[hit].mean() / 1000:.0f}k"
                       if hit.any() else f"0/{len(x)}")
            else:
                key = "--"
            row = [(r"\multirow{3}{*}{" + sl + "}") if i == 0 else "", pl,
                   MT.pm(x.miscov.mean(), x.miscov.std(), 4),
                   f"{x.singleton.mean():.2f}",
                   f"{(x.wrong_act / x.singleton).mean():.3f}",
                   f"{x.fals_acc.mean():,.0f}",
                   f"{x.fals_acted.mean():,.0f}", key]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
    (OUT / "tab_tolerance.tex").write_text("\n".join(lines[:-1]) + "\n")
    a = d[d.strategy == "answer"]
    for q, t in [(0.01, "A"), (0.05, "B")]:
        k = a[(a.q == q) & (a.policy == "aci+key")]
        num(f"tolFalsAcc{t}", k.fals_acc.mean(), "{:.0f}")
        num(f"tolFalsMed{t}", k.fals_acc.median(), "{:.0f}")
        num(f"tolFalsMax{t}", k.fals_acc.max(), "{:.0f}")
        num(f"tolTrig{t}", k.t_trigger.mean() / 1000, "{:.0f}")
        num(f"tolRev{t}", k.t_trigger.notna().sum(), "{:d}")
        w = d[(d.strategy == "withhold") & (d.q == q) & (d.policy == "aci+key")]
        num(f"tolQuar{t}", w.t_trigger.notna().sum(), "{:d}")
        num(f"tolQuarAt{t}", w.t_trigger.mean() / 1000, "{:.0f}")
    x = d[(d.strategy == "answer") & (d.q == 0.01) & (d.policy == "static")]
    num("tolTargeted", x.targeted.mean(), "{:,.0f}")
    num("tolFalsified", x.falsified.mean(), "{:,.0f}")
    num("tolModelCons", x.model_consistent_targeted.mean(), "{:,.0f}")
    num("tolFalsLow", d[d.strategy == "withhold-low"].falsified.mean(), "{:.0f}")
    full = pd.read_csv(RES / "e9_tolerance.csv")
    one = full.drop_duplicates(["seed", "aseed", "q", "strategy", "alpha"])
    honest = (one.strategy == "none").sum() * 5 \
        + (one.strategy != "none").sum() * 4
    num("tolHonestSessions", int(honest), "{:,d}")
    num("tolFalseQuar", int(full.false_quar.sum()), "{:d}")
    num("tolFalseFlag", int(full.false_flags.sum()), "{:d}")
    n = d[(d.strategy == "none") & (d.q == 0.01) & (d.policy == "aci")]
    num("tolNoneACI", n.miscov.mean())
    num("tolNoneSize", n.mean_size.mean(), "{:.2f}")
    num("tolNoneAuto", 100 * n.singleton.mean(), "{:.0f}")
    for strat, t in [("answer", "Ans"), ("withhold", "Wh")]:
        x = d[(d.strategy == strat) & (d.q == 0.01) & (d.policy == "aci")]
        num(f"tolAuto{t}", 100 * x.singleton.mean(), "{:.0f}")
        num(f"tolWrong{t}", 100 * x.wrong_act.mean(), "{:.1f}")
        num(f"tolSel{t}", 100 * (x.wrong_act / x.singleton).mean(), "{:.1f}")
    k = d[(d.strategy == "answer") & (d.q == 0.01) & (d.policy == "aci+key")]
    num("tolAutoKey", 100 * k.singleton.mean(), "{:.0f}")


def performability():
    import numpy as np
    d = pd.read_csv(RES / "e10_integrated.csv")
    d = d[d.repair == "selective"].assign(
        feasible=lambda x: x.feasible.astype(str) == "True")
    rec = [("static", 0, "static"), ("aci", 0, "ACI, no repair"),
           ("aci", 3, "ACI, $R=3$")]
    lines = []
    for ch, cl in [("iid", "i.i.d."), ("bursty", "bursty")]:
        first = True
        for des in ["MONO", "FIV+FEC-IL"]:
            for i, (pol, R, rl) in enumerate(rec):
                x = d[(d.channel == ch) & (d.design == des) & (d.policy == pol)
                      & (d.R == R)]
                kb = x.fwd_bytes / 1000
                feas = "" if x.feasible.iloc[0] else r"$^\dagger$"
                row = [(r"\multirow{6}{*}{" + cl + "}") if first else "",
                       des if i == 0 else "", rl,
                       f"{int(x.N.iloc[0])}{feas}",
                       f"{(x.accepted / kb).mean():.1f}",
                       f"{x.gamma_fwd.mean():.1f}",
                       f"{x.haz_fwd.mean():.2f}",
                       f"{(x.escal / kb).mean():.1f}",
                       (f"{x.open_lost.sum() / x.audits.sum():.3f}"
                        if pol == "aci" else "--"),
                       f"{x.miscov.mean():.4f}"]
                first = False
                lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
    (OUT / "tab_perf.tex").write_text("\n".join(lines[:-1]) + "\n")
    sel = lambda ch, des, pol, R: d[(d.channel == ch) & (d.design == des)
                                    & (d.policy == pol) & (d.R == R)]
    g = lambda ch, des, pol, R, c: sel(ch, des, pol, R)[c].mean()
    rng = np.random.default_rng(0)
    for ch, t in [("bursty", "B"), ("iid", "I")]:
        des = "FIV+FEC-IL"
        gs, hs = g(ch, des, "static", 0, "gamma_fwd"), g(ch, des, "static", 0, "haz_fwd")
        ga, ha = g(ch, des, "aci", 3, "gamma_fwd"), g(ch, des, "aci", 3, "haz_fwd")
        num(f"perfGs{t}", gs, "{:.1f}")
        num(f"perfHs{t}", hs, "{:.1f}")
        num(f"perfGa{t}", ga, "{:.1f}")
        num(f"perfHa{t}", ha, "{:.2f}")
        num(f"perfGratio{t}", gs / ga, "{:.1f}")
        num(f"perfHratio{t}", hs / ha, "{:.0f}")
        # cluster bootstrap over model seeds (each with its five channel
        # runs) for the ratios of means
        S = sel(ch, des, "static", 0).set_index(["seed", "cseed"])
        A_ = sel(ch, des, "aci", 3).set_index(["seed", "cseed"])
        seeds = np.array(sorted(S.index.get_level_values(0).unique()))
        rg, rh = [], []
        for _ in range(10000):
            bs = rng.choice(seeds, len(seeds))
            s_ = pd.concat([S.loc[b] for b in bs])
            a_ = pd.concat([A_.loc[b] for b in bs])
            rg.append(s_.gamma_fwd.mean() / a_.gamma_fwd.mean())
            rh.append(s_.haz_fwd.mean() / a_.haz_fwd.mean())
        num(f"perfGratioLo{t}", np.percentile(rg, 2.5), "{:.1f}")
        num(f"perfGratioHi{t}", np.percentile(rg, 97.5), "{:.1f}")
        num(f"perfHratioLo{t}", np.percentile(rh, 2.5), "{:.0f}")
        num(f"perfHratioHi{t}", np.percentile(rh, 97.5), "{:.0f}")
        num(f"perfGnr{t}", g(ch, des, "aci", 0, "gamma_fwd"), "{:.1f}")
        x = sel(ch, des, "aci", 0)
        num(f"perfLost{t}", 100 * x.open_lost.sum() / x.audits.sum(), "{:.0f}")
        num(f"perfQuarNR{t}", int((x.quar_at >= 0).sum()), "{:d}")
        y = sel(ch, des, "aci", 3)
        num(f"perfLostR{t}", 100 * y.open_lost.sum() / y.audits.sum(), "{:.1f}")
        num(f"perfQuarR{t}", int((y.quar_at >= 0).sum()), "{:d}")
        num(f"perfAbove{t}", int((y.miscov > 0.01).sum()), "{:d}")
        num(f"perfMis{t}", y.miscov.mean(), "{:.4f}")
        num(f"perfEsc{t}", 10 * (y.escal / y.accepted).mean(), "{:.1f}")
        num(f"perfMisMax{t}", y.miscov.max(), "{:.3f}")
        num(f"perfAccMin{t}", y.accepted.min() / 1000, "{:.0f}")
        num(f"perfBiLoss{t}", 100 * (1 - (y.gamma_bi / y.gamma_fwd).mean()),
            "{:.1f}")
        num(f"perfN{t}", int(y.N.iloc[0]), "{:d}")
    w = pd.read_csv(RES / "e10_integrated.csv")
    w = w[(w.repair == "whole")]
    for ch, t in [("bursty", "B"), ("iid", "I")]:
        x = w[(w.channel == ch) & (w.design == "FIV+FEC-IL")]
        num(f"perfWholeLost{t}", 100 * x.open_lost.sum() / x.audits.sum(), "{:.1f}")
        num(f"perfWholeG{t}", x.gamma_fwd.mean(), "{:.1f}")
        num(f"perfWholeN{t}", int(x.N.iloc[0]), "{:d}")
    sl = pd.read_csv(RES / "e10_selected.csv")
    m = sl[(sl.channel == "iid") & (sl.design == "MONO") & (sl.policy == "aci")
           & (sl.R == 3) & (sl.repair == "selective")].iloc[0]
    num("perfMonoMisVal", m.mis_mean, "{:.3f}")
    x = d[(d.channel == "iid") & (d.design == "MONO") & (d.policy == "aci")
          & (d.R == 3)]
    num("perfMonoMis", x.miscov.mean(), "{:.3f}")
    num("perfMonoAcc", x.accepted.mean() / 1000, "{:.0f}")
    # masking labeler (exp_risk.masking)
    mk = pd.read_csv(RES / "e3d_masking.csv").groupby("budget").mean(
        numeric_only=True)
    num("maskEta", mk.loc[0.5, "eta"], "{:.4f}")
    num("maskMis", mk.loc[0.5, "miscov"], "{:.4f}")
    num("maskBound", mk.loc[0.5, "bound"], "{:.4f}")


def bandwidth():
    b = pd.read_csv(RES / "e6_baselines.csv")
    g = pd.read_csv(RES / "e6_zlib_grid.csv")
    e = pd.read_csv(RES / "e6_bytes.csv")
    nb = b[(b.scenario == "nsl_iid")]
    num("zlibA", nb[nb.N == 128].evid_zlib_mean.iloc[0], "{:.1f}")
    num("zlibB", nb[nb.N == 512].evid_zlib_mean.iloc[0], "{:.1f}")
    num("bytesCombos", len(e), "{:d}")
    assert (e.residual.abs() < 1e-9).all()
    for q, t in [(0.0, "A"), (0.01, "B")]:
        x = g[(g.q == q) & g.pcsp_cheaper]
        num(f"zlibBE{t}", int(x.N.min()), "{:d}")
    z128 = nb[nb.N == 128].evid_zlib_mean.iloc[0]
    a128 = 10 + nb.evid_mean.iloc[0] + 16 + 32 * 7
    num("zlibSaveA", z128 - 3, "{:.1f}")
    num("zlibSaveB", z128 - 3 - 0.01 * a128, "{:.1f}")
    o = pd.read_csv(RES / "e7_opt.csv")
    x = o[(o.b == 3) & (o.eps == 0.1)].iloc[0]
    num("monoNexact", int(x.N_exact), "{:d}")
    num("monoEffExact", x.eff_exact, "{:.2f}")
    num("monoNrelax", x.N_relaxed, "{:.0f}")
    x = o[(o.b == 4) & (o.eps == 0.1)].iloc[0]
    num("monoNexactS", int(x.N_exact), "{:d}")
    num("monoEffExactS", x.eff_exact, "{:.2f}")
    f = pd.read_csv(RES / "e7_fig.csv")
    f = f[(f.b == 3) & (f.eps == 0.1)]
    fec = f[f["mode"] == "FIV+FEC"]
    num("fecBest", fec.exact.max(), "{:.1f}")
    num("fecBestN", int(fec.loc[fec.exact.idxmax(), "N"]), "{:d}")
    mono = f[f["mode"] == "MONO"]
    num("geMono", mono.mc_ge.max(), "{:.1f}")
    num("geMonoN", int(mono.loc[mono.mc_ge.idxmax(), "N"]), "{:d}")
    il = f[f["mode"] == "FIV+FEC-IL"]
    num("geIL", il.mc_ge.max(), "{:.1f}")
    num("geILr", int(il.loc[il.mc_ge.idxmax(), "r"]), "{:d}")
    num("geFEC", fec.mc_ge.max(), "{:.1f}")
    ff = pd.read_csv(RES / "e7_fig.csv")
    ff = ff[ff["mode"] != "FIV+FEC-IL"]
    out = (ff.exact < ff.lo_iid) | (ff.exact > ff.hi_iid)
    num("chanOut", int(out.sum()), "{:d}")
    num("chanPts", len(ff), "{:d}")
    num("chanOutMono", int((out & (ff["mode"] == "MONO")).sum()), "{:d}")
    rc = pd.read_csv(RES / "e8_recompute.csv")
    num("recompUs", rc[rc.n == 1].median_ms.iloc[0] * 1000, "{:.0f}")
    ev = pd.read_csv(RES / "e3b_evasion.csv")
    st = ev[ev.method == "static"]
    num("evLamMed", st.lam_median.mean(), "{:.2f}")
    num("evLamFull", 100 * st.lam_full.mean(), "{:.2f}")


if __name__ == "__main__":
    MT.conformance()
    MT.static()
    static_compact()
    MT.crypto()
    shift_risk()
    tolerance()
    performability()
    bandwidth()
    (OUT / "num_tdsc.tex").write_text("".join(
        f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in NUM.items()))
    for k, v in NUM.items():
        print(k, v)
