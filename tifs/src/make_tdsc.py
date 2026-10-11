"""Emit the LaTeX table bodies of the TDSC manuscript (tdsc/paper/).

Reuses the TIFS table generators with a different output directory and adds
the intrusion-tolerance table (E9) and the decision-level performability
table, which combines the erasure-link results (E7) with the zero-day
certificate results (E3).
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

import make_tables as MT
from common import RES

OUT = Path(__file__).resolve().parents[2] / "tdsc" / "paper"
MT.OUT = OUT

FRAME_HDR = 8      # as in exp_bandwidth.py
B_X_CIC = 312      # CIC-IDS-2017 evidence (78 float32 features)


def tolerance():
    d = pd.read_csv(RES / "e9_tolerance.csv")
    d = d[d.alpha == 0.01]
    lab = {"static": "static", "aci": "ACI",
           "static+revoke": "static+key",
           "aci+revoke": "ACI+key"}
    lines = []
    for q in [0.01, 0.05]:
        first = True
        for strat, sl in [("answer", "answers"),
                          ("withhold", "withholds")]:
            for i, pol in enumerate(lab):
                x = d[(d.q == q) & (d.strategy == strat) & (d.policy == pol)]
                tr = x.t_revoke.mean()
                row = [(r"\multirow{8}{*}{" + f"{q}" + "}") if first else "",
                       sl if i == 0 else "", lab[pol],
                       MT.pm(x.miscov.mean(), x.miscov.std(), 4),
                       f"{x.mean_size.mean():.2f}",
                       f"{x.fals_acc.mean():,.0f}",
                       f"{x.fals_acted.mean():,.0f}",
                       (f"{tr / 1000:.0f}k" if "revoke" in pol else "--")]
                first = False
                lines.append(" & ".join(row) + r" \\")
            if strat == "answer":
                lines.append(r"\cmidrule(l){2-8}")
        lines.append(r"\midrule")
    (OUT / "tab_tolerance.tex").write_text("\n".join(lines[:-1]) + "\n")
    # numbers quoted in the text
    n = d[d.strategy == "none"]
    a = d[(d.strategy == "answer") & (d.policy == "aci+revoke")]
    (OUT / "num_tolerance.tex").write_text(
        f"\\newcommand{{\\tolFalseQuar}}{{{int(pd.read_csv(RES / 'e9_tolerance.csv').false_quar.sum())}}}\n"
        f"\\newcommand{{\\tolNoneACI}}{{{n[(n.q == 0.01) & (n.policy == 'aci')].miscov.mean():.4f}}}\n"
        f"\\newcommand{{\\tolFalsAccA}}{{{a[a.q == 0.01].fals_acc.mean():.0f}}}\n"
        f"\\newcommand{{\\tolFalsAccB}}{{{a[a.q == 0.05].fals_acc.mean():.0f}}}\n")


def opening_bytes(N, M):
    a = B_X_CIC + 16 + 32 * math.ceil(math.log2(N))
    return math.ceil(a / M) * (M + FRAME_HDR)


def performability():
    """Correct autonomous decisions and wrong autonomous actions per
    kilobyte on air, ML-DSA-65, 200-B frames, 10% erasures, zero-day
    stream, alpha = 0.01.  Each design at the batch size that maximizes its
    verified claims per byte (E7).  Link loss is independent of claim
    content, so the delivered claims are a content-independent thinning of
    the emitted stream and the per-claim rates of E3 carry over."""
    c = pd.read_csv(RES / "e7_channel.csv")
    c = c[(c.scheme == "ML-DSA-65") & (c.M == 200) & (c.eps == 0.1)]
    s = pd.read_csv(RES / "e3_shift.csv")
    s = s[(s.scenario == "cic_zd") & (s.alpha == 0.01)]
    st = s[s.method == "static-lac"]
    ac = s[(s.method == "aci-lac") & (s.gamma == 0.01) & (s.delay == 100)
           & (s.q == 0.01)]
    rates = {
        "static": (st.singleton.mean() - st.singleton_err.mean(),
                   st.singleton_err.mean(), 0.0),
        "ACI": (ac.singleton.mean() - ac.singleton_err.mean(),
                ac.singleton_err.mean(), 0.01)}
    lines, nums = [], {}
    for chan, col, cl in [("i.i.d.", "mc_verified", "eff_mc"),
                          ("bursty", "mc_verified_ge", "eff_mc_ge")]:
        first = True
        for mode, ml in [("MONO", "MONO"), ("FIV+FEC", "FIV+FEC"),
                         ("FIV+FEC-IL", "FIV+FEC-IL")]:
            x = c[c["mode"] == mode].dropna(subset=[cl])
            b = x.loc[x[cl].idxmax()]
            N = int(b.N)
            on_air = b.frames * (200 + FRAME_HDR)
            ver = b[col]
            for rl, (good, bad, q) in rates.items():
                kb = (on_air + N * q * opening_bytes(N, 200)) / 1000
                row = [(r"\multirow{6}{*}{" + chan + "}") if first else "",
                       ml if rl == "static" else "",
                       str(N) if rl == "static" else "", rl,
                       f"{ver / kb:.1f}", f"{ver * good / kb:.1f}",
                       f"{ver * bad / kb:.2f}",
                       f"{max(0.0, ver * (1 - good - bad) / kb):.1f}"]
                nums[(chan, mode, rl)] = (ver * good / kb, ver * bad / kb)
                first = False
                lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
    (OUT / "tab_perf.tex").write_text("\n".join(lines[:-1]) + "\n")
    for k, v in nums.items():
        print(k, [round(t, 2) for t in v])


if __name__ == "__main__":
    MT.conformance()
    MT.static()
    MT.shift()
    MT.crypto()
    tolerance()
    performability()
    for f in sorted(OUT.glob("tab_*.tex")):
        print("==", f.name)
        print(f.read_text())
