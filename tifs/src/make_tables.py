"""Emit LaTeX table bodies from the result CSVs (no hand transcription)."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from common import RES
import pcsp2 as P

OUT = Path(__file__).resolve().parents[1] / "paper"


def pm(m, s, d=3):
    return f"${m:.{d}f}\\pm{s:.{d}f}$"


def conformance():
    c = pd.read_csv(RES / "e1_conformance.csv")
    names = {
        "honest": "Honest batch",
        "A1_replay_old": "A1 replay after window",
        "A1b_replay_inwin": "A1b replay within window",
        "A1c_withhold": "A1c withhold and release",
        "A2_rogue_key": "A2 rogue-key forgery",
        "A2b_insider_rogue_model": "A2b insider, unapproved model",
        "A2c_bundle_swap": "A2c on-path bundle-index swap",
        "A3_claim_tamper": "A3 claim tamper",
        "A4_epoch_rewrite": "A4 epoch rewrite",
        "A5_dup_leaf": "A5 duplicate-leaf extension",
        "A5b_truncate": "A5b truncation",
        "A6_cross_sender": "A6 cross-sender splice",
    }
    gate = {"G1_sig": "G1", "G2_bundle": "G2", "G3_fresh": "G3",
            "G3_replay": "G3", "G4_struct": "G4", "ok": "--"}
    designs = ["PCSP", "-sig", "-bundle", "-fresh", "-replay", "v1-tree"]
    lines = []
    for a, lab in names.items():
        row = [lab]
        for d in designs:
            r = c[(c.attack == a) & (c.design == d)].iloc[0]
            row.append("acc." if r.accept_rate == 1 else
                       ("rej." if r.accept_rate == 0 else f"{r.accept_rate:.2f}"))
        g = c[(c.attack == a) & (c.design == "PCSP")].iloc[0].gate
        row.append(gate.get(g, g))
        lines.append(" & ".join(row) + r" \\")
    a3b = c[c.attack == "A3b_evidence_swap_at_audit"].iloc[0]
    lines.append("A3b evidence swap at audit & "
                 + ("rej." if a3b.accept_rate == 0 else f"{a3b.accept_rate}")
                 + r" & \multicolumn{5}{c}{n/a} & audit \\")
    (OUT / "tab_conformance.tex").write_text("\n".join(lines) + "\n")


def static():
    e = pd.read_csv(RES / "e2_static.csv")
    e = e[e.bits == 8]
    lines = []
    for sc, lab in [("nsl_iid", "NSL-KDD"), ("cic_iid", "CIC-IDS-2017")]:
        first = True
        for a in [0.01, 0.05]:
            for meth, ml in [("lac", "LAC"), ("aps", "APS"), ("lac-M", "LAC-M"),
                             ("aps-M", "APS-M")]:
                d = e[(e.scenario == sc) & (e.alpha == a) & (e.method == meth)]
                row = [(r"\multirow{8}{*}{" + lab + "}") if first else "",
                       f"{a}" if meth == "lac" else "", ml,
                       pm(d.coverage.mean(), d.coverage.std(), 4),
                       pm(d.mean_size.mean(), d.mean_size.std(), 2),
                       f"{d.singleton.mean():.3f}",
                       f"{d.min_class_cov.mean():.3f}"]
                first = False
                lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
    lines = lines[:-1]
    (OUT / "tab_static.tex").write_text("\n".join(lines) + "\n")
    acc = e.groupby("scenario").top1_acc.agg(["mean", "std"])
    ncal = e.groupby("scenario").n_cal.first()
    (OUT / "num_static.tex").write_text(
        f"\\newcommand{{\\accNSL}}{{{100 * acc.loc['nsl_iid', 'mean']:.2f}}}\n"
        f"\\newcommand{{\\accCIC}}{{{100 * acc.loc['cic_iid', 'mean']:.2f}}}\n"
        f"\\newcommand{{\\ncalNSL}}{{{ncal.loc['nsl_iid']:,}}}\n"
        f"\\newcommand{{\\ncalCIC}}{{{ncal.loc['cic_iid']:,}}}\n")


def shift():
    e = pd.read_csv(RES / "e3_shift.csv")
    lines = []
    for sc, lab in [("cic_zd", "CIC zero-day"),
                    ("nsl_shift", "NSL KDDTest+")]:
        first = True
        for a in [0.01, 0.05]:
            rows = [("static-lac", "static LAC", None),
                    ("static-aps", "static APS", None),
                    ("aci-lac", "ACI, $q=0.01$", 0.01),
                    ("aci-lac", "ACI, $q=0.05$", 0.05)]
            for i, (m, ml, q) in enumerate(rows):
                d = e[(e.scenario == sc) & (e.alpha == a) & (e.method == m)]
                if q is not None:
                    d = d[(d.q == q) & (d.gamma == 0.01) & (d.delay == 100)]
                row = [(r"\multirow{8}{*}{\rotatebox{90}{\scriptsize " + lab + "}}") if first else "",
                       f"{a}" if i == 0 else "", ml,
                       pm(d.miscov.mean(), d.miscov.std(), 4),
                       f"{d.mean_size.mean():.2f}",
                       f"{d.singleton.mean():.3f}",
                       f"{d.singleton_err.mean():.4f}",
                       (f"{d.bound_exp.mean():.4f}" if q is not None else "--")]
                first = False
                lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
    (OUT / "tab_shift.tex").write_text("\n".join(lines[:-1]) + "\n")


def crypto():
    b = pd.read_csv(RES / "pqc_bench.csv")
    b = b[(b.msg_bytes == 96) & (b.impl == "liboqs")]
    lat = pd.read_csv(RES / "e8_latency.csv")
    by = pd.read_csv(RES / "e6_bytes.csv")
    rows = [("ML-DSA-44", "ML-DSA-44", 2), ("ML-DSA-65", "ML-DSA-65", 3),
            ("ML-DSA-87", "ML-DSA-87", 5),
            ("SLH-DSA-SHA2-128s", "SLH-DSA-SHA2-128s", 1),
            ("SLH-DSA-SHA2-192s", "SLH-DSA-SHA2-192s", 3),
            ("Falcon-512", "Falcon-512", 1)]
    lines = []
    for name, mech, cat in rows:
        d = b[b.mechanism == mech]
        if d.empty:
            continue
        sig = int(by[by.scheme == name].sig_bytes.iloc[0])
        pk = int(d.pk_bytes.iloc[0])
        sign = d[d.op == "sign"].median_ms.iloc[0]
        ver = d[d.op == "verify"].median_ms.iloc[0]
        pc = by[(by.scheme == name) & (by.N == 128) & (by.scenario == "nsl_iid")
                & (by["mode"] == "SET")].measured.iloc[0]
        Bo = P.batch_overhead(sig)
        nstar = math.ceil(Bo / (128.29 - 3))
        l = lat[(lat.scheme == name) & (lat.N == 1024)]
        v1024 = f"{l.verify_ms.iloc[0]:.2f}" if len(l) else "--"
        fmt = (lambda x: f"{x:.3f}" if x < 10 else f"{x:.0f}")
        lines.append(" & ".join([name, str(cat), str(sig), str(pk), fmt(sign),
                                 fmt(ver), f"{pc:.1f}", str(nstar), v1024])
                     + r" \\")
    (OUT / "tab_crypto.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    conformance()
    static()
    shift()
    crypto()
    for f in sorted(OUT.glob("tab_*.tex")):
        print("==", f.name)
        print(f.read_text())
