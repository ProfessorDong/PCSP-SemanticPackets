"""
E2 (static certificates, exchangeable data) and E3 (shift: static vs
audit-driven ACI) for the PCSP journal paper.

Outputs
  results/e2_static.csv     per (scenario, seed, alpha, method, precision)
  results/e3_shift.csv      per (scenario, seed, audit seed, alpha, q,
                             gamma, delay, method)
  results/e3_trace_<scenario>.csv   running miscoverage traces
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

import conformal as C
from common import RES, scenario

ALPHAS = [0.01, 0.05, 0.10]
METHODS = [("lac", False), ("aps", False), ("lac", True), ("aps", True)]


def quantized(P, bits):
    if bits is None:
        return P
    L = (1 << bits) - 1
    return C.dequantize(C.quantize(P, L), L)


def e2(seeds):
    rows = []
    for name in ["nsl_iid", "cic_iid"]:
        for seed in seeds:
            s = scenario(name, seed)
            z = s["P"].argmax(1)
            for bits in [None, 8, 4]:
                Pca, P = quantized(s["Pca"], bits), quantized(s["P"], bits)
                for a in ALPHAS:
                    for meth, mond in METHODS:
                        cp = C.SplitConformal(meth, mond, a, seed).fit(
                            Pca, s["yca"])
                        m = C.set_metrics(cp.sets(P), s["y"], z)
                        rows.append(dict(
                            scenario=name, seed=seed, alpha=a,
                            method=meth + ("-M" if mond else ""),
                            bits=bits or 64, coverage=m["coverage"],
                            mean_size=m["mean_size"],
                            singleton=m["singleton"],
                            singleton_err=m["singleton_err"],
                            top1_acc=m["top1_acc"],
                            min_class_cov=m["min_class_cov"],
                            n_cal=len(s["yca"]), n_test=len(s["y"]),
                            **{f"cc{k}": v for k, v in
                               enumerate(m["class_cov"])}))
            print(name, seed, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "e2_static.csv", index=False)
    return df


def bound_thm3(alpha, alpha1, gamma, q, T, D, delta=0.05):
    """Theorem 3: expectation bound and high-probability bound."""
    c = (alpha1 / gamma + 2 * D * (1 - alpha)) / (q * T)
    L = math.log(1 / delta)
    hp = c + math.sqrt(2 * L / (q * T)) + 2 * L / (3 * q * T)
    return alpha + c, alpha + hp


def pending_max(audit, delay):
    """Largest number of audited claims whose label is outstanding."""
    a = audit.astype(np.int64)
    c = np.cumsum(a)
    lag = np.concatenate([np.zeros(delay + 1, np.int64), c[:-delay - 1]]) \
        if delay + 1 <= len(c) else np.zeros_like(c)
    return int((c - lag).max())


def e3(seeds, audit_seeds, traces=True):
    rows = []
    qs = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    gammas = [0.005, 0.01, 0.02, 0.05]
    delays = [0, 100, 1000]
    for name in ["nsl_shift", "cic_zd"]:
        for seed in seeds:
            s = scenario(name, seed)
            y, z = s["y"], s["P"].argmax(1)
            T = len(y)
            Pca8, P8 = quantized(s["Pca"], 8), quantized(s["P"], 8)
            for a in [0.01, 0.05]:
                # static baselines (8-bit scores, as transmitted)
                for meth, mond in METHODS[:3]:
                    cp = C.SplitConformal(meth, mond, a, seed).fit(
                        Pca8, s["yca"])
                    M = cp.sets(P8)
                    m = C.set_metrics(M, y, z)
                    rows.append(dict(scenario=name, seed=seed, aseed=-1,
                                     alpha=a, method="static-" + meth +
                                     ("-M" if mond else ""), q=0,
                                     gamma=0, delay=0, T=T,
                                     miscov=1 - m["coverage"],
                                     mean_size=m["mean_size"],
                                     singleton=m["singleton"],
                                     singleton_err=m["singleton_err"],
                                     top1_acc=m["top1_acc"]))
                for meth in ["lac", "aps"]:
                    rng_u = np.random.default_rng(seed + 991)
                    U = rng_u.random(len(y)) if meth == "aps" else None
                    Uc = rng_u.random(len(s["yca"])) if meth == "aps" else None
                    S = C.score_matrix(P8, meth, U)
                    Sca = C.score_matrix(Pca8, meth, Uc)[
                        np.arange(len(s["yca"])), s["yca"]]
                    for ase in audit_seeds:
                        rng = np.random.default_rng(1000 * seed + ase)
                        Uaud = rng.random(T)
                        for q in qs:
                            A = Uaud < q
                            for g in gammas:
                                for d in delays:
                                    if meth == "aps" and (g != 0.01 or d != 100):
                                        continue
                                    err, size, al = C.aci_stream(
                                        S, y, Sca, a, g, A, delay=d)
                                    D = max(pending_max(A, d), 1)
                                    be, bh = bound_thm3(a, a, g, q, T, D)
                                    sing = size == 1
                                    rows.append(dict(
                                        scenario=name, seed=seed, aseed=ase,
                                        alpha=a, method="aci-" + meth, q=q,
                                        gamma=g, delay=d, T=T,
                                        miscov=float(err.mean()),
                                        mean_size=float(size.mean()),
                                        singleton=float(sing.mean()),
                                        singleton_err=float((sing & err).mean()),
                                        top1_acc=float((z == y).mean()),
                                        n_audit=int(A.sum()), D=D,
                                        bound_exp=be, bound_hp=bh,
                                        alpha_min=float(al.min()),
                                        alpha_max=float(al.max())))
                                    if (traces and seed == 0 and ase == 0
                                            and meth == "lac" and g == 0.01
                                            and d == 100 and q in (0.01, 0.05)):
                                        k = np.arange(1, T + 1)
                                        tr = pd.DataFrame(dict(
                                            t=k, run_miscov=np.cumsum(err) / k,
                                            size=size, alpha_t=al))
                                        tr.iloc[::max(1, T // 2000)].to_csv(
                                            RES / f"e3_trace_{name}_a{a}_q{q}.csv",
                                            index=False)
                if traces and seed == 0:
                    cp = C.SplitConformal("lac", False, a, seed).fit(
                        Pca8, s["yca"])
                    err = ~cp.sets(P8)[np.arange(T), y]
                    k = np.arange(1, T + 1)
                    pd.DataFrame(dict(t=k, run_miscov=np.cumsum(err) / k)
                                 ).iloc[::max(1, T // 2000)].to_csv(
                        RES / f"e3_trace_{name}_a{a}_static.csv", index=False)
            print(name, seed, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "e3_shift.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--aseeds", type=int, default=5)
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    if a.part in ("all", "e2"):
        e2(seeds)
    if a.part in ("all", "e3"):
        e3(seeds, list(range(a.aseeds)))
