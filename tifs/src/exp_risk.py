"""
E3d  Decision risk beyond marginal miscoverage, baselines, and labeling.

For the shifted streams (cic_zd, nsl_shift), alpha in {0.01, 0.05},
q in {0.01, 0.05}, gamma = 0.01, label delay 100, five model seeds x five
audit seeds (the audit coins of exp_conformal.e3, so the ACI rows reproduce
Table IV exactly), per run:
  wrong_act    wrong autonomous actions / accepted claims
  autonomous   singleton fraction
  sel_err      wrong autonomous actions / autonomous actions
  false_benign attack claims acted on as benign / attack claims
  false_alarm  benign claims acted on as an attack / benign claims
  worst_win    largest miscoverage over trailing windows of 2000 claims
Baselines on the same streams and label budget:
  static       static LAC certificates (8-bit scores)
  conf-thr     act iff the top 8-bit probability >= tau, tau calibrated on
               the calibration set for selective error <= alpha
  random       act on a random subset of the size ACI acts on (per run),
               using the top-1 label
  iw-aci       intermittent (importance-weighted) ACI, step gamma / q on
               audited claims (Zhao et al.; Wang et al.), same audits
Labeling service (cic_zd, alpha = 0.01, q = 0.01): the audited label is
flipped independently with probability eta, or is the approved model's own
top-1 output (a labeler that repeats the model's errors).

Masking labeler (masking(), same streams and audits): for a fraction b of
the attack claims the labeling service returns the model's top-1 output
instead of the true label, so that an audited miscovered attack claim is
reported as covered; the realized masking rate eta_T of Theorem 3 and the
expectation bound with that term are recorded.

Output: results/e3d_risk.csv, results/e3d_labels.csv, results/e3d_masking.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import conformal as C
from common import RES, scenario

W = 2000


def q8(P):
    return C.dequantize(C.quantize(P))


def metrics(err, size, y, top, K):
    sing = size == 1
    wa = sing & err
    atk = y != 0
    c = np.convolve(err.astype(float), np.ones(W) / W, "valid")
    return dict(miscov=float(err.mean()), mean_size=float(size.mean()),
                autonomous=float(sing.mean()), wrong_act=float(wa.mean()),
                sel_err=float(wa.sum() / max(1, sing.sum())),
                false_benign=float((sing & (top == 0) & atk).sum()
                                   / max(1, atk.sum())),
                false_alarm=float((sing & (top != 0) & ~atk).sum()
                                  / max(1, (~atk).sum())),
                worst_win=float(c.max()))


def conf_threshold(Pca, yca, a):
    """Smallest tau whose calibration selective error is <= a."""
    top = Pca.max(1)
    ok = Pca.argmax(1) == yca
    o = np.argsort(-top)
    cum_err = np.cumsum(~ok[o]) / np.arange(1, len(o) + 1)
    good = np.where(cum_err <= a)[0]
    return top[o][good[-1]] if len(good) else np.inf


def e3d(seeds=range(5), aseeds=range(5)):
    rows, lab = [], []
    for name in ["cic_zd", "nsl_shift"]:
        for seed in seeds:
            s = scenario(name, seed)
            y, K = s["y"], s["K"]
            T = len(y)
            Pca8, P8 = q8(s["Pca"]), q8(s["P"])
            top = P8.argmax(1)
            S = 1 - P8
            Sca = S_ca = (1 - Pca8)[np.arange(len(s["yca"])), s["yca"]]
            for a in [0.01, 0.05]:
                cp = C.SplitConformal("lac", False, a, seed).fit(Pca8,
                                                                  s["yca"])
                M = cp.sets(P8)
                rows.append(dict(scenario=name, seed=seed, aseed=-1, alpha=a,
                                 q=0, method="static",
                                 **metrics(~M[np.arange(T), y], M.sum(1), y,
                                           top, K)))
                tau = conf_threshold(Pca8, s["yca"], a)
                act = P8.max(1) >= tau
                size = np.where(act, 1, K)
                err = act & (top != y)
                rows.append(dict(scenario=name, seed=seed, aseed=-1, alpha=a,
                                 q=0, method="conf-thr", tau=tau,
                                 **metrics(err, size, y, top, K)))
                for ase in aseeds:
                    rng = np.random.default_rng(1000 * seed + ase)
                    Uaud = rng.random(T)
                    rr = np.random.default_rng(77 + 1000 * seed + ase)
                    for q in [0.01, 0.05]:
                        A = Uaud < q
                        err, size, _ = C.aci_stream(S, y, S_ca, a, 0.01, A,
                                                    delay=100)
                        m = metrics(err, size, y, top, K)
                        rows.append(dict(scenario=name, seed=seed, aseed=ase,
                                         alpha=a, q=q, method="aci", **m))
                        act = rr.random(T) < m["autonomous"]
                        rows.append(dict(scenario=name, seed=seed, aseed=ase,
                                         alpha=a, q=q, method="random",
                                         **metrics(act & (top != y),
                                                   np.where(act, 1, K), y,
                                                   top, K)))
                        e2, s2, _ = C.aci_stream(S, y, S_ca, a, 0.01 / q, A,
                                                 delay=100)
                        rows.append(dict(scenario=name, seed=seed, aseed=ase,
                                         alpha=a, q=q, method="iw-aci",
                                         **metrics(e2, s2, y, top, K)))
                        if name == "cic_zd" and a == 0.01 and q == 0.01:
                            res = np.arange(T) + 100
                            fb = np.zeros(T, np.int8)
                            for eta in [0.0, 0.01, 0.02, 0.05]:
                                flip = rr.random(T) < eta
                                yf = np.where(flip, 1 - y, y)
                                e3, s3, _, _ = C.aci_sched(S, y, S_ca, a, 0.01,
                                                           A, res, fb, y_fb=yf)
                                lab.append(dict(seed=seed, aseed=ase,
                                                labeler=f"flip {eta}",
                                                eta=eta, **metrics(
                                                    e3, s3, y, top, K)))
                            e3, s3, _, _ = C.aci_sched(S, y, S_ca, a, 0.01, A,
                                                       res, fb, y_fb=top)
                            lab.append(dict(seed=seed, aseed=ase,
                                            labeler="model top-1",
                                            eta=float((top != y).mean()),
                                            **metrics(e3, s3, y, top, K)))
            print(name, seed, "done", flush=True)
    pd.DataFrame(rows).to_csv(RES / "e3d_risk.csv", index=False)
    pd.DataFrame(lab).to_csv(RES / "e3d_labels.csv", index=False)


def masking(seeds=range(5), aseeds=range(5), a=0.01, q=0.01):
    rows = []
    s0 = None
    for seed in seeds:
        s = scenario("cic_zd", seed)
        y, K = s["y"], s["K"]
        T = len(y)
        Pca8, P8 = q8(s["Pca"]), q8(s["P"])
        top = P8.argmax(1)
        S = 1 - P8
        S_ca = (1 - Pca8)[np.arange(len(s["yca"])), s["yca"]]
        for ase in aseeds:
            rng = np.random.default_rng(1000 * seed + ase)
            A = rng.random(T) < q                     # the audits of e3d
            um = np.random.default_rng(5000 + 1000 * seed + ase).random(T)
            res = np.arange(T) + 100
            fb = np.zeros(T, np.int8)
            for b in [0.0, 0.1, 0.25, 0.5, 1.0]:
                yf = np.where((y != 0) & (um < b), top, y)
                err, size, _, D = C.aci_sched(S, y, S_ca, a, 0.01, A, res, fb,
                                              y_fb=yf)
                # label error that masks a miscoverage: Y not in C, Y~ in C
                # (binary sets: C = {top} whenever Y is missed)
                eta = float((err & (yf == top)).mean())
                bound = a + (a / 0.01 + 2 * D * (1 - a)) / (q * T) + eta
                rows.append(dict(seed=seed, aseed=ase, budget=b, eta=eta,
                                 bound=bound, D=D,
                                 **metrics(err, size, y, top, K)))
        print("masking", seed, flush=True)
    pd.DataFrame(rows).to_csv(RES / "e3d_masking.csv", index=False)


def traces(seed=0, ase=0, a=0.01):
    """Cumulative and trailing-window miscoverage (seed 0, audit seed 0),
    static vs ACI with q in {0.01, 0.05}; window 2000 (CIC) or 1000
    (NSL-KDD) claims, trailing so that each point is causal."""
    for name, w in [("cic_zd", 2000), ("nsl_shift", 1000)]:
        s = scenario(name, seed)
        y = s["y"]
        T = len(y)
        Pca8, P8 = q8(s["Pca"]), q8(s["P"])
        S = 1 - P8
        S_ca = (1 - Pca8)[np.arange(len(s["yca"])), s["yca"]]
        cp = C.SplitConformal("lac", False, a, seed).fit(Pca8, s["yca"])
        out = {"static": ~cp.sets(P8)[np.arange(T), y]}
        rng = np.random.default_rng(1000 * seed + ase)
        U = rng.random(T)
        for q in [0.01, 0.05]:
            out[f"aci{q}"] = C.aci_stream(S, y, S_ca, a, 0.01, U < q,
                                          delay=100)[0]
        k = np.arange(1, T + 1)
        df = {"t": k}
        for key, e in out.items():
            e = e.astype(float)
            df[f"cum_{key}"] = np.cumsum(e) / k
            df[f"win_{key}"] = pd.Series(e).rolling(w, min_periods=w)\
                .mean().to_numpy()
        step = max(1, T // 1500)
        pd.DataFrame(df).iloc[::step].to_csv(RES / f"e3d_trace_{name}.csv",
                                            index=False)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:] or ["e3d", "traces", "masking"]:
        globals()[p]()
