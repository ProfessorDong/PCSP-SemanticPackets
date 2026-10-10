"""
Adaptive adversaries against the semantic layer.

E3b  Evasion: from time T0 the intruder perturbs its own attack flows in
     feature space (gray-box, query access) until the deployed classifier
     outputs `benign`.  Static certificates vs audit-driven ACI.
E4   Selective forwarding: an on-path oracle adversary that knows which
     emitted claims are unsafe drops fragments to concentrate unsafe claims
     among those delivered.  We report the unsafe fraction among accepted
     claims and the high-probability bound of Corollary 1 (n_emit is the
     authenticated claim counter; here every batch's header is delivered,
     so n_emit = T).
E5   Accountability: a compromised authorized signer falsifies a fraction
     beta of claims; the receiver audits Bernoulli(q) claims, recomputes
     the approved model on the opened evidence on a different device, and
     flags any mismatch (falsification rate phi in the paper).  Honest false-alarm rate and detection-time
     distribution vs Proposition 2.

Outputs: results/e3b_evasion.csv, e3b_trace.csv, e4_selective.csv,
         e5_accountability.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

import conformal as C
from common import RES, scenario


def q8(P):
    return C.dequantize(C.quantize(P))


# ------------------------------------------------------------------ E3b

def evade(clf, X, Xben, rng, target=0, conf=0.9, grid=None):
    """Interpolate each row toward a random benign training sample until
    p_target >= conf; returns perturbed rows and the step used."""
    grid = grid if grid is not None else np.linspace(0.05, 1.0, 20)
    Xp = X.copy()
    lam = np.full(len(X), np.nan)
    todo = np.arange(len(X))
    anchors = Xben[rng.integers(0, len(Xben), len(X))]
    for g in grid:
        if len(todo) == 0:
            break
        cand = X[todo] + g * (anchors[todo] - X[todo])
        p = clf.predict_proba(cand)[:, target]
        hit = p >= conf
        Xp[todo[hit]] = cand[hit]
        lam[todo[hit]] = g
        todo = todo[~hit]
    return Xp, lam


def e3b(seeds=range(5), aseeds=range(5)):
    rows = []
    for seed in seeds:
        s = scenario("cic_iid", seed)
        clf, y, X = s["clf"], s["y"], s["X"].copy()
        T = len(y)
        rng = np.random.default_rng(seed)
        order = rng.permutation(T)               # deployment order
        X, y = X[order], y[order]
        T0 = T // 3
        atk = np.where((np.arange(T) >= T0) & (y > 0))[0]
        Xben = s["Xtr"][s["ytr"] == 0]
        Xadv = X.copy()
        Xadv[atk], lam = evade(clf, X[atk], Xben, rng)
        P = q8(np.asarray(clf.predict_proba(Xadv), float))
        Pca = q8(s["Pca"])
        z = P.argmax(1)
        ev_rate = float(np.mean(~np.isnan(lam)))
        for a in [0.01, 0.05]:
            cp = C.SplitConformal("lac", False, a, seed).fit(Pca, s["yca"])
            M = cp.sets(P)
            err_s = ~M[np.arange(T), y]
            size_s = M.sum(1)
            rows.append(dict(seed=seed, aseed=-1, alpha=a, method="static",
                             q=0, miscov=err_s.mean(),
                             miscov_post=err_s[T0:].mean(),
                             size_post=size_s[T0:].mean(),
                             singleton_err_post=((size_s == 1) & err_s)[T0:].mean(),
                             evaded=ev_rate, n_attack_post=len(atk)))
            Sca = 1 - Pca[np.arange(len(s["yca"])), s["yca"]]
            S = 1 - P
            for ase in aseeds:
                ra = np.random.default_rng(100 + 10 * seed + ase)
                U = ra.random(T)
                for q in [0.005, 0.01, 0.05]:
                    err, size, al = C.aci_stream(S, y, Sca, a, 0.01, U < q,
                                                 delay=100)
                    rows.append(dict(seed=seed, aseed=ase, alpha=a,
                                     method="aci", q=q, miscov=err.mean(),
                                     miscov_post=err[T0:].mean(),
                                     size_post=size[T0:].mean(),
                                     singleton_err_post=((size == 1) & err)[T0:].mean(),
                                     evaded=ev_rate, n_attack_post=len(atk)))
                    if seed == 0 and ase == 0 and q == 0.01 and a == 0.01:
                        w = 2000
                        k = np.arange(T)
                        roll = lambda v: pd.Series(v.astype(float)).rolling(
                            w, center=True, min_periods=w // 2).mean().to_numpy()
                        pd.DataFrame(dict(
                            t=k, static=roll(err_s), aci=roll(err),
                            size_aci=roll(size), size_static=roll(size_s),
                            alpha_t=al)).iloc[::200].to_csv(
                            RES / "e3b_trace.csv", index=False)
        print("e3b seed", seed, "evaded", ev_rate, flush=True)
    pd.DataFrame(rows).to_csv(RES / "e3b_evasion.csv", index=False)


# ------------------------------------------------------------------ E4

def e4(seeds=range(5), delta1=0.025, delta2=0.025):
    """Selective forwarding by an oracle adversary that knows which emitted
    claims are unsafe (an upper bound on an intruder that knows its own
    attack flows), against (A) static certificates on exchangeable
    data (cic_iid; Corollary 1 applies) and (B) static certificates on the
    zero-day stream (hypotheses violated), plus receiver-side ACI on the
    accepted zero-day stream."""
    import math
    rows = []
    Pf = 8                                        # claims per fragment
    for seed in seeds:
        for sc in ["cic_iid", "cic_zd"]:
            s = scenario(sc, seed)
            y = s["y"]
            T = len(y) - len(y) % Pf
            y = y[:T]
            P = q8(s["P"][:T])
            Pca = q8(s["Pca"])
            m = len(s["yca"])
            Sca = 1 - Pca[np.arange(m), s["yca"]]
            for a in [0.01, 0.05]:
                cp = C.SplitConformal("lac", False, a, seed).fit(Pca, s["yca"])
                err = ~cp.sets(P)[np.arange(T), y]
                unsafe = err.reshape(-1, Pf)
                n_frag = len(unsafe)
                ra = np.random.default_rng(seed)
                score = unsafe.sum(1) + 1e-3 * ra.random(n_frag)
                rank = np.argsort(score)
                for drop in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
                    keep = np.ones(n_frag, bool)
                    keep[rank[:int(drop * n_frag)]] = False
                    n_acc = int(keep.sum()) * Pf
                    n_un = int(unsafe[keep].sum())
                    # union over every prefix length n <= n_max = T, so the
                    # bound holds even if the adversary truncates the session
                    hp = (T * (a + math.sqrt(math.log(1 / delta1) / (2 * m)))
                          + math.sqrt(T * math.log(T / delta2) / 2)) / n_acc
                    row = dict(seed=seed, scenario=sc, alpha=a, drop=drop,
                               n_emit=T, n_acc=n_acc, unsafe_acc=n_un,
                               cond_unsafe=n_un / n_acc,
                               joint_unsafe=n_un / T,
                               emitted_miscov=float(err.mean()),
                               bound_exp=min(1.0, a * T / n_acc),
                               bound_hp=min(1.0, hp))
                    if sc == "cic_zd":
                        kc = np.repeat(keep, Pf)
                        rb = np.random.default_rng(seed + 77)
                        e2, _, _ = C.aci_stream(
                            (1 - P)[kc], y[kc], Sca, a, 0.01,
                            rb.random(int(kc.sum())) < 0.05, delay=100)
                        row["aci_on_accepted"] = float(e2.mean())
                    rows.append(row)
    pd.DataFrame(rows).to_csv(RES / "e4_selective.csv", index=False)


def e3c(seeds=range(5), aseeds=range(5)):
    """Audit-opening suppression: the network adversary drops the audit
    openings of its own attack flows with probability rho.  Receiver
    either skips lost audits or imputes them (Theorem 3 with imputation)."""
    rows = []
    for seed in seeds:
        s = scenario("cic_zd", seed)
        y = s["y"]
        T = len(y)
        P = q8(s["P"])
        Pca = q8(s["Pca"])
        Sca = 1 - Pca[np.arange(len(s["yca"])), s["yca"]]
        for a in [0.01, 0.05]:
            for ase in aseeds:
                r = np.random.default_rng(500 + 10 * seed + ase)
                A = r.random(T) < 0.05
                u = r.random(T)
                for rho in [0.0, 0.5, 0.9, 1.0]:
                    lost = A & (y == 1) & (u < rho)
                    for imp in [False, True]:
                        err, size, _ = C.aci_stream(1 - P, y, Sca, a, 0.01, A,
                                                    delay=100, lost=lost,
                                                    impute=imp)
                        rows.append(dict(seed=seed, aseed=ase, alpha=a, rho=rho,
                                         policy="impute" if imp else "skip",
                                         miscov=float(err.mean()),
                                         mean_size=float(size.mean()),
                                         singleton=float((size == 1).mean()),
                                         lost_frac=float(lost.sum() / A.sum())))
    pd.DataFrame(rows).to_csv(RES / "e3c_lost_openings.csv", index=False)


# ------------------------------------------------------------------ E5

def e5(seed=0, trials=2000):
    s = scenario("cic_iid", seed)
    clf_gpu, X = s["clf"], s["X"]
    T = len(X)
    # transmitter: GPU inference; receiver: CPU inference of the same model
    q_tx = C.quantize(np.asarray(clf_gpu.predict_proba(X), float))
    cpu = xgb.XGBClassifier()
    cpu.load_model(bytearray(clf_gpu.get_booster().save_raw("json")))
    cpu.set_params(device="cpu")
    q_rx = C.quantize(np.asarray(cpu.predict_proba(X), float))
    diff = np.abs(q_tx.astype(int) - q_rx.astype(int))
    honest_exact = float((diff.max(1) == 0).mean())
    honest_tol1 = float((diff.max(1) <= 1).mean())
    argmax_agree = float((q_tx.argmax(1) == q_rx.argmax(1)).mean())
    rows = [dict(kind="determinism", exact=honest_exact, tol1=honest_tol1,
                 argmax_agree=argmax_agree, max_lsb=int(diff.max()))]
    # falsification: chosen attack claims re-labelled 'benign' with
    # a confident certificate
    # honest false alarms: the receiver's check on unmodified claims
    d0 = np.abs(q_tx.astype(int) - q_rx.astype(int)).max(1)
    fa = float(((d0 > 1) | (q_tx.argmax(1) != q_rx.argmax(1))).mean())
    rows[0]["false_alarm"] = fa
    # falsification: the compromised signer relabels attack flows as
    # 'benign' with a confident certificate; beta = fraction of all
    # claims that are falsified (only attack claims can be falsified)
    rng = np.random.default_rng(seed)
    forged = np.zeros(q_tx.shape[1], np.uint8)
    forged[0] = 255
    is_atk = q_tx.argmax(1) > 0
    p_atk = float(is_atk.mean())
    for beta in [0.001, 0.01, 0.05]:
        for q in [0.005, 0.01, 0.05]:
            first, eff = [], []
            for _ in range(trials):
                n = 20000
                idx = rng.integers(0, T, n)
                lie = is_atk[idx] & (rng.random(n) < beta / p_atk)
                eff.append(lie.mean())
                aud = rng.random(n) < q
                claimed = q_tx[idx].copy()
                claimed[lie] = forged
                d = np.abs(claimed.astype(int) - q_rx[idx].astype(int)).max(1)
                bad = (d > 1) | (claimed.argmax(1) != q_rx[idx].argmax(1))
                hit = np.where(aud & bad)[0]
                first.append(hit[0] + 1 if len(hit) else np.inf)
            first, b_eff = np.array(first), float(np.mean(eff))
            for n_c in [100, 500, 1000, 2000, 5000, 10000, 20000]:
                rows.append(dict(kind="detect", beta=beta, beta_eff=b_eff,
                                 q=q, n=n_c,
                                 p_detect=float((first <= n_c).mean()),
                                 theory=1 - (1 - q * b_eff) ** n_c))
    pd.DataFrame(rows).to_csv(RES / "e5_accountability.csv", index=False)


if __name__ == "__main__":
    import sys
    parts = sys.argv[1:] or ["e5", "e4", "e3b", "e3c"]
    for p in parts:
        globals()[p]()
