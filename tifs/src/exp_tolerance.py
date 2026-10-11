"""
E9  Intrusion tolerance: a compromised authorized sender in score mode.

The zero-day CIC-IDS-2017 stream is produced by n_s = 5 senders with equal
shares.  Sender 0 is compromised and colludes with the intruder: for every
attack flow it reports, it signs the confident certificate (255, 0), i.e.
"benign".  The receiver audits Bernoulli(q) claims; every opening is lost on
the link independently with probability eps_o.

Insider strategies
  answer    the insider answers every challenge; recomputation of the
            approved model on the opened evidence flags the first audited
            falsified claim (bit-exact recomputation, see E5)
  withhold  the insider withholds the openings of its falsified claims,
            so they time out like link losses

Receiver policies
  static          score-mode sets with the calibrated threshold (gamma = 0)
  aci             audit-driven ACI with imputation, no key action
  static+revoke   static sets, key revoked / sender quarantined
  aci+revoke      ACI with imputation, key revoked / sender quarantined

Revocation follows a flag (answer) or a timeout test (withhold): after
n_a audits of a sender with x timeouts, the sender is quarantined when
x > n_a * eps_bar + sqrt(n_a * ln(n_a (n_a + 1) / delta) / 2), an anytime
Hoeffding test whose false-quarantine probability is at most delta for a
sender whose openings are lost independently with probability <= eps_bar.
Key actions take effect L claims (of the merged stream) after the trigger;
the sender's later claims are rejected by G1.

A control run without corruption measures false quarantines.

Output: results/e9_tolerance.csv, results/e9_trace.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import conformal as C
from common import RES, scenario

NS = 5
EPS_O = 0.01        # link loss of audit openings
EPS_BAR = 0.02      # loss level the quarantine test tolerates
DELTA = 1e-3
LAT = 100           # claims between trigger and effective key action
DELAY = 100         # label delay (claims)
GAMMA = 0.01


def q8(P):
    return C.dequantize(C.quantize(P))


def quarantine_time(audit_idx, timeout):
    """Anytime Hoeffding test on one sender's audit sequence; returns the
    stream index of the audit that triggers quarantine, or None."""
    n = np.arange(1, len(audit_idx) + 1)
    x = np.cumsum(timeout)
    thr = n * EPS_BAR + np.sqrt(n * np.log(n * (n + 1) / DELTA) / 2)
    hit = np.where(x > thr)[0]
    return int(audit_idx[hit[0]]) if len(hit) else None


def run(S, y, Sca, a, gamma, A, lost, keep):
    k = np.where(keep)[0]
    err, size, _ = C.aci_stream(S[k], y[k], Sca, a, gamma, A[k],
                                delay=DELAY, lost=lost[k], impute=True)
    return k, err, size


def e9(seeds=range(5), aseeds=range(5)):
    rows, trace = [], []
    for seed in seeds:
        s = scenario("cic_zd", seed)
        y = s["y"]
        T = len(y)
        P = q8(s["P"])
        Pca = q8(s["Pca"])
        Sca = 1 - Pca[np.arange(len(s["yca"])), s["yca"]]
        for ase in aseeds:
            r = np.random.default_rng(900 + 10 * seed + ase)
            sender = r.integers(0, NS, T)
            own = sender == 0
            fals = own & (y == 1)
            Pf = P.copy()
            Pf[fals] = np.array([1.0, 0.0])
            u_loss = r.random(T)
            u_aud = r.random(T)
            for q in [0.01, 0.05]:
                A = u_aud < q
                linklost = A & (u_loss < EPS_O)
                for strat in ["none", "answer", "withhold"]:
                    S = 1 - (P if strat == "none" else Pf)
                    bad = fals if strat != "none" else np.zeros(T, bool)
                    lost = linklost | (A & bad if strat == "withhold" else False)
                    # key actions per sender
                    t_act = {}
                    for snd in range(NS):
                        ai = np.where(A & (sender == snd))[0]
                        if strat == "answer" and snd == 0:
                            f = ai[bad[ai] & ~linklost[ai]]
                            trig = int(f[0]) if len(f) else None
                        else:
                            trig = quarantine_time(ai, lost[ai])
                        if trig is not None:
                            t_act[snd] = trig + LAT
                    keep_rev = np.ones(T, bool)
                    for snd, ta in t_act.items():
                        keep_rev &= ~((sender == snd) & (np.arange(T) > ta))
                    for a in [0.01, 0.05]:
                        for pol, gamma, keep in [
                                ("static", 0.0, np.ones(T, bool)),
                                ("aci", GAMMA, np.ones(T, bool)),
                                ("static+revoke", 0.0, keep_rev),
                                ("aci+revoke", GAMMA, keep_rev)]:
                            k, err, size = run(S, y, Sca, a, gamma, A, lost,
                                               keep)
                            sing = size == 1
                            fk = bad[k]
                            rows.append(dict(
                                seed=seed, aseed=ase, q=q, strategy=strat,
                                alpha=a, policy=pol, T_acc=len(k),
                                miscov=float(err.mean()),
                                mean_size=float(size.mean()),
                                singleton=float(sing.mean()),
                                singleton_err=float((sing & err).mean()),
                                fals_acc=int(fk.sum()),
                                fals_acted=int((fk & sing & err).sum()),
                                honest_miscov=float(err[~fk].mean()),
                                t_revoke=(t_act.get(0, np.nan)
                                          if "revoke" in pol else np.nan),
                                fals_before_revoke=(
                                    int((bad & (np.arange(T)
                                                <= t_act.get(0, T))).sum())
                                    if "revoke" in pol else np.nan),
                                false_quar=int(sum(1 for snd in t_act
                                                   if snd != 0
                                                   or strat == "none"))))
                            if seed == 0 and ase == 0 and q == 0.01 \
                                    and a == 0.01 and strat != "none":
                                w = 2000
                                c = np.convolve(err, np.ones(w) / w, "valid")
                                z = np.convolve(size, np.ones(w) / w, "valid")
                                idx = k[w - 1:]
                                for j in range(0, len(c), 200):
                                    trace.append(dict(strategy=strat,
                                                      policy=pol, t=int(idx[j]),
                                                      miscov=c[j], size=z[j]))
            print("seed", seed, "aseed", ase, "done", flush=True)
    pd.DataFrame(rows).to_csv(RES / "e9_tolerance.csv", index=False)
    pd.DataFrame(trace).to_csv(RES / "e9_trace.csv", index=False)


if __name__ == "__main__":
    e9()
