"""
E9  Intrusion tolerance: a compromised authorized sender in score mode.

The day-ordered zero-day CIC-IDS-2017 stream is produced by n_s = 5 senders
with equal shares.  Sender 0 is compromised and colludes with the intruder:
for the attack flows it reports, it signs the certificate (255, 0), i.e.
confidently "benign".  Four quantities are kept apart:

  targeted    attack flow of sender 0 whose report the insider overwrites
  falsified   targeted claim whose signed certificate is discrepant from the
              approved model's 8-bit output on the committed evidence
              (pcsp2.discrepant: > 1 LSB or a different arg max); a targeted
              claim on which the model already outputs (255, 0) or (254, 1)
              is a model error, not a falsification, and cannot be flagged
  flag        a valid opening of a falsified claim reaches the receiver
  wrong act   accepted singleton set that misses the true label

Event timeline (claim index of the merged accepted stream):
  t            claim accepted, set formed, action taken, audit coin drawn
  t + D_OPEN   opening arrives (unless lost on the link, prob. EPS_O, or
               withheld); a falsified opening raises a flag here
  t + DEADLINE label returns, or the audit times out; ACI applies the label
               or the conservative imputation 1{alpha_u > 0} (FIFO)
  trigger + L  key action takes effect, L counted in the sender's own
               claims; later claims of the sender are rejected by G1, and
               its audits still pending then resolve as timeouts

Insider strategies
  none          honest control (sender 0 reports the model output)
  answer        answers every challenge
  withhold      withholds the openings of its falsified claims
  withhold-low  falsifies only a fraction RHO_LOW of the attack flows it
                reports and withholds their openings, so that its timeout
                rate stays below the quarantine test's null allowance

Quarantine: after n_a resolved audits of a sender with x timeouts, quarantine
when x / n_a > eps_bar and n_a kl(x / n_a, eps_bar) > ln(n_a (n_a + 1) /
delta), an anytime Chernoff (relative-entropy) test evaluated at each
deadline; its false-quarantine probability is at most delta for a sender
whose openings are lost independently with probability <= eps_bar.

Output: results/e9_tolerance.csv, results/e9_trace.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import conformal as C
from common import RES, scenario

NS = 5
EPS_O = 0.01        # opening loss on the link (challenge or response)
EPS_BAR = 0.02      # loss level the quarantine test attributes to the link
DELTA = 1e-3
D_OPEN = 20         # opening round trip (merged claims)
DEADLINE = 100      # label return / timeout deadline (merged claims)
L_OWN = 20          # key-action latency (claims of the affected sender)
GAMMA = 0.01
RHO_LOW = 0.02
WIN = 2000          # trailing window of the trace


def discrepant(Qrep, Qmod):
    """Vectorized pcsp2.discrepant on uint8 score matrices."""
    d = np.abs(Qrep.astype(np.int16) - Qmod.astype(np.int16)).max(1)
    return (d > 1) | (Qrep.argmax(1) != Qmod.argmax(1))


def kl_bern(p, e):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return p * np.log(p / e) + (1 - p) * np.log((1 - p) / (1 - e))


def quarantine_hit(timeout):
    """Index of the first resolved audit at which the anytime Chernoff test
    rejects 'timeouts are i.i.d. with rate <= EPS_BAR', or None."""
    n = np.arange(1, len(timeout) + 1)
    x = np.cumsum(timeout)
    ph = x / n
    stat = np.where(ph > EPS_BAR, n * kl_bern(ph, EPS_BAR), 0.0)
    h = np.where(stat > np.log(n * (n + 1) / DELTA))[0]
    return int(h[0]) if len(h) else None


def run_stream(S, y, Sca, a, gamma, sender, A, arrive_ok, fals, key_on,
               strategy):
    """Simulate one receiver policy causally.  Returns per-claim arrays of
    the accepted stream and the key-action record."""
    T = len(y)
    t_idx = np.arange(T)
    # trigger times per sender (merged index), computed causally
    trig = {}
    if key_on:
        for snd in range(NS):
            mine = sender == snd
            aud = np.where(A & mine)[0]
            if snd == 0 and strategy == "answer":
                hit = aud[fals[aud] & arrive_ok[aud]]
                if len(hit):
                    trig[snd] = ("flag", int(hit[0]) + D_OPEN)
                    continue
            h = quarantine_hit(~arrive_ok[aud])
            if h is not None:
                trig[snd] = ("quarantine", int(aud[h]) + DEADLINE)
    # effective time: L_OWN further claims of the sender after the trigger
    t_eff = {}
    for snd, (_, tt) in trig.items():
        own_after = np.where((sender == snd) & (t_idx > tt))[0]
        t_eff[snd] = int(own_after[L_OWN - 1]) if len(own_after) >= L_OWN \
            else T
    keep = np.ones(T, bool)
    for snd, te in t_eff.items():
        keep &= ~((sender == snd) & (t_idx > te))
    k = np.where(keep)[0]
    # feedback of the accepted claims: label if the opening arrived before
    # its deadline and before the sender's key action, else a timeout
    arrived = arrive_ok.copy()
    for snd, te in t_eff.items():
        arrived &= ~((sender == snd) & (t_idx + D_OPEN > te))
    fb = np.where(arrived, C.FB_LABEL, C.FB_TIMEOUT).astype(np.int8)
    pos = np.full(T, -1)
    pos[k] = np.arange(len(k))
    # resolution at merged index t + DEADLINE, mapped to the accepted index
    res_m = np.minimum(t_idx[k] + DEADLINE, T - 1)
    res = np.searchsorted(k, res_m, side="left")
    err, size, al, D = C.aci_sched(S[k], y[k], Sca, a, gamma, A[k], res,
                                   fb[k])
    return k, err, size, trig, t_eff, D


def e9(seeds=range(5), aseeds=range(5)):
    rows, trace = [], []
    for seed in seeds:
        s = scenario("cic_zd", seed)
        y = s["y"]
        T = len(y)
        Qm = C.quantize(s["P"])                      # approved-model output
        Qca = C.quantize(s["Pca"])
        Sca = 1 - C.dequantize(Qca)[np.arange(len(s["yca"])), s["yca"]]
        lie = np.array([255, 0], np.uint8)
        for ase in aseeds:
            r = np.random.default_rng(900 + 10 * seed + ase)
            sender = r.integers(0, NS, T)
            u_loss, u_aud, u_low = r.random(T), r.random(T), r.random(T)
            own_atk = (sender == 0) & (y == 1)
            for strat in ["none", "answer", "withhold", "withhold-low"]:
                tgt = own_atk.copy() if strat != "none" else np.zeros(T, bool)
                if strat == "withhold-low":
                    tgt &= u_low < RHO_LOW
                Qr = Qm.copy()
                Qr[tgt] = lie
                fals = tgt & discrepant(Qr, Qm)
                S = 1 - C.dequantize(Qr)
                for q in [0.01, 0.05]:
                    A = u_aud < q
                    arrive_ok = ~(u_loss < EPS_O)
                    if strat.startswith("withhold"):
                        arrive_ok &= ~fals
                    for a in [0.01, 0.05]:
                        for pol, gamma, key in [("static", 0.0, False),
                                                ("aci", GAMMA, False),
                                                ("static+key", 0.0, True),
                                                ("aci+key", GAMMA, True)]:
                            k, err, size, trig, t_eff, D = run_stream(
                                S, y, Sca, a, gamma, sender, A, arrive_ok,
                                fals, key, strat)
                            sing = size == 1
                            fk, tk = fals[k], tgt[k]
                            honest = sender[k] != 0
                            n_hon = int((sender != 0).sum())
                            tr0 = trig.get(0, (None, np.nan))
                            rows.append(dict(
                                seed=seed, aseed=ase, q=q, strategy=strat,
                                alpha=a, policy=pol, T_acc=len(k), D=D,
                                miscov=float(err.mean()),
                                mean_size=float(size.mean()),
                                singleton=float(sing.mean()),
                                wrong_act=float((sing & err).mean()),
                                targeted=int(tgt.sum()),
                                falsified=int(fals.sum()),
                                model_consistent_targeted=int(
                                    (tgt & ~fals).sum()),
                                fals_acc=int(fk.sum()),
                                fals_acted=int((fk & sing & err).sum()),
                                tgt_acted=int((tk & sing & err).sum()),
                                key_kind=tr0[0] or "",
                                t_trigger=tr0[1],
                                t_effective=t_eff.get(0, np.nan),
                                false_flags=int(sum(
                                    1 for snd, (kind, _) in trig.items()
                                    if kind == "flag" and (snd != 0 or not
                                                           fals.any()))),
                                false_quar=int(sum(
                                    1 for snd, (kind, _) in trig.items()
                                    if kind == "quarantine" and (
                                        snd != 0 or not fals.any()))),
                                honest_avail=float(honest.sum() / n_hon),
                                honest_miscov=float(err[honest].mean())))
                            if seed == 0 and ase == 0 and q == 0.01 \
                                    and a == 0.01 and strat in (
                                        "answer", "withhold") \
                                    and pol in ("static", "aci", "aci+key"):
                                c = np.convolve(err, np.ones(WIN) / WIN,
                                                "valid")
                                idx = k[WIN - 1:]
                                for j in range(0, len(c), 200):
                                    trace.append(dict(
                                        strategy=strat, policy=pol,
                                        t=int(idx[j]), miscov=c[j],
                                        t_eff=t_eff.get(0, np.nan)))
            print("seed", seed, "aseed", ase, "done", flush=True)
    pd.DataFrame(rows).to_csv(RES / "e9_tolerance.csv", index=False)
    pd.DataFrame(trace).to_csv(RES / "e9_trace.csv", index=False)


if __name__ == "__main__":
    e9()
