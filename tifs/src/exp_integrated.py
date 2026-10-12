"""
E10  Integrated decision-level performability (Definition 2), simulated
causally on one timeline instead of composed from separate experiments.

Stream: day-ordered zero-day CIC-IDS-2017 (binary), one sender, alpha = 0.01;
        every design and batch size evaluates the same first T0 = 196,608
        claims (divisible by every N in the grid), so that all compared
        configurations see identical observations.
Link:   ML-DSA-65, mu = 200-B frame payload plus 8-B framing, frame erasures
        i.i.d. (eps = 0.1) or Gilbert-Elliott (stationary 0.1, mean burst 8
        frames); the channel state runs continuously through batch frames,
        opening frames, and retransmissions.
Modes:  static receiver in set mode (3-B claims, no audits);
        audit-driven ACI in score mode (4-B claims, K = 2), q = 0.01.
Per batch:
  1. the batch's frames are sent (MONO, FIV+FEC, or FIV+FEC-IL layout for
     the actual claim size); the receiver accepts what verifies;
  2. the receiver draws an audit coin for every accepted claim and sends an
     authenticated 26-B challenge (pcsp2.make_challenge: slot, round,
     missing-fragment bitmap, MAC) on the reverse link (i.i.d. loss eps);
  3. the sender answers from its cached opening (pcsp2.OpeningServer) with
     the requested fragments, each a 9-B fragment header plus at most
     mu - 9 bytes of the framed opening (10-B header, 312-B evidence, 16-B
     salt, log2 N path hashes); after a retransmission timer of BACKOFF
     frame times, during which the channel carries other traffic, the
     receiver sends the next round with the bitmap of the fragments still
     missing (selective repeat), for at most R repair rounds after the
     first (R + 1 rounds in total); repair="whole" instead requests the
     whole opening in every round; an opening still incomplete after the
     last round times out;
  4. labels and timeouts of the batch's audits resolve DELAY accepted
     claims after the batch's last accepted claim (FIFO), so claims acted
     on early in a batch wait for the whole batch.
The timeouts of the (honest) sender are also fed to the anytime KL
quarantine test (eps_bar = 0.02, delta = 1e-3) to measure its false
quarantines on each channel.
Operating point: N (and the parity r) is selected per design and receiver
on three validation traces (model and stream seeds 5, 6, 7, each with its
own channel seed), disjoint from the evaluation seeds: an ACI candidate is
feasible iff its mean miscoverage over the validation traces is at most
alpha and at least MIN_AUDITS audits completed on every trace; among
feasible candidates the one with the largest mean Gamma is chosen.  (On a
single trace the miscoverage of ACI fluctuates by about +-0.003 around
alpha, so requiring every trace to meet alpha selects among N <= 1024
almost at random; the largest validation miscoverage is reported.)  If none is feasible, the
largest-Gamma candidate is evaluated and marked infeasible.  The static
receiver takes the largest-Gamma N.  Evaluation: five model seeds x five
fresh channel seeds.

Output: results/e10_integrated.csv, results/e10_tuning.csv,
        results/e10_sweep.csv
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numba import njit

import conformal as C
import pcsp2 as P
from common import RES, scenario
from exp_bandwidth import layout, simulate, expected_verified, FRAME_HDR
from sigs import SIG_BYTES

MU = 200
EPS = 0.1
BURST = 8
ALPHA = 0.01
GAMMA = 0.01
DELAY = 100
B_X = 312
B_O = P.batch_overhead(SIG_BYTES["ML-DSA-65"])
NGRID = [64, 128, 256, 512, 1024, 2048, 4096]
T0 = 196_608                 # common horizon, divisible by every N in NGRID
FRAME = MU + FRAME_HDR
BACKOFF = 2 * BURST
VSEEDS = (5, 6, 7)           # validation model/stream seeds (not evaluated)
MIN_AUDITS = 100             # completed audits required per validation trace
EPS_BAR, DELTA_Q = 0.02, 1e-3


@njit(cache=True)
def _ge(n, eps, burst, u):
    p_bg = 1.0 / burst
    p_gb = eps * p_bg / (1.0 - eps)
    lost = np.empty(n, np.bool_)
    bad = u[0] < eps
    for i in range(n):
        lost[i] = bad
        if bad:
            bad = u[i + 1] >= p_bg
        else:
            bad = u[i + 1] < p_gb
    return lost


class Link:
    """Forward-link frame erasures, consumed sequentially."""

    def __init__(self, rng, bursty, n=8_000_000):
        u = rng.random(n + 1)
        self.lost = _ge(n, EPS, BURST, u) if bursty else u[:n] < EPS
        self.i = 0

    def send(self, k):
        out = self.lost[self.i:self.i + k]
        self.i += k
        return out

    def wait(self, k):
        self.i += k


def header_parity(N, b, design, rng):
    """Parity frames: FIV+FEC sized analytically for 0.99 header success
    under i.i.d. loss; FIV+FEC-IL sized by 2000 Monte-Carlo tuning draws for
    0.99 under Gilbert-Elliott loss with interleaving."""
    if design == "MONO":
        return 0
    Fh = layout(N, b, B_O, MU, "FIV")[0]
    if design == "FIV+FEC":
        r = 0
        while True:
            ev, _ = expected_verified(N, b, B_O, MU, EPS, "FIV+FEC", r)
            if ev / (N * (1 - EPS)) >= 0.99:
                return r
            r += 1
    r = 0
    while r < 6 * Fh:
        _, _, ph = simulate(N, b, B_O, MU, EPS, "FIV+FEC-IL", r, rng,
                            ge_burst=BURST, trials=2000, interleave=True,
                            return_hdr=True)
        if ph >= 0.99:
            return r
        r += 1
    return r


def batch_frames(N, b, design, r):
    """Frame plan of one batch: (n_frames, header-frame mask, claim frame
    index -> (first claim, claims per frame))."""
    if design == "MONO":
        F = layout(N, b, B_O, MU, "MONO")[0]
        return F, None, None, None
    Fh, Fc, Pk, _ = layout(N, b, B_O, MU, "FIV")
    nh = Fh + r
    nf = nh + Fc
    if design == "FIV+FEC-IL":
        hpos = np.unique(np.round(np.linspace(0, nf - 1, nh)).astype(int))
    else:
        hpos = np.arange(nh)
    hm = np.zeros(nf, bool)
    hm[hpos] = True
    return nf, hm, Fh, Pk


def quarantine_index(timeout):
    """First resolved audit at which the anytime KL test of Section IV
    rejects 'timeouts independent with rate <= EPS_BAR', or -1."""
    if len(timeout) == 0:
        return -1
    n = np.arange(1, len(timeout) + 1)
    ph = np.cumsum(timeout) / n
    p = np.clip(ph, 1e-12, 1 - 1e-12)
    kl = p * np.log(p / EPS_BAR) + (1 - p) * np.log((1 - p) / (1 - EPS_BAR))
    stat = np.where(ph > EPS_BAR, n * kl, 0.0)
    h = np.where(stat > np.log(n * (n + 1) / DELTA_Q))[0]
    return int(h[0]) if len(h) else -1


def run(s, design, N, r, policy, R, bursty, cseed, impute="level",
        repair="selective"):
    y = s["y"][:T0]
    T = T0
    Q = C.quantize(s["P"][:T])
    Qca = C.quantize(s["Pca"])
    Sca = 1 - C.dequantize(Qca)[np.arange(len(s["yca"])), s["yca"]]
    S = 1 - C.dequantize(Q)
    aci = policy == "aci"
    b = 4 if aci else 3
    q = 0.01 if aci else 0.0
    rng = np.random.default_rng(cseed)
    link = Link(rng, bursty)
    nf, hm, Fh, Pk = batch_frames(N, b, design, r)
    a_len = P.OPEN_HDR + B_X + P.SALT_LEN + P.HASH_LEN * int(math.log2(N))
    fo = P.n_opening_fragments(a_len, MU)
    fwd = rev = 0
    acc_idx, audit, res_time, fb = [], [], [], []
    n_acc = 0
    opens = lost_open = 0
    for bi in range(T // N):
        lost = link.send(nf)
        fwd += nf * FRAME
        if design == "MONO":
            got = np.arange(N) if not lost.any() else np.empty(0, int)
        else:
            if (~lost[hm]).sum() < Fh:
                got = np.empty(0, int)
            else:
                cf = np.where(~lost[~hm])[0]
                got = (cf[:, None] * Pk + np.arange(Pk)[None, :]).ravel()
        g = bi * N + got
        A = rng.random(len(g)) < q
        n_new = len(g)
        acc_idx.append(g)
        audit.append(A)
        end = n_acc + n_new - 1
        n_acc += n_new
        na = int(A.sum())
        ok = np.zeros(na, bool)
        for u in range(na):
            # round 0 requests the opening; each later round re-requests
            # the fragments still missing (selective) or all (whole)
            need = np.ones(fo, bool)
            done = False
            for rnd in range(R + 1):
                if rnd:
                    link.wait(BACKOFF)
                rev += P.AUDIT_CHALLENGE_BYTES + FRAME_HDR
                if rng.random() < EPS:
                    continue                    # challenge lost
                if repair == "whole":
                    need[:] = True
                k = int(need.sum())
                l = link.send(k)
                fwd += k * FRAME
                idx = np.where(need)[0]
                need[idx[~l]] = False
                if repair == "whole" and l.any():
                    need[:] = True              # incomplete copy discarded
                    continue
                if not need.any():
                    done = True
                    break
            ok[u] = done
        opens += na
        lost_open += int((~ok).sum())
        res_time.append(np.full(na, end + DELAY))
        f = np.where(ok, C.FB_LABEL, C.FB_TIMEOUT if impute == "level"
                     else C.FB_TIMEOUT_SET)
        fb.append(f.astype(np.int8))
    k = np.concatenate(acc_idx)
    A = np.concatenate(audit)
    res = np.zeros(len(k), np.int64)
    res[A] = np.concatenate(res_time) if len(res_time) else []
    fbv = np.zeros(len(k), np.int8)
    fbv[A] = np.concatenate(fb) if len(fb) else []
    if aci:
        err, size, al, D = C.aci_sched(S[k], y[k], Sca, ALPHA, GAMMA, A, res,
                                       fbv)
    else:
        m = len(Sca)
        kq = math.ceil((m + 1) * (1 - ALPHA))
        qh = np.sort(Sca)[kq - 1] if kq <= m else np.inf
        Sk = S[k]
        M = Sk <= qh
        M[np.arange(len(k)), Sk.argmin(1)] = True
        err = ~M[np.arange(len(k)), y[k]]
        size = M.sum(1)
        D = 0
    sing = size == 1
    correct = int((sing & ~err).sum())
    wrong = int((sing & err).sum())
    tmo = (fbv[A] != C.FB_LABEL) if aci else np.zeros(0, bool)
    return dict(design=design, N=N, r=r, policy=policy, R=R, impute=impute,
                repair=repair, channel="bursty" if bursty else "iid",
                T_emit=T, fo=fo,
                accepted=len(k), audits=opens, open_lost=lost_open,
                completed=opens - lost_open,
                quar_at=quarantine_index(tmo),
                correct=correct, wrong=wrong, escal=int((~sing).sum()),
                miscov=float(err.mean()) if len(k) else np.nan,
                sel_err=wrong / max(1, int(sing.sum())), D=D,
                fwd_bytes=fwd, rev_bytes=rev,
                gamma_fwd=correct / fwd * 1000, haz_fwd=wrong / fwd * 1000,
                gamma_bi=correct / (fwd + rev) * 1000,
                haz_bi=wrong / (fwd + rev) * 1000)


CONFIGS = [  # (policy, R, impute, repair)
    ("static", 0, "level", "selective"),
    ("aci", 0, "level", "selective"),
    ("aci", 3, "level", "selective"),
    ("aci", 3, "level", "whole"),
]
DESIGNS = ["MONO", "FIV+FEC", "FIV+FEC-IL"]


def tune():
    """Every candidate N on the three validation traces."""
    rows = []
    prng = np.random.default_rng(4242)
    par = {}
    for v in VSEEDS:
        s = scenario("cic_zd", v)
        for bursty in [False, True]:
            for design in DESIGNS:
                for pol, R, imp, rep in CONFIGS:
                    b = 4 if pol == "aci" else 3
                    for N in NGRID:
                        if (N, b, design) not in par:
                            par[(N, b, design)] = header_parity(N, b, design,
                                                                prng)
                        row = run(s, design, N, par[(N, b, design)], pol, R,
                                  bursty, 1000 + v, imp, rep)
                        row.update(vseed=v)
                        rows.append(row)
            print("tuned seed", v, bursty, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(RES / "e10_tuning.csv", index=False)
    return t


KEY = ["channel", "design", "policy", "R", "impute", "repair"]


def select(t):
    """Locked operating point per configuration (see module docstring)."""
    g = t.groupby(KEY + ["N", "r"]).agg(
        gamma=("gamma_fwd", "mean"), mis_max=("miscov", "max"),
        mis_mean=("miscov", "mean"), comp_min=("completed", "min"),
        acc_min=("accepted", "min")).reset_index()
    g["feasible"] = (g.policy == "static") | (
        (g.mis_mean <= ALPHA) & (g.comp_min >= MIN_AUDITS))
    best = (g.sort_values(["feasible", "gamma"], ascending=False)
            .groupby(KEY).head(1))
    return best


def evaluate(t, seeds=range(5), cseeds=range(5)):
    best = select(t)
    best.to_csv(RES / "e10_selected.csv", index=False)
    rows = []
    for seed in seeds:
        s = scenario("cic_zd", seed)
        for _, c in best.iterrows():
            for cs in cseeds:
                row = run(s, c.design, int(c.N), int(c.r), c.policy, int(c.R),
                          c.channel == "bursty", 10 * seed + cs, c.impute,
                          c.repair)
                row.update(seed=seed, cseed=cs, feasible=bool(c.feasible))
                rows.append(row)
        print("eval seed", seed, flush=True)
    pd.DataFrame(rows).to_csv(RES / "e10_integrated.csv", index=False)


def sweep(seeds=range(5)):
    """Gamma and H against N for the figure (fresh channel seeds)."""
    rows = []
    prng = np.random.default_rng(4343)
    par = {}
    for seed in seeds:
        s = scenario("cic_zd", seed)
        for bursty in [False, True]:
            for design in ["MONO", "FIV+FEC-IL"]:
                for pol, R, imp, rep in CONFIGS[:3]:
                    b = 4 if pol == "aci" else 3
                    for N in NGRID:
                        if (N, b, design) not in par:
                            par[(N, b, design)] = header_parity(N, b, design,
                                                                prng)
                        row = run(s, design, N, par[(N, b, design)], pol, R,
                                  bursty, 2000 + seed, imp, rep)
                        row.update(seed=seed)
                        rows.append(row)
        print("sweep seed", seed, flush=True)
    pd.DataFrame(rows).to_csv(RES / "e10_sweep.csv", index=False)


if __name__ == "__main__":
    import sys
    if "sweep" in sys.argv:
        sweep()
    elif "evaluate" in sys.argv:
        evaluate(pd.read_csv(RES / "e10_tuning.csv"))
    else:
        evaluate(tune())
