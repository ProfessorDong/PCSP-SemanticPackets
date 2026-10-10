"""
E6: byte accounting (serialized batches vs Proposition 3) and baselines.
E7: batch authentication over fragment-erasure links (i.i.d. and
    Gilbert-Elliott), comparing monolithic batches (MONO),
    fragment-independent verification (FIV) and FIV with an MDS-coded
    header (FIV+FEC).  Monte-Carlo against the closed forms of
    Proposition 4.

Outputs: results/e6_bytes.csv, e6_baselines.csv, e7_channel.csv,
         e7_nopt.csv
"""

from __future__ import annotations

import math
import os
import zlib

import numpy as np
import pandas as pd

import pcsp2 as P
from common import RES, scenario
from sigs import SIG_BYTES, StubBackend

SCHEMES = ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87", "SLH-DSA-SHA2-128s",
           "SLH-DSA-SHA2-192s", "Falcon-512"]
NS = [1, 2, 4, 8, 16, 32, 64, 100, 125, 128, 256, 512, 1024]


class FixedSizeStub(StubBackend):
    """Stub signer padded to a real scheme's signature length so that
    byte counts can be measured without signing cost."""

    def __init__(self, n):
        self.sig_bytes = n

    def sign(self, sk, msg):
        return super().sign(sk, msg).ljust(self.sig_bytes, b"\0")


# ------------------------------------------------------------------ E6

def e6():
    rows, base = [], []
    for name in ["nsl_iid", "cic_iid"]:
        s = scenario(name, 0)
        K = s["K"]
        P8 = s["P"]
        z = P8.argmax(1)
        ev = s["evid"]
        for scheme in SCHEMES:
            sb = SIG_BYTES[scheme]
            sig = FixedSizeStub(sb)
            sk, pk = sig.keygen()
            bd = P.Bundle(P.H(b"t"), P.H(b"d"))
            for mode in (P.MODE_SET, P.MODE_SCORE):
                tx = P.Transmitter(sig, sk, 1, 0, 0, bd, b"k" * 32, mode, K, 1)
                for N in NS:
                    cl = []
                    for i in range(N):
                        if mode == P.MODE_SET:
                            cert = P.setmask([z[i]], K)
                        else:
                            cert = bytes(np.rint(P8[i] * 255).astype(np.uint8))
                        cl.append(P.Claim(int(z[i]), 0, cert))
                    b = tx.encode(cl, ev[:N], t=1)
                    meas = len(b.to_bytes()) / N          # serialized bytes
                    pred = P.per_claim_bytes(N, mode, K, sb)
                    op = len(tx.open_bytes(b, N - 1))     # measured opening
                    op_pred = P.audit_response_bytes(len(ev[N - 1]), N)
                    rows.append(dict(scenario=name, scheme=scheme, sig_bytes=sb,
                                     mode="SET" if mode == 0 else "SCORE",
                                     N=N, measured=meas, predicted=pred,
                                     residual=meas - pred, opening=op,
                                     opening_bound=op_pred))
        # baselines: raw evidence and compressed raw evidence per batch
        rb = s["raw_bytes"]
        fb = s["X"].shape[1] * 4
        for N in [32, 128, 512]:
            comp = []
            for j in range(0, min(len(ev) - N + 1, 64 * N), N):   # full batches only
                chunk = b"".join(ev[j:j + N])
                comp.append(len(zlib.compress(chunk, 9)) / N)
            base.append(dict(scenario=name, N=N,
                             raw_mean=float(rb.mean()),
                             raw_median=float(np.median(rb)),
                             raw_p90=float(np.percentile(rb, 90)),
                             feat_bytes=fb,
                             evid_zlib_mean=float(np.mean(comp)),
                             evid_mean=float(np.mean([len(e) for e in ev]))))
    # zlib per-record size on a fine N grid and break-even against
    # unauthenticated compressed NSL-KDD records (Corollary 2(i))
    s = scenario("nsl_iid", 0)
    ev = s["evid"]
    Bo = P.batch_overhead(SIG_BYTES["ML-DSA-65"])
    grid = []
    for N in list(range(128, 513, 16)):
        z = float(np.mean([len(zlib.compress(b"".join(ev[j:j + N]), 9)) / N
                           for j in range(0, 64 * N, N) if j + N <= len(ev)]))
        for q in (0.0, 0.01):
            pc = P.per_claim_bytes(N, P.MODE_SET, 5, SIG_BYTES["ML-DSA-65"],
                                   q, float(np.mean([len(e) for e in ev])))
            grid.append(dict(N=N, q=q, zlib=z, pcsp=pc, pcsp_cheaper=pc < z))
    pd.DataFrame(grid).to_csv(RES / "e6_zlib_grid.csv", index=False)
    pd.DataFrame(rows).to_csv(RES / "e6_bytes.csv", index=False)
    pd.DataFrame(base).to_csv(RES / "e6_baselines.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(base)


# ------------------------------------------------------------------ E7

FRAME_HDR = 8          # link-layer framing per frame (bytes), all modes
FRAG_HDR = P.FRAG_HDR  # FIV fragment header: seq(4) + offset(2)


def layout(N, b, B_o, M, mode, r=0):
    """Return (F_h, frames for claims, claims per frame P, proof bytes).
    MONO: everything in one byte stream of L = N b + B_o bytes.
    FIV : header stream (B_o + 32 B R_claim) in F_h frames; claim frames
          carry P = 2^k claims plus (log2 N - k) sibling hashes."""
    if mode == "MONO":
        L = N * b + B_o
        return math.ceil(L / M), 0, N, 0
    # The header stream carries the 2^j subtree roots at depth j (j = 0 is
    # just R_claim); a claim frame carries an aligned run of P = 2^k claims
    # and the n - j - k sibling hashes up to its depth-j root.  (j, k) is
    # chosen to minimize the total frame count.
    n = int(math.log2(N))
    best = None
    for j in range(0, n + 1):
        Fh = math.ceil((B_o + P.HASH_LEN * (1 << j)) / M)
        for k in range(n - j, -1, -1):
            Pk = 1 << k
            pb = P.HASH_LEN * (n - j - k)
            if FRAG_HDR + pb + Pk * b <= M:
                tot = Fh + N // Pk
                if best is None or tot < best[0]:
                    best = (tot, Fh, N // Pk, Pk, pb)
                break
    if best is None:
        raise ValueError("MTU too small")
    return best[1:]


def expected_verified(N, b, B_o, M, eps, mode, r=0):
    """Closed form (Proposition 4) for i.i.d. frame erasures."""
    Fh, Fc, Pk, pb = layout(N, b, B_o, M, mode)
    if mode == "MONO":
        frames = Fh
        ok = (1 - eps) ** Fh
        return N * ok, frames
    if mode == "FIV":
        ph = (1 - eps) ** Fh
    else:                                  # FIV+FEC: any Fh of Fh + r
        n_ = Fh + r
        ph = sum(math.comb(n_, j) * (1 - eps) ** j * eps ** (n_ - j)
                 for j in range(Fh, n_ + 1))
    frames = Fh + r + Fc
    return N * ph * (1 - eps), frames


def ge_losses(nframes, eps, burst, rng):
    """Gilbert-Elliott: bad state drops every frame, good state none;
    mean burst length `burst`, stationary loss eps."""
    p_bg = 1 / burst
    p_gb = eps * p_bg / (1 - eps)
    lost = np.empty(nframes, bool)
    bad = rng.random() < eps
    for i in range(nframes):
        lost[i] = bad
        bad = (rng.random() >= p_bg) if bad else (rng.random() < p_gb)
    return lost


def simulate(N, b, B_o, M, eps, mode, r, rng, ge_burst=None, trials=400,
             interleave=False, return_hdr=False):
    """Monte-Carlo delivery.  With `interleave`, the Fh + r header frames
    are spread evenly over the batch's frame sequence instead of being
    sent first, so a burst hits fewer of them."""
    Fh, Fc, Pk, pb = layout(N, b, B_o, M, mode)
    nh = Fh + (r if mode.startswith("FIV+FEC") else 0)
    frames = nh + Fc if mode != "MONO" else Fh
    if interleave and mode != "MONO":
        hpos = np.unique(np.round(np.linspace(0, frames - 1, nh)).astype(int))
    else:
        hpos = np.arange(nh)
    hmask = np.zeros(frames, bool)
    hmask[hpos] = True
    got, hok = 0, 0
    for _ in range(trials):
        if ge_burst:
            lost = ge_losses(frames, eps, ge_burst, rng)
        else:
            lost = rng.random(frames) < eps
        if mode == "MONO":
            got += N * (not lost.any())
            hok += not lost.any()
            continue
        if mode.startswith("FIV+FEC"):
            hdr_ok = (~lost[hmask]).sum() >= Fh
        else:
            hdr_ok = not lost[hmask].any()
        hok += hdr_ok
        if hdr_ok:
            got += Pk * int((~lost[~hmask]).sum())
    if return_hdr:
        return got / trials, frames, hok / trials
    return got / trials, frames


def n_opt_mono(b, B_o, M, eps):
    """Proposition 4(i): continuous-relaxation maximizer of verified
    claims per transmitted byte for MONO batches."""
    kappa = -math.log(1 - eps) / M
    return B_o / (2 * b) * (math.sqrt(1 + 4 / (kappa * B_o)) - 1)


def e7():
    rng = np.random.default_rng(7)
    rows, nopt = [], []
    b = 3                                   # SET mode, K <= 8
    for scheme in ["ML-DSA-44", "ML-DSA-65", "SLH-DSA-SHA2-128s",
                   "Falcon-512"]:
        B_o = P.batch_overhead(SIG_BYTES[scheme])
        for M in [200, 1000]:
            for eps in [0.0, 0.01, 0.05, 0.1, 0.2]:
                for N in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
                    for mode, r in [("MONO", 0), ("FIV", 0),
                                    ("FIV+FEC", None)]:
                        if mode == "FIV+FEC":
                            Fh = layout(N, b, B_o, M, "FIV")[0]
                            # smallest r giving header success >= 0.99
                            r = 0
                            while r < 4 * Fh:
                                ev, _ = expected_verified(N, b, B_o, M, eps,
                                                          mode, r)
                                if ev / (N * (1 - eps) if eps < 1 else 1) >= 0.99 \
                                        or eps == 0:
                                    break
                                r += 1
                        ev, frames = expected_verified(N, b, B_o, M, eps,
                                                       mode, r)
                        on_air = frames * (M + FRAME_HDR)
                        mc, _ = simulate(N, b, B_o, M, eps, mode, r, rng,
                                         trials=300)
                        mc_ge, _ = (simulate(N, b, B_o, M, eps, mode, r, rng,
                                             ge_burst=8, trials=300)
                                    if eps > 0 else (mc, 0))
                        if eps > 0 and mode == "FIV+FEC":
                            # burst-aware variant: interleaved header frames,
                            # r chosen by Monte-Carlo for 0.99 header success
                            # under Gilbert-Elliott loss
                            rg = 0
                            while True:
                                _, _, ph = simulate(N, b, B_o, M, eps,
                                                    "FIV+FEC-IL", rg, rng,
                                                    ge_burst=8, trials=400,
                                                    interleave=True,
                                                    return_hdr=True)
                                if ph >= 0.99 or rg >= 6 * Fh:
                                    break
                                rg += 1
                            mg, fg = simulate(N, b, B_o, M, eps, "FIV+FEC-IL",
                                              rg, rng, ge_burst=8, trials=300,
                                              interleave=True)
                            mi, _ = simulate(N, b, B_o, M, eps, "FIV+FEC-IL",
                                             rg, rng, trials=300,
                                             interleave=True)
                            rows.append(dict(
                                scheme=scheme, M=M, eps=eps, N=N,
                                mode="FIV+FEC-IL", r=rg, frames=fg,
                                bytes_on_air=fg * (M + FRAME_HDR),
                                exp_verified=np.nan, mc_verified=mi,
                                mc_verified_ge=mg, eff_theory=np.nan,
                                eff_mc=mi / (fg * (M + FRAME_HDR)) * 1000,
                                eff_mc_ge=mg / (fg * (M + FRAME_HDR)) * 1000))
                        rows.append(dict(scheme=scheme, M=M, eps=eps, N=N,
                                         mode=mode, r=r, frames=frames,
                                         bytes_on_air=on_air,
                                         exp_verified=ev, mc_verified=mc,
                                         mc_verified_ge=mc_ge,
                                         eff_theory=ev / on_air * 1000,
                                         eff_mc=mc / on_air * 1000,
                                         eff_mc_ge=mc_ge / on_air * 1000))
                if eps > 0:
                    nopt.append(dict(scheme=scheme, M=M, eps=eps, B_o=B_o,
                                     N_opt_closed=n_opt_mono(b, B_o, M, eps)))
        print(scheme, "done", flush=True)
    # exact discrete optimum for MONO over a fine N grid
    for row in nopt:
        Ns = np.arange(1, 20001)
        L = Ns * b + row["B_o"]
        F = np.ceil(L / row["M"])
        eff = Ns * (1 - row["eps"]) ** F / (F * (row["M"] + FRAME_HDR))
        row["N_opt_exact"] = int(Ns[eff.argmax()])
        rel = n_opt_mono(b, row["B_o"], row["M"], row["eps"])
        Lr = rel * b + row["B_o"]
        row["eff_closed_N"] = float(np.interp(rel, Ns, eff))
        row["eff_exact_max"] = float(eff.max())
    pd.DataFrame(rows).to_csv(RES / "e7_channel.csv", index=False)
    pd.DataFrame(nopt).to_csv(RES / "e7_nopt.csv", index=False)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:] or ["e6", "e7"]:
        globals()[p]()
