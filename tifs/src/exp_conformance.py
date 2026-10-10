"""
E1: attack conformance matrix.  Every attack is replayed against six
receiver designs; each cell is the fraction of attacked batches the
receiver accepts (deterministic gates, so cells are 0 or 1 up to the
negligible forgery/collision probability).  The `gate` column records
which predicate rejects the attack in the full design.

Signatures: liboqs ML-DSA-65 (FIPS 204).
"""

from __future__ import annotations

import copy
import hashlib
import os
import random
from dataclasses import replace

import pandas as pd

os.environ.setdefault("OQS_INSTALL_PATH", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".venv", "oqs"))

import pcsp2 as P
import sigs
from common import RES

N = 125            # odd, so the duplicate-leaf mutation applies to v1 trees
TRIALS = 200
TAU, W = 2, 64


# ---------------------------------------------------------------- v1 tree

def v1_root(leaves):
    """Naive tree (the conference prototype): duplicates the last node on
    odd levels, no leaf/node domain separation, leaves not bound to their
    position, and N not signed."""
    lv = [hashlib.sha256(x).digest() for x in leaves]
    while len(lv) > 1:
        if len(lv) % 2:
            lv.append(lv[-1])
        lv = [hashlib.sha256(lv[i] + lv[i + 1]).digest()
              for i in range(0, len(lv), 2)]
    return lv[0]


class V1Receiver(P.Receiver):
    """Same gates as v2, but the signed message is
    root_v1(claim bodies) || H(theta) || H(D) || sid || seq || t."""

    def verify(self, batch, t_now):
        h = batch.hdr
        b = self.bundles.get(h.bundle)
        if b is None:
            return False, P.G_BUNDLE
        pk = self.keys.get((h.sid, h.kid))
        msg = v1_msg(batch, b)
        if pk is None or not self.sig.verify(pk, msg, batch.sigma):
            return False, P.G_SIG
        if abs(t_now - h.t) > self.policy.tau:
            return False, P.G_FRESH
        w = self._window(h.sid)
        if not w.check(h.seq):
            return False, P.G_REPLAY
        w.commit(h.seq)
        return True, "ok"


def v1_msg(batch, b):
    h = batch.hdr
    return (v1_root([c.body() for c in batch.claims]) + b.H_theta + b.H_D
            + h.sid.to_bytes(2, "big") + h.seq.to_bytes(4, "big")
            + h.t.to_bytes(4, "big"))


def v1_sign(tx, batch):
    batch.sigma = tx.sig.sign(tx.sk, v1_msg(batch, tx.bundle))
    return batch


# ---------------------------------------------------------------- setup

def world(sig):
    sk, pk = sig.keygen()
    sk2, pk2 = sig.keygen()                    # second honest sender
    good = P.Bundle(P.H(b"theta-approved"), P.H(b"D-approved"),
                    P.MODE_SET, 5, 10)
    rogue = P.Bundle(P.H(b"theta-rogue"), P.H(b"D-rogue"), P.MODE_SET, 5, 10)
    tx = P.Transmitter(sig, sk, 1, 0, 0, good, os.urandom(32))
    tx2 = P.Transmitter(sig, sk2, 2, 0, 0, good, os.urandom(32))
    keys = {(1, 0): pk, (2, 0): pk2}
    return tx, tx2, keys, good, rogue


def claims(rng):
    cl = []
    for _ in range(N):
        z = rng.randrange(5)
        cl.append(P.Claim(z, rng.randrange(4), P.setmask({z, rng.randrange(5)}, 5)))
    ev = [rng.randbytes(64) for _ in range(N)]
    return cl, ev


DESIGNS = {
    "PCSP": dict(policy=P.ReceiverPolicy(tau=TAU)),
    "-replay": dict(policy=P.ReceiverPolicy(tau=TAU, check_replay=False)),
    "-fresh": dict(policy=P.ReceiverPolicy(tau=TAU, check_fresh=False)),
    "-sig": dict(policy=P.ReceiverPolicy(tau=TAU, check_sig=False)),
    "-bundle": dict(policy=P.ReceiverPolicy(tau=TAU, check_bundle=False)),
    "v1-tree": dict(policy=P.ReceiverPolicy(tau=TAU), v1=True),
}


def run_trial(design, attack, sig, rng):
    tx, tx2, keys, good, rogue = world(sig) if design not in _W else _W[design]
    cfg = DESIGNS[design]
    v1 = cfg.get("v1", False)
    # bundle 1 has the same hashes as bundle 0 but a different alpha
    looser = P.Bundle(good.H_theta, good.H_D, P.MODE_SET, 5, 50)
    R = (V1Receiver if v1 else P.Receiver)(sig, keys, {0: good, 1: looser},
                                            copy.deepcopy(cfg["policy"]), W)

    def enc(t, who=tx):
        cl, ev = claims(rng)
        b = who.encode(cl, ev, t)
        b.explicit = (good.H_theta, good.H_D)
        return v1_sign(who, b) if v1 else b

    def resign(b, who=tx, bundle=None):
        """Recompute the signature after a header/claim change, for
        attacks performed by key holders."""
        bd = bundle or who.bundle
        if v1:
            b.sigma = who.sig.sign(who.sk, v1_msg(b, bd))
        else:
            R_claim = P.mth([P.claim_leaf_input(i, c)
                             for i, c in enumerate(b.claims)])
            b.sigma = who.sig.sign(who.sk, P.signed_message(
                b.hdr, bd.H_theta, bd.H_D, R_claim))
        return b

    # warm-up: deliver 3 honest batches at t = 100, 101, 102
    hist = []
    for t in (100, 101, 102):
        b = enc(t)
        assert R.verify(b, t)[0], (design, "honest batch rejected")
        hist.append(b)
    t_now = 103
    gate = "-"
    if attack == "honest":
        x = enc(t_now)
    elif attack == "A1_replay_old":           # long after: out of window
        old = hist[0]
        for t in range(103, 103 + 70):           # advance seq beyond W
            R.verify(enc(t), t)
        t_now = 173
        x = old
    elif attack == "A1b_replay_inwin":
        x = hist[-1]
    elif attack == "A1c_withhold":             # never delivered, released late
        x = enc(t_now)
        t_now += TAU + 3
    elif attack == "A2_rogue_key":
        x = enc(t_now)
        ask, _ = sig.keygen()
        x = copy.copy(x)
        x.sigma = sig.sign(ask, (v1_msg(x, good) if v1 else
                                 P.signed_message(x.hdr, good.H_theta, good.H_D,
                                                  P.mth([P.claim_leaf_input(i, c) for i, c in enumerate(x.claims)]))))
    elif attack == "A2b_insider_rogue_model":   # authorized key, unapproved bundle
        x = enc(t_now)
        x.hdr = replace(x.hdr, bundle=7)
        x.explicit = (rogue.H_theta, rogue.H_D)
        x = resign(x, bundle=rogue)
    elif attack == "A2c_bundle_swap":         # on-path rewrite of beta/alpha
        x = enc(t_now)
        x.hdr = replace(x.hdr, bundle=1, alpha_code=50)
    elif attack == "A3_claim_tamper":
        x = enc(t_now)
        c = x.claims[5]
        x.claims[5] = P.Claim((c.z + 1) % 5, c.ctx, P.setmask({(c.z + 1) % 5}, 5))
    elif attack == "A4_epoch_rewrite":
        x = enc(t_now - 30)                    # stale batch, epoch rewritten
        x.hdr = replace(x.hdr, t=t_now)
    elif attack == "A5_dup_leaf":
        x = enc(t_now)
        x.claims = x.claims + [x.claims[-1]]
        if not v1:
            x.hdr = replace(x.hdr, N=N + 1)
    elif attack == "A5b_truncate":
        x = enc(t_now)
        x.claims = x.claims[:-5]
        x.hdr = replace(x.hdr, N=N - 5)
    elif attack == "A6_cross_sender":
        tx2.seq = tx.seq                       # a sequence number sid 1 has not used
        x = enc(t_now, who=tx2)
        x.hdr = replace(x.hdr, sid=1)
    else:
        raise ValueError(attack)
    ok, why = R.verify(x, t_now)
    return ok, why


_W = {}

ATTACKS = ["honest", "A1_replay_old", "A1b_replay_inwin", "A1c_withhold",
           "A2_rogue_key", "A2b_insider_rogue_model", "A2c_bundle_swap",
           "A3_claim_tamper",
           "A4_epoch_rewrite", "A5_dup_leaf", "A5b_truncate",
           "A6_cross_sender"]


def audit_trials(sig, rng, n=TRIALS):
    """A3b: evidence substitution at audit time, against the v2 opening."""
    tx, _, keys, good, _ = world(sig)
    R = P.Receiver(sig, keys, {0: good})
    caught = 0
    for _ in range(n):
        cl, ev = claims(rng)
        b = tx.encode(cl, ev, 10)
        i = rng.randrange(N)
        x, s, path = tx.open(b, i)
        assert R.check_audit(b, i, x, s, path)
        fake = rng.randbytes(len(x))
        caught += not R.check_audit(b, i, fake, s, path)
    return caught / n


def main():
    sig = sigs.LibOQSBackend("ML-DSA-65")
    rng = random.Random(2026)
    for d in DESIGNS:
        _W[d] = world(sig)
    rows = []
    for a in ATTACKS:
        for d in DESIGNS:
            acc, whys = 0, {}
            for _ in range(TRIALS):
                ok, why = run_trial(d, a, sig, rng)
                acc += ok
                whys[why] = whys.get(why, 0) + 1
            rows.append(dict(attack=a, design=d, accept_rate=acc / TRIALS,
                             trials=TRIALS,
                             gate=max(whys, key=whys.get)))
            print(a, d, acc / TRIALS, whys, flush=True)
    a3b = audit_trials(sig, rng)
    rows.append(dict(attack="A3b_evidence_swap_at_audit", design="PCSP",
                     accept_rate=1 - a3b, trials=TRIALS, gate="audit"))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "e1_conformance.csv", index=False)
    print(df.pivot(index="attack", columns="design", values="accept_rate"))


if __name__ == "__main__":
    main()
