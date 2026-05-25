"""
Run the PCSP attack-matrix experiments for MILCOM 2026 Paper 3.

Pipeline (single command, single CPU):

  1. Load NSL-KDD train/calib/test splits.
  2. Train an XGBoost flow classifier on the 41-feature, 5-class
     superclass problem.
  3. Calibrate APS conformal prediction on the held-out calibration
     split at target coverage 1-alpha = 0.90.
  4. Build PCSP batches of size N from the test set; sign each batch.
  5. For each attack in {A1 replay, A2 model substitution,
     A3 evidence mismatch, A4 stale context} corrupt fraction eta
     of the transmitted batches and measure the receiver's FAR.
  6. Report Table I cells (FAR per attack, bytes per claim,
     verification latency) to results/master.csv.

The signature backend can be `stub` (zero-cost placeholder), or
`mldsa` / `slhdsa` (real liboqs once available).

Author: Liang Dong, MILCOM 2026 Paper 3.
"""

from __future__ import annotations

import argparse, csv, hashlib, time
from pathlib import Path

import numpy as np
import xgboost as xgb

import nslkdd_loader
import cicids_loader
from pcsp import (PCSPClaim, encode_batch, verify_batch, packet_bytes,
                  StubBackend, DilithiumPyBackend, PySPXBackend,
                  LibOQSBackend, claim_leaf, merkle_root, _sha256)

# Standard PQC signature sizes from NIST FIPS 204 / FIPS 205.
# Used for analytical bandwidth predictions (Theorem 2 RHS) when the
# experiment runs under a stub backend.
PQC_SIG_BYTES = {
    "STUB":              16,
    "ML-DSA-65":         3309,   # FIPS 204 spec; dilithium_py produces exactly this
    "SLH-DSA-SHA2-128s": 7856,
}
PQC_PK_BYTES = {
    "STUB":              32,
    "ML-DSA-65":         1952,
    "SLH-DSA-SHA2-128s":  32,
}


# ---------------------------------------------------------------------------
# APS conformal calibration
# ---------------------------------------------------------------------------

def aps_calibrate(probs_cal: np.ndarray, y_cal: np.ndarray,
                  alpha: float = 0.10) -> float:
    """Adaptive Prediction Sets (Romano et al. 2020 / Angelopoulos &
    Bates 2021).  Returns the tau threshold for in-distribution
    coverage 1-alpha.

    For each calibration example we compute the cumulative sorted
    probability mass needed to include the true label; tau is the
    (1-alpha) quantile of those scores.
    """
    n = len(y_cal)
    scores = np.empty(n, dtype=np.float64)
    order = np.argsort(-probs_cal, axis=1)
    sorted_p = np.take_along_axis(probs_cal, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)
    for i in range(n):
        rank = int(np.where(order[i] == y_cal[i])[0][0])
        scores[i] = cumsum[i, rank]
    q = np.ceil((n + 1) * (1 - alpha)) / n
    q = min(q, 1.0)
    return float(np.quantile(scores, q, method="higher"))


def aps_predict_set(probs: np.ndarray, tau: float) -> list[tuple[int, ...]]:
    """Return per-row prediction set (tuple of class indices)."""
    sets = []
    for p in probs:
        order = np.argsort(-p)
        cs = np.cumsum(p[order])
        k = int(np.searchsorted(cs, tau)) + 1
        sets.append(tuple(sorted(order[:k].tolist())))
    return sets


# ---------------------------------------------------------------------------
# Attack module
# ---------------------------------------------------------------------------

def attack_replay(batches, eta, rng, min_age: int = 10):
    """A1 replay: replace eta fraction of batches with a copy of a
    batch at least `min_age` epochs older than the receiver's clock.
    Only attacks batches j>=min_age (early batches have no qualifying
    donor in the past)."""
    n = len(batches)
    n_atk = int(n * eta)
    candidates = np.arange(min_age, n)
    n_atk = min(n_atk, len(candidates))
    idx = rng.choice(candidates, size=n_atk, replace=False)
    out = list(batches)
    for j in idx:
        donor = max(0, j - min_age)
        out[j] = batches[donor]
    return out, set(int(i) for i in idx)


def attack_model_sub(batches, eta, rng, sig, approved_pk, attacker_sk):
    """A2 model substitution: attacker re-signs eta fraction with a
    DIFFERENT model hash under its own (unapproved) keypair."""
    n = len(batches)
    n_atk = int(n * eta)
    idx = rng.choice(n, size=n_atk, replace=False)
    out = list(batches)
    forged_H_theta = _sha256(b"forged-model-v0")
    for j in idx:
        b = batches[j]
        leaves = [claim_leaf(c) for c in b.claims]
        root = merkle_root(leaves)
        msg = root + forged_H_theta + b.H_D
        forged_sig = sig.sign(attacker_sk, msg)
        out[j] = type(b)(claims=b.claims, H_theta=forged_H_theta,
                          H_D=b.H_D, sigma=forged_sig, leaves=leaves)
    return out, set(idx.tolist())


def attack_evidence_mismatch(batches, eta, rng):
    """A3 evidence mismatch: keep the original signature but swap the
    first claim's H_x for a hash of unrelated data; the Merkle root
    changes so signature verification fails."""
    n = len(batches)
    n_atk = int(n * eta)
    idx = rng.choice(n, size=n_atk, replace=False)
    out = list(batches)
    for j in idx:
        b = batches[j]
        if not b.claims:
            continue
        tampered = list(b.claims)
        c0 = tampered[0]
        tampered[0] = PCSPClaim(z=c0.z, c_alpha=c0.c_alpha,
                                 t=c0.t, ctx=c0.ctx,
                                 H_x=_sha256(b"unrelated-evidence"))
        out[j] = type(b)(claims=tampered, H_theta=b.H_theta,
                          H_D=b.H_D, sigma=b.sigma)
    return out, set(idx.tolist())


def attack_stale_context(batches, eta, rng, drift_window: int = 50):
    """A4 stale context: shift the freshness epoch of eta fraction of
    batches into the past by `drift_window` cycles, simulating a
    decision being acted on past its useful operational context."""
    n = len(batches)
    n_atk = int(n * eta)
    idx = rng.choice(n, size=n_atk, replace=False)
    out = list(batches)
    for j in idx:
        b = batches[j]
        shifted = [PCSPClaim(z=c.z, c_alpha=c.c_alpha,
                              t=c.t - drift_window,
                              ctx=c.ctx, H_x=c.H_x)
                   for c in b.claims]
        leaves = [claim_leaf(c) for c in shifted]
        # Note: signature does NOT cover t (it's part of claim_leaf via Merkle)
        # so we re-sign properly only if the attacker has the key.
        # Here we keep the original signature; verify must catch via leaf mismatch.
        out[j] = type(b)(claims=shifted, H_theta=b.H_theta,
                          H_D=b.H_D, sigma=b.sigma, leaves=leaves)
    return out, set(idx.tolist())


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def make_backend(name: str):
    name = name.lower()
    if name == "stub":
        return StubBackend()
    if name in ("mldsa", "ml-dsa", "ml-dsa-65"):
        return DilithiumPyBackend()
    if name in ("slhdsa", "slh-dsa", "slh-dsa-sha2-128s"):
        return PySPXBackend()
    if name == "oqs-mldsa":
        return LibOQSBackend("ML-DSA-65")
    if name == "oqs-slhdsa":
        return LibOQSBackend("SLH-DSA-SHA2-128s")
    raise ValueError(f"unknown backend: {name}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--N", type=int, default=128, help="batch size")
    ap.add_argument("--eta", type=float, default=0.10,
                    help="attacker corruption fraction")
    ap.add_argument("--backend", default="stub",
                    choices=["stub", "mldsa", "slhdsa",
                              "oqs-mldsa", "oqs-slhdsa"])
    ap.add_argument("--freshness-window", type=int, default=2,
                    help="freshness slack in batch-epochs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="nslkdd",
                    choices=["nslkdd", "cicids"])
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.dataset == "nslkdd":
        print(f"[setup] loading NSL-KDD ...")
        Xtr, ytr, Xca, yca, Xte, yte = nslkdd_loader.load_train_test(
            seed=args.seed)
        CLASSES = nslkdd_loader.CLASSES
        B_x_bytes = Xtr.shape[1] * 4
    else:
        print(f"[setup] loading CIC-IDS-2017 ...")
        Xtr, ytr, Xca, yca, Xte, yte = cicids_loader.load_train_test(
            seed=args.seed)
        CLASSES = cicids_loader.CLASSES
        B_x_bytes = Xtr.shape[1] * 4
    print(f"[setup] train={Xtr.shape}  calib={Xca.shape}  test={Xte.shape}")
    print(f"[setup] raw flow size B_x = {B_x_bytes} bytes")

    print(f"[train] XGBoost 5-class flow classifier ...")
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective="multi:softprob", num_class=len(CLASSES),
        n_jobs=-1, eval_metric="mlogloss",
    )
    t0 = time.time()
    clf.fit(Xtr, ytr)
    print(f"[train] done in {time.time()-t0:.1f}s")
    train_acc = clf.score(Xtr, ytr)
    test_acc  = clf.score(Xte, yte)
    print(f"[train] accuracy  train={train_acc:.3f}  test={test_acc:.3f}")

    print(f"[calib] APS conformal at coverage {1-args.alpha:.2f} ...")
    probs_ca = clf.predict_proba(Xca)
    tau = aps_calibrate(probs_ca, yca, alpha=args.alpha)
    print(f"[calib] tau = {tau:.4f}")

    probs_te = clf.predict_proba(Xte)
    sets_te  = aps_predict_set(probs_te, tau)
    cov_emp  = float(np.mean([yte[i] in sets_te[i]
                                for i in range(len(yte))]))
    print(f"[calib] empirical coverage on test set = {cov_emp:.3f} "
          f"(target {1-args.alpha:.2f})")

    print(f"[pcsp] backend = {args.backend}")
    sig = make_backend(args.backend)
    sk, pk = sig.keygen()
    attacker_sig = make_backend(args.backend)
    atk_sk, _ = attacker_sig.keygen()
    H_theta = _sha256(clf.get_booster().save_raw())
    H_D     = _sha256(probs_ca.tobytes())
    approved = {H_theta}

    # Build batches from the test set.  One batch == one signed packet,
    # one freshness epoch shared by all N claims (matches real protocols
    # where a single signed message carries one timestamp).
    z_te = clf.predict(Xte)
    n_pkt = len(yte) // args.N
    batches = []
    for b in range(n_pkt):
        sl = slice(b * args.N, (b + 1) * args.N)
        t_batch = b      # one epoch per batch
        cs = []
        for k in range(args.N):
            i = sl.start + k
            cs.append(PCSPClaim(
                z=int(z_te[i]),
                c_alpha=tuple(int(c) for c in sets_te[i]),
                t=t_batch,
                ctx=int(_sha256(Xte[i].tobytes())[0] & 0x01),
                H_x=_sha256(Xte[i].tobytes()),
            ))
        batches.append(encode_batch(cs, H_theta, H_D, sig, sk))
    print(f"[pcsp] built {len(batches)} batches of N={args.N}")

    # Verification cost: timing one verify on an honest batch.
    t0 = time.time(); _ok = sig.verify(pk, b"x" * 64, sig.sign(sk, b"x" * 64))
    t_verify_ms = (time.time() - t0) * 1e3
    bytes_per_claim = packet_bytes(batches[0], sig.sig_bytes) / args.N

    # Run each attack and measure FAR for FIVE policies, isolating
    # what each PCSP gate buys:
    #   raw      : no signature, no checks (accept everything that arrives)
    #   semantic : signature absent; receiver only checks freshness
    #   signed   : signature checked, NO model-hash check, NO freshness check
    #   pcsp     : full PCSP -- signature + model hash + freshness window
    # The first two policies use the same packet structure but disable
    # the relevant gate at the receiver; the last two test full PCSP.
    results = []
    attacks = {
        "A1_replay": lambda B: attack_replay(B, args.eta, rng),
        "A2_model_sub": lambda B: attack_model_sub(B, args.eta, rng,
                                                    sig, pk, atk_sk),
        "A3_evidence_mismatch":
            lambda B: attack_evidence_mismatch(B, args.eta, rng),
        "A4_stale_context":
            lambda B: attack_stale_context(B, args.eta, rng,
                                            drift_window=args.freshness_window + 1),
    }
    policies = {
        # name : (check_signature?, check_model_hash?, check_freshness?)
        "raw":      (False, False, False),
        "semantic": (False, False, True),
        "signed":   (True,  False, False),
        "pcsp":     (True,  True,  True),
    }

    # Per-batch receiver epoch: receiver verifies batch j at its
    # arrival epoch (= j).  Honest batches pass freshness; an A1
    # replay arrives with t from a much older batch (j-100) so falls
    # outside the freshness window.
    def cur_epoch_for(j: int) -> int:
        return j

    def verify_policy(b, j, check_sig, check_model, check_fresh):
        # Always recompute Merkle leaves from the (possibly tampered) claims
        leaves = [claim_leaf(c) for c in b.claims]
        root   = merkle_root(leaves)
        msg    = root + b.H_theta + b.H_D
        sig_ok   = (not check_sig)   or sig.verify(pk, msg, b.sigma)
        model_ok = (not check_model) or (b.H_theta in approved)
        if not (sig_ok and model_ok):
            return [False] * len(b.claims)
        if check_fresh:
            ce = cur_epoch_for(j)
            return [abs(ce - c.t) <= args.freshness_window
                     for c in b.claims]
        return [True] * len(b.claims)

    for attack_name, atk in attacks.items():
        adv_batches, atk_idx = atk(batches)
        for policy_name, (chk_s, chk_m, chk_f) in policies.items():
            false_accepts = 0
            total_atk_claims = 0
            for j, b in enumerate(adv_batches):
                if j not in atk_idx:
                    continue
                accepted = verify_policy(b, j, chk_s, chk_m, chk_f)
                false_accepts += sum(1 for a in accepted if a)
                total_atk_claims += len(accepted)
            far = false_accepts / max(1, total_atk_claims)
            results.append({
                "attack": attack_name,
                "policy": policy_name,
                "backend": sig.name, "N": args.N, "eta": args.eta,
                "FAR": round(far, 4),
                "coverage": round(cov_emp, 4),
                "acc_test": round(test_acc, 4),
            })

    # Print a compact per-policy / per-attack table.
    print()
    print(f"{'attack':<22}" + "".join(f"{p:>10}"
                                       for p in policies.keys()))
    for atk_name in attacks.keys():
        row = [f"{atk_name:<22}"]
        for pol in policies.keys():
            r = [x for x in results if x["attack"] == atk_name
                                       and x["policy"] == pol][0]
            row.append(f"{r['FAR']:>10.4f}")
        print("".join(row))

    # Predicted bytes-per-claim under each candidate signature scheme
    # (Theorem 2 evaluated at the experiment's N).
    print()
    print("Analytical Theorem 2 bytes/claim at N =", args.N)
    # Per-claim shared (Merkle + per-claim fields): use the actual
    # batch[0] structure for the measured per-claim bytes excluding
    # signature.  Subtract stub sig and add candidate sig.
    base_bytes = (packet_bytes(batches[0], 0) / args.N)
    for scheme, ssz in PQC_SIG_BYTES.items():
        per_claim = base_bytes + ssz / args.N
        N_star = (ssz / max(1, 4096/8 - base_bytes)) if base_bytes < 512 else float("nan")
        print(f"  {scheme:<22}  {per_claim:7.1f} B/claim  (N* ~ {N_star:.1f})")

    out = (Path(__file__).resolve().parent.parent / "results"
            / f"{args.dataset}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"[save] wrote {out}")


if __name__ == "__main__":
    main()
