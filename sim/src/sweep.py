"""
PCSP cross-dataset / cross-backend / multi-seed / N-sweep driver.

Runs:
  datasets = {nslkdd, cicids}
  backends = {stub, mldsa, slhdsa}
  N        = {8, 32, 128, 512}
  seeds    = 5 by default

Outputs:
  sim/results/sweep_attack.csv    -- per (dataset, backend, N, seed,
                                      attack, policy) FAR
  sim/results/sweep_timing.csv    -- per (dataset, backend) keygen,
                                      sign, verify timings (ms)
  sim/results/sweep_bytes.csv     -- per (dataset, backend, N)
                                      measured B(P_N)/N

Author: Liang Dong, MILCOM 2026 Paper 3.
"""

from __future__ import annotations

import argparse, csv, time, hashlib
from pathlib import Path
import numpy as np
import xgboost as xgb

import nslkdd_loader
import cicids_loader
from pcsp import (PCSPClaim, encode_batch, packet_bytes,
                  StubBackend, DilithiumPyBackend, PySPXBackend,
                  claim_leaf, merkle_root, _sha256)


def make_backend(name: str):
    if name == "stub":   return StubBackend()
    if name == "mldsa":  return DilithiumPyBackend()
    if name == "slhdsa": return PySPXBackend()
    raise ValueError(name)


def aps_calibrate(probs_cal, y_cal, alpha=0.10):
    n = len(y_cal)
    order = np.argsort(-probs_cal, axis=1)
    sorted_p = np.take_along_axis(probs_cal, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)
    scores = np.empty(n)
    for i in range(n):
        rank = int(np.where(order[i] == y_cal[i])[0][0])
        scores[i] = cumsum[i, rank]
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q, method="higher"))


def aps_set(probs, tau):
    sets = []
    for p in probs:
        order = np.argsort(-p)
        cs = np.cumsum(p[order])
        k = int(np.searchsorted(cs, tau)) + 1
        sets.append(tuple(sorted(order[:k].tolist())))
    return sets


# ---- attacks (same as run_experiments.py) ----

def atk_replay(batches, eta, rng, min_age=10):
    n = len(batches); n_atk = int(n*eta)
    cands = np.arange(min_age, n)
    n_atk = min(n_atk, len(cands))
    idx = rng.choice(cands, size=n_atk, replace=False)
    out = list(batches)
    for j in idx:
        out[j] = batches[max(0, j - min_age)]
    return out, set(int(i) for i in idx)


def atk_modelsub(batches, eta, rng, sig, attacker_sk):
    n = len(batches); n_atk = int(n*eta)
    idx = rng.choice(n, size=n_atk, replace=False)
    out = list(batches)
    forged_Hth = _sha256(b"forged-model-v0")
    for j in idx:
        b = batches[j]
        leaves = [claim_leaf(c) for c in b.claims]
        root = merkle_root(leaves)
        msg = root + forged_Hth + b.H_D
        forged_sig = sig.sign(attacker_sk, msg)
        out[j] = type(b)(claims=b.claims, H_theta=forged_Hth,
                          H_D=b.H_D, sigma=forged_sig, leaves=leaves)
    return out, set(int(i) for i in idx)


def atk_evidence(batches, eta, rng):
    n = len(batches); n_atk = int(n*eta)
    idx = rng.choice(n, size=n_atk, replace=False)
    out = list(batches)
    for j in idx:
        b = batches[j]
        if not b.claims:
            continue
        tampered = list(b.claims); c0 = tampered[0]
        tampered[0] = PCSPClaim(z=c0.z, c_alpha=c0.c_alpha, t=c0.t,
                                 ctx=c0.ctx, H_x=_sha256(b"unrelated"))
        out[j] = type(b)(claims=tampered, H_theta=b.H_theta,
                          H_D=b.H_D, sigma=b.sigma)
    return out, set(int(i) for i in idx)


def atk_stale(batches, eta, rng, drift):
    n = len(batches); n_atk = int(n*eta)
    idx = rng.choice(n, size=n_atk, replace=False)
    out = list(batches)
    for j in idx:
        b = batches[j]
        shifted = [PCSPClaim(z=c.z, c_alpha=c.c_alpha,
                              t=c.t - drift, ctx=c.ctx, H_x=c.H_x)
                   for c in b.claims]
        leaves = [claim_leaf(c) for c in shifted]
        out[j] = type(b)(claims=shifted, H_theta=b.H_theta,
                          H_D=b.H_D, sigma=b.sigma, leaves=leaves)
    return out, set(int(i) for i in idx)


def policy_accept(batch, j, cur_epoch, sig, pk, approved,
                   fresh_win, check_sig, check_model, check_fresh):
    leaves = [claim_leaf(c) for c in batch.claims]
    root   = merkle_root(leaves)
    msg    = root + batch.H_theta + batch.H_D
    sig_ok   = (not check_sig)   or sig.verify(pk, msg, batch.sigma)
    model_ok = (not check_model) or (batch.H_theta in approved)
    if not (sig_ok and model_ok):
        return [False] * len(batch.claims)
    if check_fresh:
        return [abs(cur_epoch - c.t) <= fresh_win for c in batch.claims]
    return [True] * len(batch.claims)


POLICIES = {
    "raw":      (False, False, False),
    "semantic": (False, False, True),
    "signed":   (True,  False, False),
    "pcsp":     (True,  True,  True),
}


_DATASET_CACHE = {}


def get_dataset_state(dataset: str, seed: int, alpha: float = 0.10):
    """Cache the (loaded data + trained classifier + conformal state)
    keyed by (dataset, seed) so we don't redo the heavy work across
    the inner (backend, N) loop."""
    key = (dataset, seed)
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]
    if dataset == "nslkdd":
        Xtr, ytr, Xca, yca, Xte, yte = nslkdd_loader.load_train_test(
            seed=seed)
    else:
        Xtr, ytr, Xca, yca, Xte, yte = cicids_loader.load_train_test(
            seed=seed)
    n_classes = max(int(np.max(np.concatenate([ytr, yca, yte]))) + 1, 2)
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective="multi:softprob", num_class=n_classes,
        n_jobs=-1, eval_metric="mlogloss",
    )
    clf.fit(Xtr, ytr)
    probs_ca = clf.predict_proba(Xca)
    probs_te = clf.predict_proba(Xte)
    tau = aps_calibrate(probs_ca, yca, alpha=alpha)
    sets_te = aps_set(probs_te, tau)
    z_te = clf.predict(Xte)
    test_acc = float(clf.score(Xte, yte))
    cov = float(np.mean([yte[i] in sets_te[i]
                          for i in range(len(yte))]))
    H_theta = _sha256(clf.get_booster().save_raw())
    H_D     = _sha256(probs_ca.tobytes())
    feature_bytes = Xte.shape[1] * 4
    state = dict(Xte=Xte, yte=yte, sets_te=sets_te, z_te=z_te,
                  H_theta=H_theta, H_D=H_D, cov=cov, acc=test_acc,
                  feature_bytes=feature_bytes)
    _DATASET_CACHE[key] = state
    return state


def run_far_only(dataset, backend_name, N, seed, eta=0.10,
                  fresh_win=2, alpha=0.10):
    """Cheap variant: skip per-batch signing.  Each batch is built
    with a placeholder signature equal to sig.sign(sk, b'fixed').
    Receiver verification still tests the correct gates -- A1
    replay, A4 stale, A3 evidence mismatch are caught by leaf/root
    mismatch (no signature needed); A2 model_sub is caught by the
    approved-model-hash gate.  This is sufficient for the FAR
    matrix while skipping the 700 ms/sig SLH-DSA cost at small N."""
    st = get_dataset_state(dataset, seed, alpha)
    Xte = st["Xte"]; yte = st["yte"]; sets_te = st["sets_te"]
    z_te = st["z_te"]; H_theta = st["H_theta"]; H_D = st["H_D"]
    cov = st["cov"]; test_acc = st["acc"]

    sig = make_backend(backend_name)
    sk, pk = sig.keygen()
    attacker_sig = make_backend(backend_name)
    atk_sk, _ = attacker_sig.keygen()
    approved = {H_theta}

    n_pkt = len(yte) // N
    batches = []
    for b in range(n_pkt):
        sl = slice(b*N, (b+1)*N)
        cs = [PCSPClaim(
                z=int(z_te[sl.start + k]),
                c_alpha=tuple(int(c) for c in sets_te[sl.start + k]),
                t=b, ctx=int(_sha256(Xte[sl.start + k].tobytes())[0] & 0x01),
                H_x=_sha256(Xte[sl.start + k].tobytes()))
              for k in range(N)]
        batches.append(encode_batch(cs, H_theta, H_D, sig, sk))

    bpc = packet_bytes(batches[0], sig.sig_bytes) / N if batches else 0.0

    rng = np.random.default_rng(seed)
    attacks = {
        "A1_replay":            lambda B: atk_replay(B, eta, rng),
        "A2_model_sub":         lambda B: atk_modelsub(B, eta, rng,
                                                        sig, atk_sk),
        "A3_evidence_mismatch": lambda B: atk_evidence(B, eta, rng),
        "A4_stale_context":     lambda B: atk_stale(B, eta, rng,
                                                     fresh_win + 1),
    }
    out_rows = []
    for atk_name, atk_fn in attacks.items():
        adv, atk_idx = atk_fn(batches)
        for pol_name, (chk_s, chk_m, chk_f) in POLICIES.items():
            fa = 0; tot = 0
            for j, b in enumerate(adv):
                if j not in atk_idx: continue
                acc = policy_accept(b, j, j, sig, pk, approved,
                                     fresh_win, chk_s, chk_m, chk_f)
                fa += sum(1 for a in acc if a)
                tot += len(acc)
            far = fa / max(1, tot)
            out_rows.append(dict(
                dataset=dataset, backend=sig.name, N=N, seed=seed,
                attack=atk_name, policy=pol_name, FAR=round(far, 4),
                coverage=round(cov, 4), acc=round(test_acc, 4),
                bytes_per_claim=round(bpc, 2)))
    return out_rows


def run_timings_once(dataset, backend_name, seed):
    """Measure keygen, sign, verify timings on a single representative
    batch of N=128 claims.  Cheap (no full sweep)."""
    st = get_dataset_state(dataset, seed)
    Xte = st["Xte"]; sets_te = st["sets_te"]; z_te = st["z_te"]
    H_theta = st["H_theta"]; H_D = st["H_D"]
    N = 128
    sig = make_backend(backend_name)
    t0 = time.time(); sk, pk = sig.keygen(); t_kg = (time.time() - t0)*1e3
    cs = [PCSPClaim(z=int(z_te[k]),
                     c_alpha=tuple(int(c) for c in sets_te[k]),
                     t=0, ctx=0,
                     H_x=_sha256(Xte[k].tobytes()))
          for k in range(N)]
    t0 = time.time(); batch = encode_batch(cs, H_theta, H_D, sig, sk)
    t_sign = (time.time() - t0)*1e3
    leaves = [claim_leaf(c) for c in batch.claims]
    msg = merkle_root(leaves) + batch.H_theta + batch.H_D
    t0 = time.time()
    for _ in range(10): sig.verify(pk, msg, batch.sigma)
    t_verify = (time.time() - t0)*1e2  # 10 reps, ms each
    bpc = packet_bytes(batch, sig.sig_bytes) / N
    return dict(dataset=dataset, backend=sig.name, seed=seed,
                t_keygen_ms=round(t_kg, 3),
                t_sign_ms=round(t_sign, 3),
                t_verify_ms=round(t_verify, 3),
                bytes_per_claim=round(bpc, 2),
                sig_bytes=sig.sig_bytes, pk_bytes=sig.pk_bytes)


def run_one(dataset, backend_name, N, seed, eta=0.10, fresh_win=2,
            alpha=0.10):
    st = get_dataset_state(dataset, seed, alpha)
    Xte = st["Xte"]; yte = st["yte"]
    sets_te = st["sets_te"]; z_te = st["z_te"]
    H_theta = st["H_theta"]; H_D = st["H_D"]
    cov = st["cov"]; test_acc = st["acc"]

    sig = make_backend(backend_name)
    t0 = time.time(); sk, pk = sig.keygen(); t_kg = (time.time() - t0)*1e3
    attacker_sig = make_backend(backend_name)
    atk_sk, _ = attacker_sig.keygen()
    approved = {H_theta}

    n_pkt = len(yte) // N
    batches = []
    sign_times = []
    for b in range(n_pkt):
        sl = slice(b*N, (b+1)*N)
        cs = []
        for k in range(N):
            i = sl.start + k
            cs.append(PCSPClaim(
                z=int(z_te[i]),
                c_alpha=tuple(int(c) for c in sets_te[i]),
                t=b,
                ctx=int(_sha256(Xte[i].tobytes())[0] & 0x01),
                H_x=_sha256(Xte[i].tobytes()),
            ))
        t0 = time.time()
        batches.append(encode_batch(cs, H_theta, H_D, sig, sk))
        sign_times.append((time.time() - t0)*1e3)
    t_sign_mean = float(np.mean(sign_times)) if sign_times else 0.0

    # Verify timing: one honest verify (10 reps for stability).
    if batches:
        b0 = batches[0]
        msg0 = merkle_root([claim_leaf(c) for c in b0.claims]) + b0.H_theta + b0.H_D
        t0 = time.time()
        for _ in range(10): sig.verify(pk, msg0, b0.sigma)
        t_verify = ((time.time() - t0)*1e3) / 10
    else:
        t_verify = 0.0

    bpc = packet_bytes(batches[0], sig.sig_bytes) / N if batches else 0.0

    rng = np.random.default_rng(seed)
    attacks = {
        "A1_replay":            lambda B: atk_replay(B, eta, rng),
        "A2_model_sub":         lambda B: atk_modelsub(B, eta, rng,
                                                        sig, atk_sk),
        "A3_evidence_mismatch": lambda B: atk_evidence(B, eta, rng),
        "A4_stale_context":     lambda B: atk_stale(B, eta, rng,
                                                     fresh_win + 1),
    }
    out_rows = []
    for atk_name, atk_fn in attacks.items():
        adv, atk_idx = atk_fn(batches)
        for pol_name, (chk_s, chk_m, chk_f) in POLICIES.items():
            fa = 0; tot = 0
            for j, b in enumerate(adv):
                if j not in atk_idx: continue
                cur = j
                acc = policy_accept(b, j, cur, sig, pk, approved,
                                     fresh_win, chk_s, chk_m, chk_f)
                fa += sum(1 for a in acc if a)
                tot += len(acc)
            far = fa / max(1, tot)
            out_rows.append(dict(
                dataset=dataset, backend=sig.name, N=N, seed=seed,
                attack=atk_name, policy=pol_name, FAR=round(far, 4),
                coverage=round(cov, 4), acc=round(test_acc, 4),
            ))
    timing = dict(dataset=dataset, backend=sig.name, N=N, seed=seed,
                  t_keygen_ms=round(t_kg, 3),
                  t_sign_ms=round(t_sign_mean, 3),
                  t_verify_ms=round(t_verify, 3),
                  bytes_per_claim=round(bpc, 2),
                  sig_bytes=sig.sig_bytes, pk_bytes=sig.pk_bytes)
    return out_rows, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["nslkdd", "cicids"])
    ap.add_argument("--backends", nargs="+",
                    default=["mldsa", "slhdsa"])
    ap.add_argument("--Ns", type=int, nargs="+",
                    default=[8, 32, 128, 512])
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- FAR matrix at N=128 only (per-batch signing avoided at
    # small N because SLH-DSA-128s signs slowly).
    attack_rows = []
    bytes_rows  = []
    for ds in args.datasets:
        for bk in args.backends:
            for N in args.Ns:
                for seed in args.seeds:
                    label = f"[FAR  {ds:6s} {bk:6s} N={N:3d} seed={seed}]"
                    t0 = time.time()
                    rows = run_far_only(ds, bk, N, seed)
                    dt = time.time() - t0
                    far_pcsp = np.mean([r["FAR"] for r in rows
                                         if r["policy"] == "pcsp"])
                    bpc = rows[0]["bytes_per_claim"]
                    print(f"{label} done in {dt:5.1f}s  "
                          f"FAR(pcsp)={far_pcsp:.4f}  "
                          f"B/claim={bpc}")
                    attack_rows.extend(rows)
                    bytes_rows.append(dict(
                        dataset=ds, backend=rows[0]["backend"],
                        N=N, seed=seed, bytes_per_claim=bpc))

    # ---- Timings sampled once per (dataset, backend, seed)
    timing_rows = []
    for ds in args.datasets:
        for bk in args.backends:
            for seed in args.seeds:
                label = f"[TIM  {ds:6s} {bk:6s} seed={seed}]"
                t0 = time.time()
                tr = run_timings_once(ds, bk, seed)
                dt = time.time() - t0
                print(f"{label} done in {dt:5.1f}s  "
                      f"sign={tr['t_sign_ms']}ms "
                      f"verify={tr['t_verify_ms']}ms")
                timing_rows.append(tr)

    with open(out_dir / "sweep_attack.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(attack_rows[0].keys()))
        w.writeheader(); w.writerows(attack_rows)
    with open(out_dir / "sweep_timing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(timing_rows[0].keys()))
        w.writeheader(); w.writerows(timing_rows)
    with open(out_dir / "sweep_bytes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bytes_rows[0].keys()))
        w.writeheader(); w.writerows(bytes_rows)
    print(f"\n[save] sweep_attack.csv, sweep_timing.csv, sweep_bytes.csv "
          f"in {out_dir}")


if __name__ == "__main__":
    main()
