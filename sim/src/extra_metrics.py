"""
Per-seed PCSP measurement of the two distinct FAR metrics the paper
reports separately:

  - gate-bypass rate    Pr[Acc | Attacked]
  - semantic FAR        Pr[Acc AND z != Y]   on clean traces

plus the calibration and acceptance-side statistics used to verify
Theorem 1's set-valued certificate-validity bound:

  - top-1 accuracy
  - APS conformal coverage  Pr[Y in C_alpha(X)]
  - mean conformal set size E[|C_alpha(X)|]
  - singleton fraction      Pr[|C_alpha(X)| == 1]
  - rule-of-three 95% upper bound on observed-zero events
"""

from __future__ import annotations
from pathlib import Path
import csv, sys, math
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep import (get_dataset_state, make_backend, POLICIES,
                    atk_replay, atk_modelsub, atk_evidence, atk_stale,
                    policy_accept)
from pcsp import (PCSPClaim, encode_batch, packet_bytes,
                  claim_leaf, merkle_root, _sha256)


def rule_of_three_upper_95(n):
    """Upper 95% confidence bound on probability of an event observed 0
    times in n trials.  Standard rule-of-three: 3/n is tight as n grows."""
    if n <= 0:
        return 1.0
    return min(1.0, 3.0 / n)


def run_one(dataset, backend_name, N, seed, eta=0.10, fresh_win=2,
            alpha=0.10):
    """Same shape as sweep.run_far_only, but also tracks set size,
    singletons, coverage, and semantic correctness for the headline
    metrics demanded by the audit."""
    st = get_dataset_state(dataset, seed, alpha)
    Xte = st["Xte"]; yte = st["yte"]
    sets_te = st["sets_te"]; z_te = st["z_te"]
    H_theta = st["H_theta"]; H_D = st["H_D"]
    cov = st["cov"]; test_acc = st["acc"]

    mean_set = float(np.mean([len(s) for s in sets_te]))
    singleton_frac = float(np.mean([len(s) == 1 for s in sets_te]))

    sig = make_backend(backend_name)
    sk, pk = sig.keygen()
    attacker_sig = make_backend(backend_name)
    atk_sk, _ = attacker_sig.keygen()
    approved = {H_theta}

    n_pkt = len(yte) // N
    batches = []
    # Track per-claim ground truth + top-1 to compute semantic FAR
    batch_yte = []
    batch_zte = []
    for b in range(n_pkt):
        sl = slice(b*N, (b+1)*N)
        # ctx is input-derived (a hash bucket of the input features),
        # not label-derived: the transmitter does not have access to Y
        # when constructing the claim.
        cs = [PCSPClaim(
                z=int(z_te[sl.start + k]),
                c_alpha=tuple(int(c) for c in sets_te[sl.start + k]),
                t=b,
                ctx=int(_sha256(Xte[sl.start + k].tobytes())[0] & 0x01),
                H_x=_sha256(Xte[sl.start + k].tobytes()))
              for k in range(N)]
        batches.append(encode_batch(cs, H_theta, H_D, sig, sk))
        batch_yte.append(yte[sl])
        batch_zte.append(z_te[sl])

    rng = np.random.default_rng(seed)
    attacks = {
        "A1_replay":            lambda B: atk_replay(B, eta, rng),
        "A2_model_sub":         lambda B: atk_modelsub(B, eta, rng,
                                                        sig, atk_sk),
        "A3_evidence_mismatch": lambda B: atk_evidence(B, eta, rng),
        "A4_epoch_rollback":    lambda B: atk_stale(B, eta, rng,
                                                     fresh_win + 1),
    }

    rows = []
    for atk_name, atk_fn in attacks.items():
        adv, atk_idx = atk_fn(batches)
        for pol_name, (chk_s, chk_m, chk_f) in POLICIES.items():
            fa_attacked = 0   # accepted among attacked packets
            tot_attacked = 0
            sem_fa = 0        # accepted AND z != Y (across all packets)
            acc_total = 0
            tot_all = 0
            for j, b in enumerate(adv):
                acc_list = policy_accept(
                    b, j, j, sig, pk, approved,
                    fresh_win, chk_s, chk_m, chk_f)
                ys = batch_yte[j]
                zs = batch_zte[j]
                for k, accpt in enumerate(acc_list):
                    if accpt:
                        acc_total += 1
                        if int(zs[k]) != int(ys[k]):
                            sem_fa += 1
                    tot_all += 1
                if j in atk_idx:
                    fa_attacked += sum(1 for a in acc_list if a)
                    tot_attacked += len(acc_list)
            bypass = fa_attacked / max(1, tot_attacked)
            semfar = sem_fa     / max(1, tot_all)
            cond_unsafe = sem_fa / max(1, acc_total)
            ub95 = rule_of_three_upper_95(tot_attacked) if bypass == 0 else None
            rows.append(dict(
                dataset=dataset, backend=sig.name, N=N, seed=seed,
                attack=atk_name, policy=pol_name,
                gate_bypass_rate=round(bypass, 6),
                gate_bypass_n=tot_attacked,
                gate_bypass_ub95=ub95,
                semantic_far=round(semfar, 6),
                conditional_unsafe=round(cond_unsafe, 6),
                accept_rate=round(acc_total / max(1, tot_all), 6),
                mean_set_size=round(mean_set, 3),
                singleton_frac=round(singleton_frac, 4),
                coverage=round(cov, 4),
                top1_acc=round(test_acc, 4)))
    return rows


def main():
    out_dir = Path(__file__).resolve().parent.parent / "results"
    rows = []
    for ds in ["nslkdd", "cicids"]:
        for bk in ["mldsa"]:                     # backend doesn't change FAR
            for seed in [0, 1, 2, 3, 4]:
                print(f"  [{ds:6s} {bk} seed={seed}] ...", flush=True)
                rows.extend(run_one(ds, bk, N=128, seed=seed))
    out = out_dir / "extra_metrics.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"saved {out}")

    # Headline summary for the paper
    import pandas as pd
    df = pd.read_csv(out)
    print("\n=== headline (PCSP gate, mean over 5 seeds) ===")
    for ds in ["nslkdd", "cicids"]:
        sub = df[(df.dataset == ds) & (df.policy == "pcsp")]
        n_attacked = sub.gate_bypass_n.sum()
        rate = sub.gate_bypass_rate.mean()
        cov = sub.coverage.mean()
        ms  = sub.mean_set_size.mean()
        sf  = sub.singleton_frac.mean()
        acc = sub.top1_acc.mean()
        ar  = sub.accept_rate.mean()
        semf= sub.semantic_far.mean()
        condu = sub.conditional_unsafe.mean()
        ub = 3 / max(1, n_attacked)
        print(f"  {ds}: bypass={rate:.4f}  n_attacked={n_attacked}  "
              f"UB95={ub:.2e}")
        print(f"         coverage={cov:.4f}  mean_set={ms:.2f}  "
              f"singleton={sf:.4f}  top1_acc={acc:.4f}")
        print(f"         accept_rate={ar:.4f}  semantic_far={semf:.4f}  "
              f"cond_unsafe={condu:.4f}")


if __name__ == "__main__":
    main()
