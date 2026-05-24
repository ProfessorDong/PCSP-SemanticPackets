"""
Fast bytes-per-claim sweep across N for both datasets and both PQC backends.
Skips attacks entirely: builds one batch at each N and records packet_bytes/N.
Used to populate Table II (verifies Theorem 2 closed form across N).
"""

from __future__ import annotations
from pathlib import Path
import csv, time
import numpy as np

from sweep import (get_dataset_state, make_backend)
from pcsp import (PCSPClaim, encode_batch, packet_bytes,
                  claim_leaf, merkle_root, _sha256)

DATASETS = ["nslkdd", "cicids"]
BACKENDS = ["mldsa", "slhdsa"]
NS       = [8, 32, 128, 512]
SEED     = 0


def one_batch_bytes(dataset, backend_name, N):
    st = get_dataset_state(dataset, SEED)
    Xte = st["Xte"]; sets_te = st["sets_te"]; z_te = st["z_te"]
    H_theta = st["H_theta"]; H_D = st["H_D"]
    if len(z_te) < N:
        return None
    sig = make_backend(backend_name)
    sk, _ = sig.keygen()
    cs = [PCSPClaim(z=int(z_te[k]),
                    c_alpha=tuple(int(c) for c in sets_te[k]),
                    t=0, ctx=int(k & 1),
                    H_x=_sha256(Xte[k].tobytes()))
          for k in range(N)]
    batch = encode_batch(cs, H_theta, H_D, sig, sk)
    return packet_bytes(batch, sig.sig_bytes) / N


def main():
    out_dir = Path(__file__).resolve().parent.parent / "results"
    rows = []
    for ds in DATASETS:
        for bk in BACKENDS:
            for N in NS:
                t0 = time.time()
                bpc = one_batch_bytes(ds, bk, N)
                dt = time.time() - t0
                if bpc is None:
                    continue
                # Closed-form prediction:
                # B_sem = ~43.7 B (3*32 SHA-256 + 4-bit z + 15-bit C_a + 2B ctx + 8B t)
                # B_s   = 3293 (mldsa) or 7856 (slhdsa)
                bs = 3293 if bk == "mldsa" else 7856
                pred = 43.7 + bs / N
                rows.append(dict(dataset=ds, backend=bk, N=N,
                                  bpc_measured=round(bpc, 2),
                                  bpc_predicted=round(pred, 2),
                                  diff=round(bpc - pred, 3)))
                print(f"  {ds:6s} {bk:6s} N={N:4d}  "
                      f"meas={bpc:7.2f}  pred={pred:7.2f}  "
                      f"diff={bpc - pred:+.2f}  ({dt:.1f}s)")

    out = out_dir / "sweep_bytes_per_N.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
