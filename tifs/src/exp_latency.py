"""
E8: end-to-end PCSP v2 encode / verify latency per batch (Python protocol
code + liboqs C signatures), pinned to one core.  Output:
results/e8_latency.csv
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

os.environ.setdefault("OQS_INSTALL_PATH", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".venv", "oqs"))

import pcsp2 as P
import sigs
from common import RES

MECHS = {"ML-DSA-44": "ML-DSA-44", "ML-DSA-65": "ML-DSA-65",
         "Falcon-512": "Falcon-512",
         "SLH-DSA-SHA2-128s": "SLH_DSA_PURE_SHA2_128S"}


def main():
    os.sched_setaffinity(0, {2})
    rng = np.random.default_rng(0)
    rows = []
    for name, mech in MECHS.items():
        sig = sigs.LibOQSBackend(mech)
        sk, pk = sig.keygen()
        bd = P.Bundle(P.H(b"t"), P.H(b"d"), P.MODE_SET, 5, 1)
        for N in [1, 8, 64, 128, 512, 1024]:
            reps = 5 if "SLH" in name else 30
            te, tv, ta = [], [], []
            for r in range(reps + 1):
                tx = P.Transmitter(sig, sk, 1, 0, 0, bd, b"k" * 32,
                                   P.MODE_SET, 5, 1)
                rx = P.Receiver(sig, {(1, 0): pk}, {0: bd},
                                sig_len=(1, sig.sig_bytes)
                                if "Falcon" in name else None)
                cl = [P.Claim(int(z), 0, P.setmask([int(z)], 5))
                      for z in rng.integers(0, 5, N)]
                ev = [rng.bytes(128) for _ in range(N)]
                t0 = time.perf_counter()
                b = tx.encode(cl, ev, t=5)
                data = b.to_bytes()
                t1 = time.perf_counter()
                ok, _, _ = rx.verify_bytes(data, 5)     # parse (G4) + gates
                t2 = time.perf_counter()
                x, s, path = tx.open(b, N // 2)
                t3 = time.perf_counter()
                assert ok and rx.check_audit(b, N // 2, x, s, path)
                t4 = time.perf_counter()
                if r:                                  # drop warm-up
                    te.append(t1 - t0); tv.append(t2 - t1)
                    ta.append(t4 - t3)
            rows.append(dict(scheme=name, N=N, reps=reps,
                             encode_ms=1e3 * np.median(te),
                             verify_ms=1e3 * np.median(tv),
                             audit_check_ms=1e3 * np.median(ta),
                             verify_us_per_claim=1e6 * np.median(tv) / N))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES / "e8_latency.csv", index=False)




def recompute():
    """Audit recomputation cost: CPU inference of the approved XGBoost
    model on one opened 78-feature flow (and per flow in a 1024 batch),
    pinned to one core.  Output: results/e8_recompute.csv"""
    import xgboost as xgb
    from common import scenario
    os.sched_setaffinity(0, {2})
    s = scenario("cic_zd", 0)
    cpu = xgb.XGBClassifier()
    cpu.load_model(bytearray(s["clf"].get_booster().save_raw("json")))
    cpu.set_params(device="cpu", n_jobs=1)
    X = s["X"][:1024]
    rows = []
    for n in (1, 1024):
        ts = []
        for r in range(201):
            t0 = time.perf_counter()
            cpu.predict_proba(X[:n])
            ts.append(time.perf_counter() - t0)
        ts = np.array(ts[1:])
        rows.append(dict(n=n, median_ms=1e3 * np.median(ts),
                         p99_ms=1e3 * np.percentile(ts, 99),
                         per_flow_us=1e6 * np.median(ts) / n))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES / "e8_recompute.csv", index=False)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:] or ["main", "recompute"]:
        globals()[p]()
