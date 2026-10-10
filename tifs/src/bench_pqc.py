#!/usr/bin/env python3
"""Micro-benchmark of PQC signature schemes and SHA-256 Merkle hashing.

Measures keygen / sign / verify latency for the liboqs (C, AVX2 dispatch)
implementations of ML-DSA (FIPS 204), SLH-DSA (FIPS 205) and Falcon-512,
plus pure-Python dilithium_py ML-DSA-65 and pyspx (SPHINCS+ round-3 C
reference via CFFI) SHA2-128s for comparison.  Single-core pinned.

Requires: OQS_INSTALL_PATH=<repo>/.venv/oqs  (liboqs 0.16.0 shared build)

Outputs:
  tifs/results/pqc_bench.csv
  tifs/results/pqc_bench_env.json
"""
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RES = ROOT / "tifs" / "results"
CORE = 2

if "OQS_INSTALL_PATH" not in os.environ:
    os.environ["OQS_INSTALL_PATH"] = str(ROOT / ".venv" / "oqs")

import oqs  # noqa: E402

# (paper name, liboqs 0.16 mechanism name)
MECHS = [
    ("ML-DSA-44", "ML-DSA-44"),
    ("ML-DSA-65", "ML-DSA-65"),
    ("ML-DSA-87", "ML-DSA-87"),
    ("SLH-DSA-SHA2-128s", "SLH_DSA_PURE_SHA2_128S"),
    ("SLH-DSA-SHA2-128f", "SLH_DSA_PURE_SHA2_128F"),
    ("SLH-DSA-SHA2-192s", "SLH_DSA_PURE_SHA2_192S"),
    ("SLH-DSA-SHAKE-128s", "SLH_DSA_PURE_SHAKE_128S"),
    ("Falcon-512", "Falcon-512"),
]
MSG_SIZES = [96, 160]
R_FAST = 200
R_SLOW = 20          # SLH-DSA 's' signing and keygen
R_PY = 20            # pure-Python / reference comparison
WARMUP = 3


def stats(ts):
    ms = sorted(t * 1e3 for t in ts)
    n = len(ms)

    def pct(p):
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        return ms[f] + (ms[c] - ms[f]) * (k - f)

    return statistics.median(ms), pct(0.05), pct(0.95), statistics.fmean(ms)


def timeit(fn, reps, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    out = []
    pc = time.perf_counter
    for _ in range(reps):
        t0 = pc()
        fn()
        out.append(pc() - t0)
    return out


rows = []


def add(impl, mech, pk, sk, sig, msg, op, ts):
    med, p5, p95, mean = stats(ts)
    rows.append(dict(impl=impl, mechanism=mech, pk_bytes=pk, sk_bytes=sk,
                     sig_bytes=sig, msg_bytes=msg, op=op, reps=len(ts),
                     median_ms=f"{med:.6f}", p5_ms=f"{p5:.6f}",
                     p95_ms=f"{p95:.6f}", mean_ms=f"{mean:.6f}"))
    print(f"{impl:10s} {mech:20s} msg={msg:4} {op:7s} n={len(ts):4} "
          f"med={med:10.4f} ms p5={p5:10.4f} p95={p95:10.4f}", flush=True)


def bench_liboqs():
    enabled = set(oqs.get_enabled_sig_mechanisms())
    sizes = {}
    for name, alg in MECHS:
        assert alg in enabled, alg
        slow = "SLH" in alg and alg.endswith("S")
        with oqs.Signature(alg) as s:
            d = s.details
            pk_b, sk_b, sig_b = (d["length_public_key"],
                                 d["length_secret_key"], d["length_signature"])
            sizes[name] = dict(liboqs_name=alg, pk=pk_b, sk=sk_b, sig=sig_b,
                               nist_level=d["claimed_nist_level"],
                               version=d["version"])
            r_kg = R_SLOW if slow else R_FAST
            add("liboqs", name, pk_b, sk_b, sig_b, 0, "keygen",
                timeit(s.generate_keypair, r_kg))
            pk = s.generate_keypair()
            for mb in MSG_SIZES:
                msg = os.urandom(mb)
                sig = s.sign(msg)
                assert s.verify(msg, sig, pk)
                add("liboqs", name, pk_b, sk_b, len(sig), mb, "sign",
                    timeit(lambda: s.sign(msg), R_SLOW if slow else R_FAST))
                add("liboqs", name, pk_b, sk_b, len(sig), mb, "verify",
                    timeit(lambda: s.verify(msg, sig, pk), R_FAST))
    return sizes


def bench_hash():
    reps = 200_000
    for label, n in (("SHA-256 node (32B||32B)", 64), ("SHA-256 leaf (64B)", 64),
                     ("SHA-256 node (0x01||32B||32B)", 65)):
        data = os.urandom(n)
        h = hashlib.sha256
        # batch timing: per-call overhead of perf_counter would dominate
        batches = []
        for _ in range(WARMUP):
            for _ in range(1000):
                h(data).digest()
        for _ in range(200):
            t0 = time.perf_counter()
            for _ in range(reps // 200):
                h(data).digest()
            batches.append((time.perf_counter() - t0) / (reps // 200))
        add("hashlib", label, 0, 0, 32, n, "hash", batches)
    # node from two separate 32-B children (concatenation included)
    a, b = os.urandom(32), os.urandom(32)
    batches = []
    for _ in range(200):
        t0 = time.perf_counter()
        for _ in range(1000):
            hashlib.sha256(a + b).digest()
        batches.append((time.perf_counter() - t0) / 1000)
    add("hashlib", "SHA-256 node (a+b concat)", 0, 0, 32, 64, "hash", batches)


def bench_python_refs():
    msg_sizes = MSG_SIZES
    try:
        from dilithium_py.ml_dsa import ML_DSA_65
        pk, sk = ML_DSA_65.keygen()
        sig = ML_DSA_65.sign(sk, b"x")
        pkb, skb, sigb = len(pk), len(sk), len(sig)
        add("dilithium_py", "ML-DSA-65", pkb, skb, sigb, 0, "keygen",
            timeit(ML_DSA_65.keygen, R_PY, 1))
        for mb in msg_sizes:
            m = os.urandom(mb)
            sig = ML_DSA_65.sign(sk, m)
            assert ML_DSA_65.verify(pk, m, sig)
            add("dilithium_py", "ML-DSA-65", pkb, skb, sigb, mb, "sign",
                timeit(lambda: ML_DSA_65.sign(sk, m), R_PY, 1))
            add("dilithium_py", "ML-DSA-65", pkb, skb, sigb, mb, "verify",
                timeit(lambda: ML_DSA_65.verify(pk, m, sig), R_PY, 1))
    except ImportError as e:
        print("dilithium_py unavailable:", e)
    try:
        import pyspx.sha2_128s as spx
        seed = os.urandom(spx.crypto_sign_SEEDBYTES)
        pk, sk = spx.generate_keypair(seed)
        pkb, skb, sigb = len(pk), len(sk), spx.crypto_sign_BYTES
        add("pyspx", "SPHINCS+-SHA2-128s", pkb, skb, sigb, 0, "keygen",
            timeit(lambda: spx.generate_keypair(seed), R_PY, 1))
        for mb in msg_sizes:
            m = os.urandom(mb)
            sig = spx.sign(m, sk)
            assert spx.verify(m, sig, pk)
            add("pyspx", "SPHINCS+-SHA2-128s", pkb, skb, len(sig), mb, "sign",
                timeit(lambda: spx.sign(m, sk), R_PY, 1))
            add("pyspx", "SPHINCS+-SHA2-128s", pkb, skb, len(sig), mb,
                "verify", timeit(lambda: spx.verify(m, sig, pk), R_PY, 1))
    except ImportError as e:
        print("pyspx unavailable:", e)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"error: {e}"


def read(p):
    try:
        return Path(p).read_text().strip()
    except OSError:
        return None


def env_meta(sizes):
    from importlib.metadata import version, PackageNotFoundError

    def ver(p):
        try:
            return version(p)
        except PackageNotFoundError:
            return None
    lscpu = sh("lscpu")
    model = next((l.split(":", 1)[1].strip() for l in lscpu.splitlines()
                  if l.startswith("Model name")), None)
    cpu = f"/sys/devices/system/cpu/cpu{CORE}/cpufreq/"
    return dict(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        cpu_model=model,
        pinned_core=CORE,
        core_thread_siblings=read(f"/sys/devices/system/cpu/cpu{CORE}/topology/thread_siblings_list"),
        scaling_governor=read(cpu + "scaling_governor"),
        scaling_driver=read(cpu + "scaling_driver"),
        energy_perf_preference=read(cpu + "energy_performance_preference"),
        cpuinfo_max_freq_khz=read(cpu + "cpuinfo_max_freq"),
        intel_no_turbo=read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
        kernel=platform.release(),
        uname=" ".join(platform.uname()),
        python=sys.version,
        liboqs_version=oqs.oqs_version(),
        liboqs_python_version=oqs.oqs_python_version(),
        liboqs_build="shared, Release, OQS_DIST_BUILD=ON (runtime CPU dispatch), "
                     "OQS_USE_OPENSSL=OFF, gcc " + sh("gcc -dumpfullversion"),
        oqs_install_path=os.environ.get("OQS_INSTALL_PATH"),
        dilithium_py_version=ver("dilithium-py"),
        pyspx_version=ver("PySPX"),
        cpu_flags_avx2="avx2" in lscpu,
        mechanisms=sizes,
        lscpu=lscpu,
        reps=dict(fast=R_FAST, slow_slh_s=R_SLOW, python_refs=R_PY,
                  warmup=WARMUP),
        timer="time.perf_counter, per-call (hash: per-call mean over 1000-call batches)",
    )


def main():
    os.sched_setaffinity(0, {CORE})
    RES.mkdir(parents=True, exist_ok=True)
    sizes = bench_liboqs()
    bench_hash()
    bench_python_refs()
    cols = ["impl", "mechanism", "pk_bytes", "sk_bytes", "sig_bytes",
            "msg_bytes", "op", "reps", "median_ms", "p5_ms", "p95_ms",
            "mean_ms"]
    with open(RES / "pqc_bench.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(RES / "pqc_bench_env.json", "w") as f:
        json.dump(env_meta(sizes), f, indent=2)
    print("wrote", RES / "pqc_bench.csv")


if __name__ == "__main__":
    main()
