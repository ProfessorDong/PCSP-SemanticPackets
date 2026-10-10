#!/usr/bin/env bash
# Regenerate every result, figure, and table of the PCSP journal paper.
# Requires the venv described in README.md (xgboost, numba, liboqs-python).
set -euo pipefail
cd "$(dirname "$0")/src"
ROOT=$(cd ../.. && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
export OQS_INSTALL_PATH=${OQS_INSTALL_PATH:-$ROOT/.venv/oqs}
$PY tests/test_pcsp2.py            # protocol unit tests
$PY bench_pqc.py                   # liboqs micro-benchmarks
$PY exp_conformance.py             # E1 attack x design matrix
$PY exp_conformal.py --part all    # E2 static certificates, E3 shift + ACI
$PY exp_adversary.py               # E3b evasion, E4 selective fwd, E5 audits
$PY exp_bandwidth.py               # E6 byte accounting, E7 erasure links
$PY exp_latency.py                 # E8 end-to-end latency
$PY make_figs.py
$PY make_tables.py
