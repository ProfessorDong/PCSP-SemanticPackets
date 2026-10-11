# PCSP journal experiments (`tifs/`)

Code and measured results for the journal version of *Proof-Carrying Semantic
Packets*. The manuscripts (`tifs/paper/`, `tdsc/`) are not part of the repository.

| Script | What it produces |
|---|---|
| `src/pcsp2.py` | PCSP v2 wire format, RFC 6962 claim/evidence trees, salted evidence commitments, receiver gates G1-G4, RFC 4303-style replay window, audit openings |
| `src/conformal.py` | split conformal (LAC, randomized APS, Mondrian), 8-bit score quantization, receiver-side audit-driven ACI (numba) |
| `src/exp_conformance.py` | `results/e1_conformance.csv`: 11 attacks x 6 receiver designs, liboqs ML-DSA-65 |
| `src/exp_conformal.py` | `results/e2_static.csv`, `results/e3_shift.csv`, `results/e3_trace_*.csv` |
| `src/exp_adversary.py` | `results/e3b_*.csv` (evasion), `e4_selective.csv`, `e5_accountability.csv` |
| `src/exp_bandwidth.py` | `results/e6_*.csv` (byte accounting, baselines), `e7_*.csv` (erasure links) |
| `src/exp_latency.py`, `src/bench_pqc.py` | `results/e8_latency.csv`, `results/pqc_bench*.{csv,json}` |
| `src/exp_tolerance.py` | `results/e9_tolerance.csv`, `results/e9_trace.csv`: compromised sender colluding with the intruder (revocation, timeout quarantine, ACI validity) |
| `src/make_figs.py`, `src/make_tables.py` | matplotlib figures and LaTeX table bodies |
| `src/make_tdsc.py`, `src/make_pgfdata.py` | table bodies (incl. tolerance and decision-level performability) and pgfplots data series for the TDSC manuscript |

`./run_all.sh` regenerates everything (about 30 min on one desktop with a GPU;
XGBoost uses CUDA).

## Environment

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install xgboost scikit-learn pandas pyarrow numba matplotlib scipy \
    dilithium-py pyspx cmake ninja liboqs-python==0.16.0.1
# build liboqs 0.16.0 (shared, Release) and install to .venv/oqs, then
export OQS_INSTALL_PATH=$PWD/.venv/oqs
```

## Data

* NSL-KDD: the official `KDDTrain+.txt` (125,973 rows,
  sha256 `1b86d2f9...aadf95`) and `KDDTest+.txt` in `sim/data/nslkdd_official/`.
* CIC-IDS-2017: `MachineLearningCSV.zip` from UNB CIC expanded into
  `sim/data/cicids2017/MachineLearningCVE/`.
