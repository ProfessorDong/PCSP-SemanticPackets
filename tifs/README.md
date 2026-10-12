# PCSP journal experiments (`tifs/`)

Code and measured results for the journal version of *Proof-Carrying Semantic
Packets*. The manuscripts (`tifs/paper/`, `tdsc/`) are not part of the repository.

| Script | What it produces |
|---|---|
| `src/pcsp2.py` | PCSP v2 wire format, RFC 9162 claim/evidence trees, salted evidence commitments, receiver gates G1-G4 (G4 binds the class to its certificate), RFC 4303-style replay window, FIV registrations pinned to bundle and key, audit openings with authenticated selective-repeat repair (26-B challenge carrying slot, round, missing-fragment bitmap, MAC; cached openings; 9-B fragment header) |
| `src/conformal.py` | split conformal (LAC, randomized APS, Mondrian), 8-bit score quantization, receiver-side audit-driven ACI (numba) |
| `src/exp_conformance.py` | `results/e1_conformance.csv`: 12 attacks (plus honest traffic) x 6 receiver designs, liboqs ML-DSA-65 |
| `src/exp_conformal.py` | `results/e2_static.csv`, `results/e3_shift.csv`, `results/e3_trace_*.csv` |
| `src/exp_adversary.py` | `results/e3b_*.csv` (evasion), `e4_selective.csv`, `e5_accountability.csv` |
| `src/exp_risk.py` | `results/e3d_*.csv`: error per autonomous action, false-benign rate, worst-window risk, confidence-threshold / random-abstention / importance-weighted ACI baselines, labeling-service errors (`e3d_masking.csv`: labels that mask model errors), local-risk traces |
| `src/exp_bandwidth.py` | `results/e6_*.csv` (byte accounting, length-prefixed zlib baseline), `e7_*.csv` (erasure links; `e7_fig.csv` exact expectations and Monte Carlo with 95% intervals (Clopper-Pearson for MONO), `e7_opt.csv` integer MONO optimum) |
| `src/exp_latency.py`, `src/bench_pqc.py` | `results/e8_latency.csv`, `results/e8_recompute.csv` (audit model recomputation), `results/pqc_bench*.{csv,json}` |
| `src/exp_tolerance.py` | `results/e9_tolerance.csv`, `results/e9_trace.csv`: compromised sender colluding with the intruder; falsification is a model discrepancy on the committed evidence (not the attack label); causal opening, deadline, and key-action timing; KL anytime quarantine test |
| `src/exp_integrated.py` | `results/e10_*.csv`: coupled simulation of batches, audits, fragmented openings with selective-repeat repair (whole-opening repeat as baseline), and ACI on one lossy link over a common horizon of 196,608 claims; batch size selected on validation seeds 5-7 (`e10_selected.csv`: mean miscoverage <= alpha, >= 100 completed audits per trace), then evaluated on held-out seeds with cluster-bootstrap intervals and honest-sender quarantine counts (decision-level performability) |
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
