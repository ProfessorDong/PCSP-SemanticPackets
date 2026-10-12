# PCSP-SemanticPackets

**Proof-Carrying Semantic Packets for zero-trust tactical edge networks
under bandwidth collapse.**

## Current version: `tifs/` (journal experiments)

`tifs/` holds the current implementation and the measured results for the
journal version of this work (manuscript in preparation; not hosted here).
See [`tifs/README.md`](tifs/README.md). Relative to the earlier prototype in
`sim/`, it

* uses a redesigned wire format (PCSP v2: 3-B claims in set mode, a 53-B
  signed header, separate RFC 9162 claim and evidence Merkle trees, salted
  evidence commitments, a signed bundle index and claim counter);
* implements the RFC 4303-style replay window, real serialization/parsing,
  and fragment-independent verification;
* uses valid split-conformal certificates (LAC / randomized APS / Mondrian)
  and receiver-side, audit-driven adaptive conformal inference;
* uses the official NSL-KDD `KDDTrain+` file (125,973 records) and the
  corrected CIC-IDS-2017 release (Liu et al., IEEE CNS 2022), including a
  time-ordered zero-day split;
* measures FIPS 204 / FIPS 205 signatures from liboqs 0.16.0.

`./tifs/run_all.sh` regenerates every result CSV.

## Earlier prototype: `sim/`

The `sim/` directory is the earlier conference-draft prototype, kept for
reference. That draft was withdrawn and is not published. Known issues that
`tifs/` corrects: the non-randomized APS sets over-covered, the NSL-KDD
training file was not the official release, the Merkle tree duplicated odd
leaves without signing the batch size, the in-window replay check was not
implemented, and `pyspx` implements round-3 SPHINCS+, which has the
SLH-DSA-SHA2-128s signature size but is not FIPS 205. The description below
refers to `sim/`.

This repository contains the PCSP encoder/verifier, two real
network-intrusion datasets' loaders, a four-attack adversary, and
the per-seed sweep scripts that reproduce every reported number,
table, and figure in the conference draft.  The paper PDF itself is
not hosted here.

## Highlights

* **PCSP packet construction** — each semantic claim is bound to
  compact cryptographic evidence about provenance, freshness, model
  identity, uncertainty, and authorization, then signed under a
  post-quantum signature scheme.
* **Theorem 1 (Receiver-Side Soundness)** — for an exchangeable
  deployment sample $(X,Y)$ independent of the adversary's
  selection rule, under conformal calibration (coverage $1-\alpha$),
  EUF-CMA security of the signature, and collision-resistance of
  the commitment hash, the joint certificate-validity failure rate
  $\Pr[\mathsf{Acc}\wedge Y\notin C_\alpha(X)]$ against any
  computationally bounded adversary is at most
  $\alpha + \mathrm{negl}(\lambda)$.
* **Theorem 2 (Bandwidth-Saving Regime)** — aggregating $N$ semantic
  claims under one signature drives the per-claim cryptographic
  overhead to $O(1/N)$ and yields explicit break-even
  $N^\star = (B_p + B_s + 2B_H) / (B_x - B_\mathrm{sem})$.
* **Real PQC + real data** — measurements with byte-format
  FIPS-204-conformant ML-DSA-65 (3309 B, `dilithium_py`) and
  FIPS-205-conformant SLH-DSA-SHA2-128s (7856 B, `pyspx`) over the
  public NSL-KDD and CIC-IDS-2017 datasets. Note: ML-DSA-65 is NIST
  security category 3; SLH-DSA-SHA2-128s is category 1, included as
  the smallest stateless hash-based baseline (not security-matched).
* **Empirical headline** — gate-bypass rate $0/n$ on all four attack
  classes (replay, rogue-key model substitution, evidence-leaf
  tampering, epoch rollback) with rule-of-three 95% upper bound below
  $7\times10^{-5}$; APS coverage 99.7% (NSL-KDD) and 100%
  (CIC-IDS-2017) at $\alpha=10\%$; NSL-KDD compression
  $6.8\times$ (ML-DSA-65) and $4.5\times$ (SLH-DSA-SHA2-128s) at
  $N=128$; verification latency $0.72$–$5.17$ ms per packet.

## Layout

```
sim/
  src/
    pcsp.py              # PCSP packet, Merkle aggregation, signature backends
    nslkdd_loader.py     # NSL-KDD 5-superclass loader
    cicids_loader.py     # CIC-IDS-2017 8-superclass loader
    run_experiments.py   # single (dataset, backend, N) run
    sweep.py             # multi-seed / N-sweep / FAR-only / timings driver
    bytes_only_sweep.py  # fast per-N B/claim sweep (Table II)
    render_tables.py     # CSV → LaTeX table fragments
  results/               # measured CSVs reproducing every table cell
  data/                  # gitignored; see Datasets below
```

## Datasets

The two real datasets used in the paper are **not** redistributed
here.  Download them directly from the original hosts:

* **NSL-KDD** — University of New Brunswick, *Canadian Institute for
  Cybersecurity*.
  `KDDTrain+.txt`, `KDDTest+.txt`, `KDDTest-21.txt`, and the 20%
  training split should be placed in `sim/data/`.
* **CIC-IDS-2017** — UNB CIC, *Intrusion Detection Evaluation Dataset
  (CIC-IDS2017)*.  The `MachineLearningCSV.zip` archive expands into
  `sim/data/cicids2017/MachineLearningCVE/` (8 daily CSVs,
  approximately 2.83 M flow records, 78 features).

Both loaders perform their own one-hot encoding, float32 conversion,
label normalisation (including the `\xef\xbf\xbd` mojibake in
CIC-IDS-2017 "Web Attack" labels), and 5- or 8-class superclass
projection.

## Quick start

```bash
# Python deps
pip install numpy pandas xgboost scikit-learn dilithium-py pyspx

# Run one configuration (NSL-KDD, ML-DSA-65, N=128, seed 0)
cd sim/src
python run_experiments.py --dataset nslkdd --backend mldsa --N 128 --seed 0

# Full sweep producing every CSV in sim/results/
python sweep.py
python bytes_only_sweep.py
```

Reproduction wall time is under 5 min on a single CPU at $N{=}128$;
the per-$N$ sweep adds about 40 s end-to-end.

## Signature backends

* `StubBackend` — deterministic placeholder, testing only.
* `DilithiumPyBackend` — pure-Python ML-DSA-65 via `dilithium_py`
  (FIPS 204).
* `PySPXBackend` — pure-Python SLH-DSA-SHA2-128s via `pyspx`
  (FIPS 205).
* `LibOQSBackend` — CFFI binding to `liboqs` (C/AVX2) for
  production-latency benchmarking; lazy-imported so the test
  environment never bootstraps a C build it does not need.

All four backends implement the same `SigBackend` interface, so
`run_experiments.py` and `sweep.py` accept any of them via a single
`--backend` flag.

## Citation

If you use this code, please cite the repository (a journal version is in preparation):

```bibtex
@misc{dong_pcsp_code,
  author       = {Liang Dong},
  title        = {{PCSP-SemanticPackets: Code and Measured Results for
                   Proof-Carrying Semantic Packets}},
  year         = {2026},
  howpublished = {\url{https://github.com/ProfessorDong/PCSP-SemanticPackets}}
}
```

## License

Code released under the MIT License (see `LICENSE`).  The two
datasets used in the paper are governed by their respective
upstream licenses (UNB CIC for NSL-KDD and CIC-IDS-2017).
