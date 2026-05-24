"""
CIC-IDS-2017 loader for PCSP experiments (MILCOM 2026 Paper 3).

CIC-IDS-2017 (Sharafaldin et al., 2018) is a labeled flow-level
intrusion-detection benchmark built from 5 days of captured traffic
in a realistic small-network testbed.  The CICFlowMeter-extracted
CSVs in ``MachineLearningCVE/`` give 78 numeric flow features per
record plus a single ``Label`` column.

We collapse the 14 attack categories into a 8-superclass schema
(BENIGN + 7 attack groups) that fits the PCSP semantic-claim
alphabet:

  BENIGN, DOS, DDOS, PortScan, BruteForce, WebAttack, Botnet, Other

CSVs are loaded across all 8 day-files, concatenated, sub-sampled
(by default to 200k rows) for fast experimentation, cleaned of
NaN/Inf values, then split into train / calibration / test.

Author: Liang Dong, MILCOM 2026 Paper 3.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

# Map fine-grained labels to PCSP superclasses.  Labels with
# trailing whitespace or odd capitalisation are normalised below.
SUPERCLASS = {
    "benign":                  "benign",
    "dos hulk":                "dos",
    "dos goldeneye":           "dos",
    "dos slowloris":           "dos",
    "dos slowhttptest":        "dos",
    "heartbleed":              "dos",
    "ddos":                    "ddos",
    "portscan":                "portscan",
    "ftp-patator":             "brute_force",
    "ssh-patator":             "brute_force",
    "web attack  brute force": "web_attack",
    "web attack  xss":   "web_attack",
    "web attack  sql injection": "web_attack",
    # variants without unicode dash:
    "web attack - brute force": "web_attack",
    "web attack - xss":         "web_attack",
    "web attack - sql injection": "web_attack",
    "web attack brute force":  "web_attack",
    "web attack xss":          "web_attack",
    "web attack sql injection": "web_attack",
    "bot":                     "botnet",
    "infiltration":            "other",
}

CLASSES = ["benign", "dos", "ddos", "portscan", "brute_force",
            "web_attack", "botnet", "other"]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}


def _data_dir() -> Path:
    return (Path(__file__).resolve().parent.parent
            / "data" / "cicids2017" / "MachineLearningCVE")


def _normalise_label(s: str) -> str:
    # CIC-IDS-2017's 'Web Attack ...' labels contain a mojibake
    # sequence (Unicode replacement char) from a corrupted source
    # encoding.  Collapse all non-ASCII bytes to a single space
    # and squash repeated whitespace.
    s = s.strip().lower()
    s = ''.join(ch if ord(ch) < 128 else ' ' for ch in s)
    return ' '.join(s.split())


def load_all(max_rows: int | None = 200_000, seed: int = 0
             ) -> pd.DataFrame:
    """Read all 8 day-CSVs, concatenate, clean, sub-sample.
    """
    here = _data_dir()
    if not here.exists():
        raise FileNotFoundError(f"CIC-IDS-2017 CSVs not at {here}")
    files = sorted(here.glob("*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding="latin-1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        dfs.append(df)
    big = pd.concat(dfs, ignore_index=True)
    # Normalise labels and map to superclass.
    big["Label"] = big["Label"].astype(str).map(_normalise_label)
    big["superclass"] = big["Label"].map(SUPERCLASS).fillna("other")
    big["y"] = big["superclass"].map(CLASS_INDEX)
    # Drop rows whose features contain NaN/Inf.
    big = big.replace([np.inf, -np.inf], np.nan)
    big = big.dropna()
    # Drop the label column from features later; keep "y" for labels.
    if max_rows and len(big) > max_rows:
        big = big.sample(n=max_rows, random_state=seed
                          ).reset_index(drop=True)
    return big


def load_train_test(max_rows: int | None = 200_000,
                     calib_frac: float = 0.20,
                     test_frac: float = 0.20,
                     seed: int = 0):
    """Return (X_train, y_train, X_calib, y_calib, X_test, y_test).

    All features are numeric in CIC-IDS-2017's ML-ready CSVs, so we
    z-score using train-set statistics and drop the Label column.
    """
    df = load_all(max_rows=max_rows, seed=seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * test_frac)
    n_cal  = int(len(df) * calib_frac)
    te_idx = idx[:n_test]
    ca_idx = idx[n_test:n_test + n_cal]
    tr_idx = idx[n_test + n_cal:]

    feature_cols = [c for c in df.columns
                     if c not in ("Label", "superclass", "y")]

    def featurise(sub):
        X = sub[feature_cols].astype("float32").copy()
        return X, sub["y"].values

    Xtr, ytr = featurise(df.iloc[tr_idx])
    Xca, yca = featurise(df.iloc[ca_idx])
    Xte, yte = featurise(df.iloc[te_idx])

    mu = Xtr.mean()
    sd = Xtr.std().replace(0, 1.0)
    for X in (Xtr, Xca, Xte):
        X[feature_cols] = (X[feature_cols] - mu) / sd

    return (Xtr.values, ytr, Xca.values, yca, Xte.values, yte)


if __name__ == "__main__":
    Xtr, ytr, Xca, yca, Xte, yte = load_train_test()
    print(f"CIC-IDS-2017 (200k rows sub-sampled)")
    print(f"train: {Xtr.shape}  classes: {np.bincount(ytr)}")
    print(f"calib: {Xca.shape}  classes: {np.bincount(yca)}")
    print(f"test : {Xte.shape}  classes: {np.bincount(yte)}")
    print(f"feature count: {Xtr.shape[1]} (raw B_x = {Xtr.shape[1]*4} B)")
