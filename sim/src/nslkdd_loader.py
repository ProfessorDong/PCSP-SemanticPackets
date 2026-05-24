"""
NSL-KDD loader for PCSP experiments (MILCOM 2026 Paper 3).

The NSL-KDD dataset (Tavallaee et al., 2009) refines the venerable
KDD'99 intrusion-detection benchmark by removing duplicated records
and rebalancing record difficulty.  Each row is a flow with 41
features (3 categorical, 38 numeric) plus a class label.

We download the files into sim/data/.  The defcom17 GitHub mirror
serves a deduplicated KDDTrain+.txt with ~20k records, which is
plenty for our PCSP demo.

Columns (NSL-KDD format):
  0-40  : features (see COL_NAMES)
  41    : attack_class
  42    : difficulty_level (unused here)

Five superclasses we collapse the 39 distinct attack labels into:
  normal, dos, probe, r2l (remote-to-local), u2r (user-to-root)

Author: Liang Dong, MILCOM 2026 Paper 3.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

# Standard NSL-KDD 41-feature schema + 2 trailing label columns.
COL_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack_class", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]

# Map fine-grained attacks to the 5 standard superclasses.
ATTACK_GROUPS = {
    "normal": ["normal"],
    "dos": ["back", "land", "neptune", "pod", "smurf", "teardrop",
            "apache2", "udpstorm", "processtable", "mailbomb",
            "worm"],
    "probe": ["ipsweep", "nmap", "portsweep", "satan", "mscan",
              "saint"],
    "r2l": ["ftp_write", "guess_passwd", "imap", "multihop", "phf",
            "spy", "warezclient", "warezmaster", "sendmail", "named",
            "snmpgetattack", "snmpguess", "xlock", "xsnoop",
            "httptunnel"],
    "u2r": ["buffer_overflow", "loadmodule", "perl", "rootkit", "ps",
            "sqlattack", "xterm"],
}

# Inverse map: attack name -> super-class
SUPERCLASS = {atk: g for g, names in ATTACK_GROUPS.items() for atk in names}
CLASSES = ["normal", "dos", "probe", "r2l", "u2r"]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}


def _data_dir() -> Path:
    return (Path(__file__).resolve().parent.parent / "data").resolve()


def load_split(filename: str) -> pd.DataFrame:
    """Read one NSL-KDD CSV split and return it with a `superclass`
    column added.
    """
    path = _data_dir() / filename
    df = pd.read_csv(path, header=None, names=COL_NAMES)
    df["superclass"] = df["attack_class"].map(
        lambda a: SUPERCLASS.get(a, "r2l"))   # unknowns -> r2l bucket
    df["y"] = df["superclass"].map(CLASS_INDEX)
    return df


def load_train_test(
        train_file: str = "KDDTrain+.txt",
        test_file: str = "KDDTest+.txt",
        calib_frac: float = 0.20,
        seed: int = 0):
    """Return (X_train, y_train, X_calib, y_calib, X_test, y_test).

    Categorical columns are one-hot encoded; numeric columns are
    Z-scored using the train-set mean/std.  Calibration is a held-out
    slice of the training set (PCSP conformal-prediction calibration
    requires data exchangeable with the deployment distribution; the
    NSL-KDD test set is intentionally distribution-shifted, so we
    calibrate on a training-set slice and report the resulting
    coverage gap on the test set as part of the experiment.)
    """
    train_df = load_split(train_file)
    test_df  = load_split(test_file)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_df))
    n_cal = int(len(train_df) * calib_frac)
    cal_idx, tr_idx = idx[:n_cal], idx[n_cal:]
    train_part = train_df.iloc[tr_idx].reset_index(drop=True)
    calib_part = train_df.iloc[cal_idx].reset_index(drop=True)

    # One-hot the categorical columns using train-only categories.
    train_cat = pd.get_dummies(train_part[CATEGORICAL],
                                columns=CATEGORICAL, drop_first=False)
    cat_cols = list(train_cat.columns)

    def _featurize(df):
        cat = pd.get_dummies(df[CATEGORICAL], columns=CATEGORICAL,
                              drop_first=False).reindex(columns=cat_cols,
                                                         fill_value=0)
        num = df.drop(columns=CATEGORICAL + ["attack_class", "difficulty",
                                              "superclass", "y"])
        return pd.concat([num.reset_index(drop=True),
                          cat.reset_index(drop=True)], axis=1)

    X_train = _featurize(train_part).astype("float32")
    X_calib = _featurize(calib_part).astype("float32")
    X_test  = _featurize(test_df).astype("float32")

    # Z-score using train statistics (categorical 0/1 left as is).
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    mu = X_train[num_cols].mean()
    sd = X_train[num_cols].std().replace(0, 1.0)
    for X in (X_train, X_calib, X_test):
        X[num_cols] = (X[num_cols] - mu) / sd

    return (X_train.values, train_part["y"].values,
            X_calib.values, calib_part["y"].values,
            X_test.values,  test_df["y"].values)


if __name__ == "__main__":
    Xtr, ytr, Xca, yca, Xte, yte = load_train_test()
    print(f"train: {Xtr.shape}  classes: {np.bincount(ytr)}")
    print(f"calib: {Xca.shape}  classes: {np.bincount(yca)}")
    print(f"test : {Xte.shape}  classes: {np.bincount(yte)}")
