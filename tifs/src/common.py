"""Shared scenario setup with cached class-probability matrices.

Scenarios
  nsl_iid    NSL-KDD, KDDTrain+ folds (exchangeable)
  nsl_shift  NSL-KDD, trained/calibrated on KDDTrain+, deployed on KDDTest+
  cic_iid    CIC-IDS-2017, 7 classes, random folds over all five days
  cic_zd     CIC-IDS-2017 binary (benign/attack), trained and calibrated on
             Mon-Wed, deployed on a time-ordered Thu-Fri stream (zero-day
             attack families)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

import datasets as D
import models as Mo

warnings.filterwarnings("ignore")
RES = Path(__file__).resolve().parents[1] / "results"
FIG = Path(__file__).resolve().parents[1] / "figs"
PCACHE = D.CACHE / "probs"


def scenario(name: str, seed: int = 0) -> dict:
    """Return dict(Pca, yca, P, y, K, raw_bytes, X (deployment features),
    clf, classes)."""
    PCACHE.mkdir(parents=True, exist_ok=True)
    f = PCACHE / f"{name}_s{seed}.npz"
    need_model = True
    if name.startswith("nsl"):
        d = D.nslkdd(seed)
        Xtr, ytr, Xca, yca = d["Xtr"], d["ytr"], d["Xca"], d["yca"]
        K, classes = 5, d["classes"]
        if name == "nsl_iid":
            X, y, rb, ev = d["Xte"], d["yte"], d["raw_bytes_te"], d["evid_te"]
        else:
            X, y, rb, ev = d["Xsh"], d["ysh"], d["raw_bytes_sh"], d["evid_sh"]
    elif name == "cic_iid":
        d = D.cicids(seed, temporal=False)
        Xtr, ytr, Xca, yca = d["Xtr"], d["ytr"], d["Xca"], d["yca"]
        X, y, rb, ev = d["Xte"], d["yte"], d["raw_bytes_te"], d["evid_te"]
        K, classes = 7, d["classes"]
    elif name == "cic_zd":
        d = D.cicids(seed, temporal=True)
        b = lambda v: (v > 0).astype(np.int64)
        Xtr, ytr, Xca, yca = d["Xtr_t"], b(d["ytr_t"]), d["Xca_t"], b(d["yca_t"])
        X, y, rb = d["Xsh"], b(d["ysh"]), d["raw_bytes_sh"]
        ev = [r.tobytes() for r in X]
        K, classes = 2, ["benign", "attack"]
    else:
        raise ValueError(name)
    clf = Mo.train_xgb(Xtr, ytr, K, seed=seed, tag=name.split("_")[0]
                       + ("zd" if name == "cic_zd" else ""))
    if f.exists():
        z = np.load(f)
        Pca, P = z["Pca"], z["P"]
    else:
        Pca, P = Mo.proba(clf, Xca), Mo.proba(clf, X)
        np.savez_compressed(f, Pca=Pca, P=P)
    return dict(name=name, Pca=Pca, yca=np.asarray(yca), P=P,
                y=np.asarray(y), K=K, raw_bytes=np.asarray(rb, float),
                X=X, evid=ev, clf=clf, classes=classes,
                Xtr=Xtr, ytr=np.asarray(ytr))
