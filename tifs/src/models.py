"""Classifier training with on-disk caching (one model per dataset/seed)."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb

CACHE = Path(__file__).resolve().parents[2] / "sim" / "data" / "cache" / "models"


def train_xgb(X, y, K, seed=0, tag=None, n_estimators=300, max_depth=8):
    CACHE.mkdir(parents=True, exist_ok=True)
    key = tag or hashlib.sha256(X[:1000].tobytes() + y[:1000].tobytes()
                                ).hexdigest()[:16]
    f = CACHE / f"xgb_{key}_s{seed}_K{K}.json"
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1,
        objective="multi:softprob" if K > 2 else "binary:logistic",
        tree_method="hist", device="cuda", random_state=seed,
        eval_metric="mlogloss" if K > 2 else "logloss")
    if f.exists():
        clf.load_model(f)
        return clf
    clf.fit(X, y)
    clf.save_model(f)
    return clf


def proba(clf, X):
    P = clf.predict_proba(X)
    return np.asarray(P, dtype=np.float64)


def model_hash(clf) -> bytes:
    return hashlib.sha256(bytes(clf.get_booster().save_raw("json"))).digest()
