"""
Split-conformal certificates for PCSP.

Scores (smaller = more conforming):
  LAC  s(x,y) = 1 - p_y(x)                       (Sadinle et al. 2019)
  APS  s(x,y,u) = sum_{y': p_y' > p_y} p_y' + u p_y   (Romano et al. 2020,
       randomized; u ~ U[0,1] drawn once per example)
Mondrian (class-conditional) variants calibrate one threshold per class.

All functions operate on probability matrices; `quantize` maps p to the
8-bit wire representation used by SCORE-mode packets, and calibration
applied to quantized scores keeps the finite-sample guarantee because
the same deterministic map is applied to calibration and test points.
"""

from __future__ import annotations

import numpy as np
from numba import njit


def quantize(p: np.ndarray, levels: int = 255) -> np.ndarray:
    """Deterministic rounding of probabilities to `levels` steps (uint8)."""
    return np.clip(np.rint(p * levels), 0, levels).astype(np.uint8)


def dequantize(qp: np.ndarray, levels: int = 255) -> np.ndarray:
    return qp.astype(np.float64) / levels


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """k-th smallest calibration score, k = ceil((m+1)(1-alpha)).
    Returns +inf when k > m (the set must then be the full label set)."""
    m = len(scores)
    if alpha <= 0:
        return np.inf
    if alpha >= 1:
        return -np.inf
    k = int(np.ceil((m + 1) * (1 - alpha)))
    if k > m:
        return np.inf
    return float(np.partition(scores, k - 1)[k - 1])


# ----------------------------------------------------------------- scores

def lac_scores(P: np.ndarray) -> np.ndarray:
    """Score matrix S[i, y] for every candidate label."""
    return 1.0 - P


def aps_scores(P: np.ndarray, U: np.ndarray) -> np.ndarray:
    n, K = P.shape
    S = np.empty_like(P, dtype=np.float64)
    for y in range(K):
        py = P[:, [y]]
        S[:, y] = (P * (P > py)).sum(axis=1) + U * P[:, y]
    return S


def score_matrix(P, method, U=None):
    if method == "lac":
        return lac_scores(P)
    if method == "aps":
        return aps_scores(P, U)
    raise ValueError(method)


# -------------------------------------------------------------- calibrate

class SplitConformal:
    def __init__(self, method="lac", mondrian=False, alpha=0.1, seed=0,
                 nonempty=True):
        self.method, self.mondrian, self.alpha = method, mondrian, alpha
        self.nonempty = nonempty
        self.rng = np.random.default_rng(seed)

    def _S(self, P):
        U = self.rng.random(len(P)) if self.method == "aps" else None
        return score_matrix(P, self.method, U)

    def fit(self, P_cal, y_cal):
        S = self._S(P_cal)
        s = S[np.arange(len(y_cal)), y_cal]
        self.cal_scores = s
        self.cal_labels = y_cal
        self.K = P_cal.shape[1]
        if self.mondrian:
            self.qhat = np.array([conformal_quantile(s[y_cal == k], self.alpha)
                                  for k in range(self.K)])
        else:
            self.qhat = np.full(self.K, conformal_quantile(s, self.alpha))
        return self

    def sets(self, P):
        """Boolean membership matrix (n x K)."""
        S = self._S(P)
        M = S <= self.qhat[None, :]
        if self.nonempty:          # adding the top-1 label only enlarges sets
            M[np.arange(len(P)), P.argmax(axis=1)] = True
        return M


# -------------------------------------------------------------- metrics

def set_metrics(M: np.ndarray, y: np.ndarray, z: np.ndarray | None = None):
    n, K = M.shape
    cov = M[np.arange(n), y]
    size = M.sum(axis=1)
    out = dict(coverage=float(cov.mean()), mean_size=float(size.mean()),
               singleton=float((size == 1).mean()),
               empty=float((size == 0).mean()))
    if z is not None:
        sing = size == 1
        out["singleton_err"] = float((sing & ~cov).mean())
        out["top1_acc"] = float((z == y).mean())
    cc = []
    for k in range(K):
        if (y == k).sum() > 0:
            cc.append(float(cov[y == k].mean()))
    out["min_class_cov"] = float(min(cc))
    out["class_cov"] = cc
    return out


# -------------------------------------------------------------- ACI

def aci_stream(S_stream: np.ndarray, y: np.ndarray, cal_scores: np.ndarray,
               alpha: float, gamma: float, audit: np.ndarray,
               delay: int = 0, alpha0: float | None = None,
               lost: np.ndarray | None = None, impute: bool = True):
    """Audit-driven adaptive conformal inference at the receiver.

    S_stream : (T, K) score matrix computed from the received certificate
    y        : (T,) labels (only consulted at audited steps, after `delay`)
    audit    : (T,) Bernoulli(q) audit indicators drawn by the receiver
    Update   : alpha_{t+1} = alpha_t + gamma * A_{t-d} * (alpha - err_{t-d})
    (labels return in FIFO order, d steps after the audited claim).
    The top-1 label is always kept in the set (non-empty sets).

    `lost` marks audits whose opening never arrives.  With `impute` the
    receiver counts such an audit as an error iff its set was not the full
    label set (alpha_u > 0), which keeps Theorem 3 valid; otherwise the
    lost audit is skipped (biased when the adversary chooses what to drop).

    Returns per-step miscoverage indicators, set sizes and alpha_t trace.
    """
    a0 = alpha if alpha0 is None else alpha0
    fb = np.zeros(len(y), dtype=np.int8)
    if lost is not None:
        fb[np.asarray(lost, bool)] = 1 if impute else 2
    return _aci(np.ascontiguousarray(S_stream, dtype=np.float64),
                np.ascontiguousarray(y, dtype=np.int64),
                np.sort(np.asarray(cal_scores, dtype=np.float64)),
                float(alpha), float(gamma),
                np.ascontiguousarray(audit, dtype=np.bool_), int(delay),
                float(a0), fb)


@njit(cache=True)
def _aci(S, y, cal_sorted, alpha, gamma, audit, delay, a, fb):
    T, K = S.shape
    m = cal_sorted.shape[0]
    err = np.zeros(T, dtype=np.bool_)
    size = np.zeros(T, dtype=np.int16)
    alphas = np.empty(T)
    for t in range(T):
        alphas[t] = a
        if a <= 0.0:
            qh = np.inf
        elif a >= 1.0:
            qh = -np.inf
        else:
            k = int(np.ceil((m + 1) * (1.0 - a)))
            qh = np.inf if k > m else cal_sorted[k - 1]
        top = 0
        for j in range(1, K):
            if S[t, j] < S[t, top]:
                top = j
        n = 0
        covered = False
        for j in range(K):
            if S[t, j] <= qh or j == top:
                n += 1
                if j == y[t]:
                    covered = True
        err[t] = not covered
        size[t] = n
        s = t - delay
        if s >= 0 and audit[s]:
            if fb[s] == 0:
                a = a + gamma * (alpha - (1.0 if err[s] else 0.0))
            elif fb[s] == 1:
                a = a + gamma * (alpha - (1.0 if alphas[s] > 0.0 else 0.0))
    return err, size, alphas


# ------------------------------------------------- ACI, causal schedule

FB_LABEL, FB_TIMEOUT, FB_SKIP, FB_TIMEOUT_SET = 0, 1, 2, 3


def aci_sched(S_stream, y, cal_scores, alpha, gamma, audit, res_time, fb,
              y_fb=None, alpha0=None):
    """Audit-driven ACI with an explicit event schedule (Algorithm 1).

    audit    : (T,) audit coins of the accepted claims
    res_time : (T,) accepted-claim index at which audit u resolves (its
               label returns or its deadline passes); updates are applied
               in FIFO order of u once t >= res_time[u]
    fb       : (T,) FB_LABEL     label returned (error w.r.t. y_fb)
                    FB_TIMEOUT   opening missing: e_u = 1{alpha_u > 0}
                    FB_TIMEOUT_SET opening missing: e_u = 1{C_u != Y}
                    FB_SKIP      ignored (biased; for comparison only)
    y_fb     : labels returned by the labeling service (default: y); the
               reported miscoverage is always w.r.t. the true y.
    Returns (err, size, alphas, D) with D the largest number of audits
    outstanding at any time, counting the current one."""
    a0 = alpha if alpha0 is None else alpha0
    y = np.ascontiguousarray(y, dtype=np.int64)
    yf = y if y_fb is None else np.ascontiguousarray(y_fb, dtype=np.int64)
    return _aci_sched(np.ascontiguousarray(S_stream, dtype=np.float64), y, yf,
                      np.sort(np.asarray(cal_scores, dtype=np.float64)),
                      float(alpha), float(gamma),
                      np.ascontiguousarray(audit, dtype=np.bool_),
                      np.ascontiguousarray(res_time, dtype=np.int64),
                      np.ascontiguousarray(fb, dtype=np.int8), float(a0))


@njit(cache=True)
def _aci_sched(S, y, yf, cal_sorted, alpha, gamma, audit, res_time, fb, a):
    T, K = S.shape
    m = cal_sorted.shape[0]
    err = np.zeros(T, dtype=np.bool_)
    size = np.zeros(T, dtype=np.int16)
    alphas = np.empty(T)
    errf = np.zeros(T, dtype=np.bool_)
    q = np.empty(T, dtype=np.int64)        # FIFO of audited indices
    head, tail, D = 0, 0, 0
    for t in range(T):
        alphas[t] = a
        if a <= 0.0:
            qh = np.inf
        elif a >= 1.0:
            qh = -np.inf
        else:
            k = int(np.ceil((m + 1) * (1.0 - a)))
            qh = np.inf if k > m else cal_sorted[k - 1]
        top = 0
        for j in range(1, K):
            if S[t, j] < S[t, top]:
                top = j
        n = 0
        cov = False
        covf = False
        for j in range(K):
            if S[t, j] <= qh or j == top:
                n += 1
                if j == y[t]:
                    cov = True
                if j == yf[t]:
                    covf = True
        err[t] = not cov
        errf[t] = not covf
        size[t] = n
        if audit[t]:
            q[tail] = t
            tail += 1
        if tail - head > D:
            D = tail - head
        while head < tail and res_time[q[head]] <= t:
            u = q[head]
            head += 1
            f = fb[u]
            if f == 0:
                a = a + gamma * (alpha - (1.0 if errf[u] else 0.0))
            elif f == 1:
                a = a + gamma * (alpha - (1.0 if alphas[u] > 0.0 else 0.0))
            elif f == 3:
                a = a + gamma * (alpha - (1.0 if size[u] < K else 0.0))
    return err, size, alphas, D
