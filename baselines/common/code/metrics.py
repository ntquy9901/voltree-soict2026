from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

_EPSILON = 1e-8

def _check_pair(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"y and p must have the same shape, got {y.shape} vs {p.shape}")
    if y.size == 0:
        raise ValueError("y and p must be non-empty")
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("y and p must be finite")
    return y, p

def mse(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _check_pair(y, p)
    return float(np.mean((y - p) ** 2))

def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(mse(y, p)))

def mae(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _check_pair(y, p)
    return float(np.mean(np.abs(y - p)))

def r2(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _check_pair(y, p)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)

def per_obs_se(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    y, p = _check_pair(y, p)
    return (y - p) ** 2

def per_obs_qlike(y: np.ndarray, p: np.ndarray, floor: float = _EPSILON) -> np.ndarray:
    if not (np.isfinite(floor) and floor > 0.0):
        raise ValueError(f"floor must be finite and positive, got {floor}")
    y, p = _check_pair(y, p)
    y = np.maximum(y, floor)
    p = np.maximum(p, floor)
    ratio = y / p
    return ratio - np.log(ratio) - 1.0

def qlike(y: np.ndarray, p: np.ndarray, floor: float = _EPSILON) -> float:
    return float(np.mean(per_obs_qlike(y, p, floor=floor)))

@dataclass(frozen=True)
class DMResult:

    dm_hln: float
    p_value: float
    mean_diff: float
    n: int

def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int) -> DMResult:
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("loss_a and loss_b must have the same shape")
    if a.ndim != 1:
        raise ValueError("losses must be one-dimensional per-observation series")
    if isinstance(h, bool) or not float(h).is_integer():
        raise ValueError(f"horizon h must be an integer, got {h!r}")
    h = int(h)
    if h < 1:
        raise ValueError("horizon h must be >= 1")
    n = a.size
    if n < 2:
        raise ValueError("Diebold-Mariano requires at least two observations")
    if h >= n:
        raise ValueError(f"horizon h must be < n (HLN factor / HAC lag undefined for h >= n), got h={h}, n={n}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("losses must be finite")

    d = a - b
    mean_diff = float(d.mean())
    dev = d - mean_diff
    max_lag = h - 1

    gamma0 = float(np.dot(dev, dev) / n)
    long_run = gamma0
    for lag in range(1, max_lag + 1):
        gamma = float(np.dot(dev[lag:], dev[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1)
        long_run += 2.0 * weight * gamma

    if long_run <= 0.0:
        raise ValueError("non-positive long-run variance; DM statistic undefined")

    dm_stat = mean_diff / np.sqrt(long_run / n)
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = float(dm_stat * hln_factor)
    p_value = float(2.0 * stats.t.cdf(-abs(dm_hln), df=n - 1))

    return DMResult(dm_hln=dm_hln, p_value=p_value, mean_diff=mean_diff, n=n)
