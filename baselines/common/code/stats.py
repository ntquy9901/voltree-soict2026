from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "baselines" / "common" / "code"))
import metrics as _metrics

def _aggregate_by_date(values: np.ndarray, dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    dates = np.asarray(dates)
    if values.shape[0] != dates.shape[0]:
        raise ValueError("values and dates must have the same length")
    uniq, inverse = np.unique(dates, return_inverse=True)
    sums = np.zeros(uniq.size, dtype=float)
    counts = np.zeros(uniq.size, dtype=float)
    np.add.at(sums, inverse, values)
    np.add.at(counts, inverse, 1.0)
    return uniq, sums / counts

def date_clustered_dm(
    loss_a: np.ndarray, loss_b: np.ndarray, dates: np.ndarray, h: int
) -> dict[str, float]:
    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    if loss_a.shape != loss_b.shape:
        raise ValueError("loss_a and loss_b must have the same shape (aligned per-observation)")
    _, a_by_date = _aggregate_by_date(loss_a, dates)
    _, b_by_date = _aggregate_by_date(loss_b, dates)
    res = _metrics.diebold_mariano(a_by_date, b_by_date, h=h)
    return {
        "dm_hln": res.dm_hln,
        "p_value": res.p_value,
        "mean_diff": res.mean_diff,
        "n_dates": int(a_by_date.size),
    }

def _circular_block_indices(
    n: int, block: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    return idx.reshape(n_boot, n_blocks * block)[:, :n]

def block_bootstrap_ci(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    dates: np.ndarray,
    block: int | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, object]:
    dates_u, d_by_date = _aggregate_by_date(
        np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float), dates
    )
    t = d_by_date.size
    if t < 2:
        raise ValueError("block bootstrap requires at least two distinct dates")
    if block is None:
        block = math.ceil(t ** (1.0 / 3.0))
    block = int(block)
    if not 1 <= block <= t:
        raise ValueError("block length must satisfy 1 <= block <= n_dates")

    rng = np.random.default_rng(seed)
    idx = _circular_block_indices(t, block, n_boot, rng)
    boot_means = d_by_date[idx].mean(axis=1)
    ci_low = float(np.quantile(boot_means, alpha / 2.0))
    ci_high = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return {
        "mean_diff": float(d_by_date.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "block": block,
        "n_boot": int(n_boot),
        "significant": bool(ci_low > 0.0 or ci_high < 0.0),
    }

def model_confidence_set(
    loss_dict: dict[str, np.ndarray],
    dates: np.ndarray,
    alpha: float = 0.10,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, object]:
    names = list(loss_dict.keys())
    if len(names) < 1:
        raise ValueError("model_confidence_set requires at least one model")
    cols = [_aggregate_by_date(loss_dict[name], dates)[1] for name in names]
    loss_mat = np.column_stack(cols)
    t = loss_mat.shape[0]
    if t < 2:
        raise ValueError("MCS requires at least two distinct dates")

    block = math.ceil(t ** (1.0 / 3.0))
    rng = np.random.default_rng(seed)
    boot_idx = _circular_block_indices(t, block, n_boot, rng)

    alive = list(range(len(names)))
    p_values: dict[str, float] = {}
    running_max_p = 0.0
    _tiny = 1e-12

    while len(alive) > 1:
        sub = loss_mat[:, alive]
        centered = sub - sub.mean(axis=1, keepdims=True)
        rbar = centered.mean(axis=0)
        boot_rbar = centered[boot_idx].mean(axis=1)
        zeta = boot_rbar - rbar[None, :]
        var = (zeta ** 2).mean(axis=0)
        safe = var > _tiny
        t_stat = np.where(safe, rbar / np.sqrt(np.where(safe, var, 1.0)), 0.0)
        t_boot = np.where(safe[None, :], zeta / np.sqrt(np.where(safe, var, 1.0))[None, :], 0.0)
        t_max = float(t_stat.max())
        t_max_boot = t_boot.max(axis=1)
        p = float(np.mean(t_max_boot >= t_max))
        running_max_p = max(running_max_p, p)

        if p >= alpha:
            break
        worst_local = int(np.argmax(t_stat))
        worst_global = alive.pop(worst_local)
        p_values[names[worst_global]] = running_max_p

    for g in alive:
        p_values[names[g]] = max(running_max_p, alpha)
    mcs_set = [names[g] for g in alive]
    return {"mcs_set": mcs_set, "p_values": p_values}
