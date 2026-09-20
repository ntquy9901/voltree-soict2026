"""Panel-aware statistical comparison of forecast losses (plan section 18).

The residual-study panel is indexed by ``(ticker_id, target_date)``: many tickers share the
same ``target_date``, so per-observation loss differentials are cross-sectionally dependent and a
naive Diebold-Mariano over all rows treats ``n_rows`` as the sample size, understating the variance.
Every test here first collapses the per-observation loss differential to ONE value per unique date
(cross-sectional mean), removing that dependence; residual serial dependence at horizon ``h`` is then
handled by the HLN long-run-variance lag (``h - 1``) or a circular block bootstrap on the date series.

Reuses the tested Diebold-Mariano implementation from ``submission/soict_lstm_gat/metrics.py``
(``diebold_mariano`` with the Harvey-Leybourne-Newbold 1997 small-sample correction). It is NOT
reimplemented here.

Pure numpy + scipy (via the reused DM). RNG is always a local ``np.random.default_rng(seed)`` — no
global RNG state.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"))
import metrics as _metrics  # noqa: E402  (reuse the tested DM/HLN implementation)


def _aggregate_by_date(values: np.ndarray, dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse ``values`` to the mean within each unique date.

    Returns ``(unique_dates_sorted, mean_per_date)``. ``np.unique`` yields dates in chronological
    order (dates are sortable ints/datetimes), giving a time-ordered aggregated series.
    """
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
    """Date-clustered Diebold-Mariano on a dependent panel.

    Aggregates each loss series to one value per unique date (cross-sectional mean), then runs the
    reused HLN Diebold-Mariano on the date-level series. Because the mean is linear,
    ``mean(loss_a) - mean(loss_b)`` per date equals the mean loss differential per date, so this is
    exactly DM on the date-aggregated differential. ``mean_diff < 0`` favours model A (smaller loss).

    Returns ``{"dm_hln", "p_value", "mean_diff", "n_dates"}``.
    """
    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    if loss_a.shape != loss_b.shape:                 # both aggregate over the SAME `dates` array; require
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
    """Return an ``[n_boot, n]`` array of circular block-bootstrap row indices into a length-``n``
    series. Each resample concatenates ``ceil(n / block)`` contiguous wrap-around blocks and is
    truncated to length ``n``."""
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n      # [n_boot, n_blocks, block]
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
    """Circular block-bootstrap confidence interval for the mean loss differential.

    The per-observation differential ``loss_a - loss_b`` is aggregated to one value per unique date
    (length ``T``); the circular block bootstrap (Politis-Romano) then resamples contiguous
    wrap-around blocks of the date series to preserve short-range serial dependence. Default block
    length ``ceil(T ** (1/3))``. Returns the observed ``mean_diff`` and the
    ``[alpha/2, 1 - alpha/2]`` percentile interval of the resampled means; ``significant`` is True
    when the interval excludes 0.
    """
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
    """Model Confidence Set (Hansen, Lunde & Nason 2011, Econometrica) — ``T_max`` elimination variant.

    Each model's per-observation loss is aggregated to one value per unique date, giving a ``[T, M]``
    matrix ``L``. Per round, losses are centered per date about the current set's cross-model mean
    (``r_{t,i} = L_{t,i} - mean_i L_{t,i}``); ``rbar_i = mean_t r_{t,i}`` is model ``i``'s relative
    loss. A circular block bootstrap on the date index (block ``ceil(T ** (1/3))``, indices drawn once
    and reused across rounds) gives the bootstrap variance ``var_i`` of ``rbar_i`` and the studentized
    statistics ``t_i = rbar_i / sqrt(var_i)``. The range/elimination statistic is
    ``T_max = max_i t_i`` (the model with the largest positive relative loss is the worst); its null
    distribution is ``max_i (rbar_i^*b - rbar_i) / sqrt(var_i)``. If ``p = P(T_max^* >= T_max) < alpha``
    the worst model is eliminated and the loop repeats on the survivors, otherwise all survivors form
    the MCS. Reported per-model p-value is the running maximum of the round p-values up to elimination
    (the MCS p-value); survivors carry the final (non-rejecting) running maximum.

    Caveat: this is the standard ``T_max`` elimination MCS with a bootstrap-variance studentization,
    not the full quadratic (``T_Q``) statistic; single fixed split, no walk-forward. Returns
    ``{"mcs_set": [names kept], "p_values": {name: elimination p-value}}``.
    """
    names = list(loss_dict.keys())
    if len(names) < 1:
        raise ValueError("model_confidence_set requires at least one model")
    cols = [_aggregate_by_date(loss_dict[name], dates)[1] for name in names]
    loss_mat = np.column_stack(cols)                     # [T, M]
    t = loss_mat.shape[0]
    if t < 2:
        raise ValueError("MCS requires at least two distinct dates")

    block = math.ceil(t ** (1.0 / 3.0))
    rng = np.random.default_rng(seed)
    boot_idx = _circular_block_indices(t, block, n_boot, rng)   # [n_boot, T] reused every round

    alive = list(range(len(names)))
    p_values: dict[str, float] = {}
    running_max_p = 0.0
    _tiny = 1e-12

    while len(alive) > 1:
        sub = loss_mat[:, alive]                          # [T, m]
        centered = sub - sub.mean(axis=1, keepdims=True)  # per-date de-meaning
        rbar = centered.mean(axis=0)                      # [m]
        boot_rbar = centered[boot_idx].mean(axis=1)       # [n_boot, m]
        zeta = boot_rbar - rbar[None, :]                  # [n_boot, m], centered bootstrap
        var = (zeta ** 2).mean(axis=0)                    # [m] bootstrap variance
        safe = var > _tiny
        t_stat = np.where(safe, rbar / np.sqrt(np.where(safe, var, 1.0)), 0.0)
        t_boot = np.where(safe[None, :], zeta / np.sqrt(np.where(safe, var, 1.0))[None, :], 0.0)
        t_max = float(t_stat.max())
        t_max_boot = t_boot.max(axis=1)                   # [n_boot]
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
