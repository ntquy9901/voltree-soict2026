from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "baselines" / "common"))
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
