"""Expected-schedule earnings dates: a strictly leakage-safe replacement for realized (actual) dates.

Rationale (see the 2026-09-19 paper): the earnings feature measures distance from the forecast target
``t+h`` to the nearest earnings release. Using the *realized* (ex-post crawled) date assumes the exact
date of a FUTURE release was known at the forecast origin ``t`` -- a point-in-time leak. The
**expected schedule** instead predicts each release date from the firm's OWN past reporting cadence
(previous release + median of its PRIOR inter-release gaps), so every value is knowable at ``t`` and no
future information enters. For HOSE we first reduce the multi-filing disclosure stream to one clean
QUARTERLY event per (ticker, fiscal-year, quarter) -- the earliest disclosure across accounting scopes
(parent / consolidated / combined), which are released essentially the same day -- so the cadence is
regular enough for the prediction to be meaningful (SP500 dates are already quarterly).

Pure, unit-tested helpers; ``expected_schedule`` and ``hose_quarterly_dates`` are what the runner injects.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
MIN_HISTORY = 3                       # need > MIN_HISTORY releases before the cadence median is meaningful
QUARTERS = ("Q1", "Q2", "Q3", "Q4")   # keep only quarterly filings (drop FY / H1) for the HOSE schedule


def predict_schedule(dates):
    """Index-aligned strictly-causal expected schedule for one ticker's release dates.

    First ``MIN_HISTORY`` entries equal the actual dates (anchors, no prediction); each later entry ``i``
    is ``actual[i-1] + median(prior gaps[:i-1])`` using only gaps strictly before event ``i``. ``pred[i]``
    corresponds to ``actual[i]`` (no reordering)."""
    d = np.sort(np.asarray(dates).astype("datetime64[D]"))
    preds = list(d[:MIN_HISTORY])
    if len(d) > MIN_HISTORY:
        gaps = np.diff(d).astype(int)
        for i in range(MIN_HISTORY, len(d)):
            # gaps[:i-1] = intervals BETWEEN releases strictly before event i (d[0..i-1]); gaps[i-1]
            # would be the gap TO event i (needs the future date d[i]) -> excluded to stay leakage-safe.
            preds.append(d[i - 1] + np.timedelta64(int(np.median(gaps[:i - 1])), "D"))
    return np.array(preds, dtype="datetime64[D]")


def expected_schedule(edates):
    """Map ticker -> sorted expected schedule (``datetime64[ns]``) for ``full_matrix.panel``.

    Only the causally-PREDICTED dates (event index ``>= MIN_HISTORY``) are exposed; the first
    ``MIN_HISTORY`` actual anchor dates are deliberately NOT emitted. Reason (integration leak): the panel
    uses the schedule's *next* date after each forecast target as a forward-looking feature, so exposing an
    un-predicted future actual anchor would leak its realized date at any earlier origin. Every exposed date
    is derived purely from prior gaps, so it is knowable at the forecast origin. Tickers with
    ``<= MIN_HISTORY`` observed releases expose an empty schedule (earnings features stay neutral).

    The schedule is built once over the ticker's full history (not re-derived per origin); this stays
    leakage-safe only while the horizon ``h`` is well below the typical inter-release gap (quarterly ~91d
    vs max h=22 trading days), so any scheduled date near a target ``t+h`` is anchored to a release a
    quarter earlier -- comfortably before the origin ``t``. Revisit if ``h`` approaches the earnings cadence."""
    return {tk: np.sort(predict_schedule(dates)[MIN_HISTORY:]).astype("datetime64[ns]")
            for tk, dates in edates.items()}


def hose_quarterly_dates(disclosures_csv=None):
    """One clean quarterly release per (ticker, fiscal-year, quarter): the EARLIEST disclosure across
    accounting scopes, from the SSC disclosure archive. Returns ticker -> sorted actual quarterly dates."""
    path = Path(disclosures_csv) if disclosures_csv else (REPO / "data" / "raw" / "vn_earnings" / "hose_disclosures.csv")
    d = pd.read_csv(path)
    d = d[d["quarter"].isin(QUARTERS)].copy()
    d["announcement_date"] = pd.to_datetime(d["announcement_date"])
    d = d.sort_values("announcement_date").drop_duplicates(["ticker", "fiscal_year", "quarter"], keep="first")
    return {tk: np.sort(g["announcement_date"].to_numpy()) for tk, g in d.groupby("ticker")}


def expected_vs_actual(edates):
    """Per-(ticker, event) discrepancy between the expected-schedule and the actual release date (days).
    Columns: ticker, event_index, actual_date, predicted_date, abs_err_days, signed_err_days."""
    rows = []
    for tk, dates in edates.items():
        d = np.sort(np.asarray(dates).astype("datetime64[D]"))
        if len(d) <= MIN_HISTORY:
            continue
        pred = predict_schedule(d)
        for i in range(MIN_HISTORY, len(d)):
            signed = int((pred[i] - d[i]).astype("timedelta64[D]").astype(int))
            rows.append({"ticker": tk, "event_index": i, "actual_date": d[i].astype("datetime64[D]"),
                         "predicted_date": pred[i].astype("datetime64[D]"), "abs_err_days": abs(signed),
                         "signed_err_days": signed})
    return pd.DataFrame(rows, columns=["ticker", "event_index", "actual_date", "predicted_date",
                                       "abs_err_days", "signed_err_days"])


def summarize(disc):
    """Summary of the expected-vs-actual discrepancy distribution."""
    if len(disc) == 0:
        return {"n_events": 0, "n_tickers": 0}
    err = disc["abs_err_days"].to_numpy()
    out = {"n_events": int(len(err)), "n_tickers": int(disc["ticker"].nunique()),
           "median_abs_days": float(np.median(err)), "mean_abs_days": float(err.mean())}
    for t in (1, 3, 7, 14):
        out[f"within_{t}d_pct"] = float(100.0 * np.mean(err <= t))
    return out
