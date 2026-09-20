"""Strictly point-in-time (PIT) earnings schedule via own-firm reporting cadence + a data-quality check.

Backs the paper's leakage-safe earnings robustness (Limitations): instead of the archived *realized*
announcement date (which assumes exact-date foreknowledge at the forecast origin), predict each release
date from the firm's OWN past reporting rhythm -- the previous release plus the median of its PRIOR
inter-release gaps. Only past releases enter, so no future information leaks.

Functions:
  * ``predict_schedule``  -- index-aligned PIT prediction for one ticker's date array (pred[i] <-> actual[i]).
  * ``pit_cadence``       -- dict ticker -> sorted index-aligned schedule (datetime64[ns]) INCLUDING the
                             first MIN_HISTORY actual anchors; for discrepancy analysis ONLY. It is NOT
                             leakage-safe as a panel feature -- the model path uses
                             ``baselines/2026-09-19_expected_schedule/code/expected_schedule.py::expected_schedule``,
                             which exposes only the causally-predicted dates (index >= MIN_HISTORY).
  * ``pit_vs_actual``     -- per-(ticker, event) discrepancy table |predicted - actual| in days.
  * ``summarize``         -- per-market summary of the discrepancy distribution.
  * ``main``              -- runs the discrepancy check over ALL HOSE + SP500 tickers/history, writes
                             ``results/gamma_gbm/earnings_pit_discrepancy_<market>.csv`` + prints a summary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_HISTORY = 3   # need >=3 prior releases before the cadence median is meaningful
THRESHOLDS = (1, 2, 3, 5, 7, 14)   # day-tolerances reported by summarize


def predict_schedule(dates):
    """Index-aligned strictly-causal PIT prediction for one ticker's release dates.

    Returns an array the same length as the sorted input. The first ``MIN_HISTORY`` entries equal the
    actual dates (no prediction yet); each later entry ``i`` is ``actual[i-1] + median(prior gaps[:i-1])``,
    using only gaps strictly before event ``i`` so the prediction is knowable at the forecast origin.
    ``pred[i]`` corresponds to ``actual[i]`` (no reordering)."""
    d = np.sort(np.asarray(dates).astype("datetime64[D]"))
    preds = list(d[:MIN_HISTORY])
    if len(d) > MIN_HISTORY:
        gaps = np.diff(d).astype(int)
        for i in range(MIN_HISTORY, len(d)):
            step = int(np.median(gaps[:i - 1]))                  # gaps BETWEEN releases before event i (excl. gap TO d[i]) -> causal
            preds.append(d[i - 1] + np.timedelta64(step, "D"))
    return np.array(preds, dtype="datetime64[D]")


def pit_cadence(edates):
    """Map ticker -> sorted index-aligned schedule (``datetime64[ns]``), anchors INCLUDED.

    DISCREPANCY-ANALYSIS ONLY -- do NOT feed this to ``full_matrix.panel``: the first ``MIN_HISTORY``
    entries are the raw actual dates (anchors), and the panel uses the next scheduled date as a
    forward-looking feature, so an anchor would leak a future realized date at an earlier origin. The
    leakage-safe model path is ``expected_schedule`` (baselines/2026-09-19_expected_schedule), which drops
    the anchors."""
    return {tk: np.sort(predict_schedule(dates)).astype("datetime64[ns]") for tk, dates in edates.items()}


def pit_vs_actual(edates):
    """Per-(ticker, event) discrepancy between the PIT-predicted and the actual release date.

    Only events from index ``MIN_HISTORY`` on are emitted (the first three are anchors with no
    prediction). Columns: ticker, event_index, actual_date, predicted_date, abs_err_days, signed_err_days
    (predicted - actual)."""
    rows = []
    for tk, dates in edates.items():
        d = np.sort(np.asarray(dates).astype("datetime64[D]"))
        if len(d) <= MIN_HISTORY:
            continue
        pred = predict_schedule(d)
        for i in range(MIN_HISTORY, len(d)):
            signed = int((pred[i] - d[i]).astype("timedelta64[D]").astype(int))
            rows.append({"ticker": tk, "event_index": i,
                         "actual_date": d[i].astype("datetime64[D]"),
                         "predicted_date": pred[i].astype("datetime64[D]"),
                         "abs_err_days": abs(signed), "signed_err_days": signed})
    return pd.DataFrame(rows, columns=["ticker", "event_index", "actual_date", "predicted_date",
                                       "abs_err_days", "signed_err_days"])


def summarize(disc):
    """Summary of the discrepancy distribution: n events/tickers, coverage within each day-tolerance,
    and central tendency. Returns a dict."""
    if len(disc) == 0:
        return {"n_events": 0, "n_tickers": 0}
    err = disc["abs_err_days"].to_numpy()
    out = {"n_events": int(len(err)), "n_tickers": int(disc["ticker"].nunique()),
           "median_abs_days": float(np.median(err)), "mean_abs_days": float(err.mean()),
           "p90_abs_days": float(np.percentile(err, 90))}
    for t in THRESHOLDS:
        out[f"within_{t}d_pct"] = float(100.0 * np.mean(err <= t))
    return out


def _load_edates(market):  # pragma: no cover - thin data loader (reads the crawled earnings parquet)
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    fn = "hose_earnings_combined.parquet" if market == "hose" else "sp500_earnings.parquet"
    e = pd.read_parquet(repo / "results" / "gamma_gbm" / fn)
    e["earnings_date"] = pd.to_datetime(e["earnings_date"])
    return {tk: g["earnings_date"].to_numpy() for tk, g in e.groupby("ticker")}


def main():  # pragma: no cover - data-driven driver over the full crawled history
    import json
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    summary = {}
    for market in ("hose", "sp500"):
        disc = pit_vs_actual(_load_edates(market))
        out_csv = repo / "results" / "gamma_gbm" / f"earnings_pit_discrepancy_{market}.csv"
        disc.to_csv(out_csv, index=False)
        summary[market] = summarize(disc)
        print(f"{market}: {summary[market]}", flush=True)
        print(f"  wrote {out_csv} ({len(disc)} rows)", flush=True)
    (repo / "results" / "gamma_gbm" / "earnings_pit_discrepancy_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":  # pragma: no cover
    main()
