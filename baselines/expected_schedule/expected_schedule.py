from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MIN_HISTORY = 3
QUARTERS = ("Q1", "Q2", "Q3", "Q4")

def predict_schedule(dates):
    d = np.sort(np.asarray(dates).astype("datetime64[D]"))
    preds = list(d[:MIN_HISTORY])
    if len(d) > MIN_HISTORY:
        gaps = np.diff(d).astype(int)
        for i in range(MIN_HISTORY, len(d)):

            preds.append(d[i - 1] + np.timedelta64(int(np.median(gaps[:i - 1])), "D"))
    return np.array(preds, dtype="datetime64[D]")

def expected_schedule(edates):
    return {tk: np.sort(predict_schedule(dates)[MIN_HISTORY:]).astype("datetime64[ns]")
            for tk, dates in edates.items()}

def hose_quarterly_dates(disclosures_csv=None):
    path = Path(disclosures_csv) if disclosures_csv else (REPO / "data" / "raw" / "vn_earnings" / "hose_disclosures.csv")
    d = pd.read_csv(path)
    d = d[d["quarter"].isin(QUARTERS)].copy()
    d["announcement_date"] = pd.to_datetime(d["announcement_date"])
    d = d.sort_values("announcement_date").drop_duplicates(["ticker", "fiscal_year", "quarter"], keep="first")
    return {tk: np.sort(g["announcement_date"].to_numpy()) for tk, g in d.groupby("ticker")}

def expected_vs_actual(edates):
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
    if len(disc) == 0:
        return {"n_events": 0, "n_tickers": 0}
    err = disc["abs_err_days"].to_numpy()
    out = {"n_events": int(len(err)), "n_tickers": int(disc["ticker"].nunique()),
           "median_abs_days": float(np.median(err)), "mean_abs_days": float(err.mean())}
    for t in (1, 3, 7, 14):
        out[f"within_{t}d_pct"] = float(100.0 * np.mean(err <= t))
    return out
