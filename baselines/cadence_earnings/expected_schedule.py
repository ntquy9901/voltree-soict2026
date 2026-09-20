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
    path = Path(disclosures_csv) if disclosures_csv else (REPO / "data" / "earnings" / "hose_disclosures.csv")
    d = pd.read_csv(path)
    d = d[d["quarter"].isin(QUARTERS)].copy()
    d["announcement_date"] = pd.to_datetime(d["announcement_date"])
    d = d.sort_values("announcement_date").drop_duplicates(["ticker", "fiscal_year", "quarter"], keep="first")
    return {tk: np.sort(g["announcement_date"].to_numpy()) for tk, g in d.groupby("ticker")}
