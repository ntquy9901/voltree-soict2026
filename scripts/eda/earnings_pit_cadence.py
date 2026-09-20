from __future__ import annotations

import numpy as np
import pandas as pd

MIN_HISTORY = 3
THRESHOLDS = (1, 2, 3, 5, 7, 14)

def predict_schedule(dates):
    d = np.sort(np.asarray(dates).astype("datetime64[D]"))
    preds = list(d[:MIN_HISTORY])
    if len(d) > MIN_HISTORY:
        gaps = np.diff(d).astype(int)
        for i in range(MIN_HISTORY, len(d)):
            step = int(np.median(gaps[:i - 1]))
            preds.append(d[i - 1] + np.timedelta64(step, "D"))
    return np.array(preds, dtype="datetime64[D]")

def pit_cadence(edates):
    return {tk: np.sort(predict_schedule(dates)).astype("datetime64[ns]") for tk, dates in edates.items()}

def pit_vs_actual(edates):
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
    if len(disc) == 0:
        return {"n_events": 0, "n_tickers": 0}
    err = disc["abs_err_days"].to_numpy()
    out = {"n_events": int(len(err)), "n_tickers": int(disc["ticker"].nunique()),
           "median_abs_days": float(np.median(err)), "mean_abs_days": float(err.mean()),
           "p90_abs_days": float(np.percentile(err, 90))}
    for t in THRESHOLDS:
        out[f"within_{t}d_pct"] = float(100.0 * np.mean(err <= t))
    return out

def _load_edates(market):
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    fn = "hose_earnings.parquet" if market == "hose" else "sp500_earnings.parquet"
    e = pd.read_parquet(repo / "data" / "earnings" / fn)
    e["earnings_date"] = pd.to_datetime(e["earnings_date"])
    return {tk: g["earnings_date"].to_numpy() for tk, g in e.groupby("ticker")}

def main():
    import json
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    summary = {}
    for market in ("hose", "sp500"):
        disc = pit_vs_actual(_load_edates(market))
        out_csv = repo / "results" / "xgb" / f"earnings_pit_discrepancy_{market}.csv"
        disc.to_csv(out_csv, index=False)
        summary[market] = summarize(disc)
        print(f"{market}: {summary[market]}", flush=True)
        print(f"  wrote {out_csv} ({len(disc)} rows)", flush=True)
    (repo / "results" / "xgb" / "earnings_pit_discrepancy_summary.json").write_text(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
