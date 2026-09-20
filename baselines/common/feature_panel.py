import glob
from pathlib import Path

import numpy as np
import pandas as pd

import metrics as M
import pipeline_config as pc

REPO = Path(__file__).resolve().parents[2]
FL = pc.QLIKE_FLOOR
WK, MO = 5, 22
HAR = ["har_daily", "har_weekly", "har_monthly"]
OWN = HAR + ["rq", "mr_change", "mr_slope5", "mr_slope10", "mr_dev5", "mr_z22"]
EARN = ["earn_prox", "earn_soon", "earn_pre", "earn_post"]
TRAIN_START = "2015-01-01"
FOLDS = ["2022-07-01", "2023-01-01", "2023-07-01", "2024-01-01", "2024-07-01", "2025-01-01",
         "2025-07-01", "2026-01-01", "2100-01-01"]

def _feat(d):
    pk = d["parkinson_variance"].to_numpy(float); lpk = pd.Series(np.log(np.maximum(pk, FL)), index=d.index)
    d["logpk"] = lpk
    d["rq"] = np.sqrt(pd.Series(pk ** 2, index=d.index).rolling(WK, min_periods=1).mean())
    d["mr_change"] = lpk.diff(1); d["mr_slope5"] = (lpk - lpk.shift(WK)) / WK
    d["mr_slope10"] = (lpk - lpk.shift(2 * WK)) / (2 * WK); d["mr_dev5"] = lpk - lpk.rolling(WK).mean()
    d["mr_z22"] = (lpk - lpk.rolling(MO).mean()) / (lpk.rolling(MO).std() + FL)
    return d

def load(market):
    d = "sp500_clean" if market == "sp500" else market
    frames = {}
    for p in glob.glob(str(REPO / "data" / "processed_enriched" / d / "*.csv")):
        tk = Path(p).stem
        if tk.endswith("_rejections"):
            continue
        fr = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        frames[tk] = _feat(fr).assign(ticker=tk, sector=-1)
    edates = {}
    if market == "sp500":
        e = pd.read_parquet(REPO / "data" / "earnings" / "sp500_earnings.parquet")
        edates = {tk: np.sort(g["earnings_date"].to_numpy()) for tk, g in e.groupby("ticker")}
    return frames, {}, edates

def _signed(T, ed):
    n = len(T)
    if ed is None or len(ed) == 0:
        return np.full(n, 1e9), np.full(n, 1e9)
    tv = pd.DatetimeIndex(pd.to_datetime(T)).asi8 / 86_400_000_000_000.0
    ev = np.sort(pd.DatetimeIndex(pd.to_datetime(ed)).asi8 / 86_400_000_000_000.0)
    idx = np.searchsorted(ev, tv, side="left")
    nxt = np.where(idx < len(ev), ev[np.clip(idx, 0, len(ev) - 1)] - tv, 1e9)
    prv = np.where(idx > 0, tv - ev[np.clip(idx - 1, 0, len(ev) - 1)], 1e9)
    return np.maximum(nxt, 0.0), np.maximum(prv, 0.0)

def panel(frames, edates, h):
    rows = []
    for tk, d in frames.items():
        e = d.copy(); e["y"] = e["parkinson_variance"].shift(-h)
        if edates:
            T = e["date"].shift(-h).to_numpy(); nxt, prv = _signed(T, edates.get(tk)); dist = np.minimum(nxt, prv)
            e["earn_prox"] = np.maximum(0.0, 1.0 - dist / WK); e["earn_soon"] = (dist <= 3).astype(float)
            e["earn_pre"] = np.maximum(0.0, 1.0 - nxt / WK); e["earn_post"] = np.maximum(0.0, 1.0 - prv / (2 * WK))
        rows.append(e)
    a = pd.concat(rows, ignore_index=True)
    return a.dropna(subset=OWN + ["y"]).reset_index(drop=True)

def _har_ols(tr, te):
    x = lambda df: np.column_stack([np.ones(len(df)), df[HAR].to_numpy(float)])
    c = np.linalg.lstsq(x(tr), np.maximum(tr["y"].to_numpy(float), FL), rcond=None)[0]
    return np.maximum(x(te) @ c, 0.01 * np.maximum(tr["y"], FL).mean())

def all_metrics(y, p):
    return {"mse": M.mse(y, p), "rmse": M.rmse(y, p), "mae": M.mae(y, p),
            "r2": M.r2(y, p), "qlike": float(np.mean(M.per_obs_qlike(y, p, floor=FL)))}
