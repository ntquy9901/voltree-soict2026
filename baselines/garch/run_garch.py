import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[1]
for _p in (str(REPO / "baselines" / "common"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config
import feature_panel as D
import metrics as M
import garch_model as GM

FL = D.FL

def _int_dates(values):
    return np.asarray(values, dtype="datetime64[ns]").astype("int64")

def _ticker_series(frames):
    series = {}
    for tk, d in frames.items():
        s = d[["date", "daily_return"]].dropna(subset=["daily_return"]).sort_values("date")
        di = _int_dates(s["date"].to_numpy())
        series[tk] = (di, s["daily_return"].to_numpy(float), {int(v): i for i, v in enumerate(di)})
    return series

def _ticker_task(payload):
    returns, jobs = payload
    out = []
    for (h, k, n_train, ti, trows, tri, trrows) in jobs:
        p = GM.fit_params(returns[:n_train], "garch")
        g_test = GM.forecast(returns, p, "garch", ti, h)
        g_train = GM.forecast(returns, p, "garch", tri, h) if tri is not None else None
        out.append((h, k, trows, g_test, p.ok, trrows, g_train))
    return out

def _active_folds(a, embargo, min_rows):
    folds = []
    for k in range(len(D.FOLDS) - 1):
        ts, tend = pd.Timestamp(D.FOLDS[k]), pd.Timestamp(D.FOLDS[k + 1])
        tr = a[(a.date >= D.TRAIN_START) & (a.date < ts - embargo)]
        te = a[(a.date >= ts) & (a.date < tend)].reset_index(drop=True)
        if len(te) == 0 or len(tr) < min_rows:
            continue
        folds.append((k, ts, tend, tr.reset_index(drop=True), te))
    return folds

def _idx(pos, dates):
    return np.array([pos[int(x)] for x in _int_dates(dates)], dtype=int)

def _estimable(series, tickers, cutoff, min_obs):
    return {tk for tk in tickers
            if int(np.searchsorted(series[tk][0], cutoff, side="left")) >= min_obs}

def _build_jobs(series, folds, embargo, h, last_k, min_obs):
    jobs = {tk: [] for tk in series}
    te_sizes = {k: len(te) for (k, _, _, _, te) in folds}
    ts_last, tr_last = next((ts, tr) for (k, ts, tend, tr, te) in folds if k == last_k)
    est_tr = _estimable(series, tr_last["ticker"].unique(), int((ts_last - embargo).value), min_obs)
    tr_scored = tr_last[tr_last["ticker"].isin(est_tr)].reset_index(drop=True)
    tr_g = {tk: sub for tk, sub in tr_scored.groupby("ticker")}
    for (k, ts, tend, tr, te) in folds:
        cutoff = int((ts - embargo).value)
        te_g = {tk: sub for tk, sub in te.groupby("ticker")}
        tickers = set(te_g) | (set(tr_g) if k == last_k else set())
        for tk in tickers:
            di, _, pos = series[tk]
            n_train = int(np.searchsorted(di, cutoff, side="left"))
            sub = te_g.get(tk)
            ti = _idx(pos, sub["date"].to_numpy()) if sub is not None else np.empty(0, dtype=int)
            trows = sub.index.to_numpy() if sub is not None else np.empty(0, dtype=int)
            tri = trrows = None
            if k == last_k:
                subtr = tr_g.get(tk)
                tri = _idx(pos, subtr["date"].to_numpy()) if subtr is not None else np.empty(0, dtype=int)
                trrows = subtr.index.to_numpy() if subtr is not None else np.empty(0, dtype=int)
            jobs[tk].append((h, k, n_train, ti, trows, tri, trrows))
    return jobs, te_sizes, tr_scored

def _dispatch(series, jobs, n_jobs):
    payloads = [(series[tk][1], jobs[tk]) for tk in series if jobs[tk]]
    if n_jobs == 1:
        results = [_ticker_task(pl) for pl in payloads]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(_ticker_task, payloads))
    return [row for chunk in results for row in chunk]

def _checkpoint(out, out_path):
    if out_path is None:
        return
    tmp = Path(str(out_path) + ".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(out_path)
    print(f"[checkpoint] wrote {out_path.name} ({len(out)} horizons)", flush=True)

def run_garch(market, load_fn=None, n_jobs=1, out_path=None):
    load_fn = load_fn or D.load
    min_rows = config.MIN_ROWS.get(market, config.MIN_ROWS["default"])
    frames, _, _ = load_fn(market)
    series = _ticker_series(frames)
    out = {}
    min_obs = config.MIN_TRAIN_OBS
    for h in config.HORIZONS:
        a = D.panel(frames, {}, h)
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        folds = _active_folds(a, embargo, min_rows)
        n_excluded = 0
        kept = []
        for (k, ts, tend, tr, te) in folds:
            est = _estimable(series, te["ticker"].unique(), int((ts - embargo).value), min_obs)
            keep = te["ticker"].isin(est)
            n_excluded += int((~keep).sum())
            te_s = te[keep].reset_index(drop=True)
            if len(te_s):
                kept.append((k, ts, tend, tr, te_s))
        folds = kept
        if not folds:
            continue
        last_k = folds[-1][0]
        jobs, te_sizes, tr_scored = _build_jobs(series, folds, embargo, h, last_k, min_obs)
        rows = _dispatch(series, jobs, n_jobs)

        g_test = {k: np.full(n, np.nan) for k, n in te_sizes.items()}
        g_train = np.full(len(tr_scored), np.nan)
        n_fb = 0
        for (rh, k, trows, gt, g_ok, trrows, gtr) in rows:
            g_test[k][trows] = gt
            n_fb += int(not g_ok)
            if trrows is not None:
                g_train[trrows] = gtr

        y_p, g_p = [], []
        for (k, ts, tend, tr, te) in folds:
            g_p.append(g_test[k])
            y_p.append(te["y"].to_numpy(float))
        y = np.concatenate(y_p)
        g = np.concatenate(g_p)
        if np.isnan(g).any():
            raise ValueError(f"unfilled forecast rows at h={h} (ticker/date misalignment)")
        q = float(np.mean(M.per_obs_qlike(y, g, floor=FL)))
        y_tr = tr_scored["y"].to_numpy(float)
        tr_q = float(np.mean(M.per_obs_qlike(y_tr, g_train, floor=FL)))
        out[f"h{h}"] = {
            "n": int(len(y)),
            "n_excluded": int(n_excluded),
            "qlike": {"GARCH": q},
            "mse": {"GARCH": M.mse(y, g)},
            "rmse": {"GARCH": M.rmse(y, g)},
            "mae": {"GARCH": M.mae(y, g)},
            "n_fallback": {"GARCH": n_fb},
            "train_metrics": {"GARCH": tr_q},
            "fit_diagnostics": {"GARCH": {"verdict": "overfit" if q > tr_q * 1.25 else "ok",
                                          "train_qlike": tr_q, "test_qlike": q}},
        }
        _checkpoint(out, out_path)
    return out

def main():
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    n_jobs = max(1, (os.cpu_count() or 2) - 1)
    outp = REPO / "results" / "xgb" / f"garch_{market}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    out = run_garch(market, n_jobs=n_jobs, out_path=outp)
    for h, r in out.items():
        print(f"  {market} {h} (n={r['n']:,}) GARCH QLIKE {r['qlike']['GARCH']:.4f}", flush=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"saved {outp.relative_to(REPO)}", flush=True)

if __name__ == "__main__":
    main()
