import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[1]
for _p in (str(REPO), str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "common"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config
import full_matrix as FM
import vn_gbm_graph_stage1 as S1
import metrics as M
import stats as ST
import paper_metrics_sp500 as PM

FL = FM.FL
OWN = config.own_set(FM.OWN)

def dm_matrix(err, dates, h):
    out = {}
    for a, b in itertools.combinations(err, 2):
        out[f"{a}_vs_{b}"] = float(ST.date_clustered_dm(err[a], err[b], dates, h)["p_value"])
    return out

def _checkpoint(out, out_path):
    if out_path is None:
        return
    tmp = Path(str(out_path) + ".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(out_path)
    print(f"[checkpoint] wrote {out_path.name} ({len(out)} horizons)", flush=True)

def run(market, load_fn=None, out_path=None):
    load_fn = load_fn or FM.load
    min_rows = config.MIN_ROWS.get(market, config.MIN_ROWS["default"])
    frames, sect, edates = load_fn(market)
    if market != "sp500":
        ep = REPO / "data" / "earnings" / "hose_earnings.parquet"
        if ep.exists():
            e = pd.read_parquet(ep)
            edates = {tk: np.sort(g["earnings_date"].to_numpy()) for tk, g in e.groupby("ticker")}
    has_earn = bool(edates)
    out = {}
    for h in config.HORIZONS:
        a = FM.panel(frames, edates, h)
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        tickers = sorted(a["ticker"].unique())
        n = len(tickers)
        Wu, Ws = FM._uniform(n), PM.adj_sector(tickers, sect)
        models = ["HAR", "GBM", "GBM+market", "GBM+corr", "GBM+sector", "GBM+plac", "GBM+oracle"]
        if has_earn:
            models = models[:6] + ["GBM+earn", "GBM+earn+corr"] + models[6:]
        preds = {m: [] for m in models}
        yy, dts = [], []
        for k in range(len(S1.FOLDS) - 1):
            ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
            tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
            te = a[(a.date >= ts) & (a.date < tend)]
            if len(te) == 0 or len(tr) < min_rows:
                continue
            Wc, _ = S1.build_graph(tr, tickers, np.random.default_rng(S1.RNG_SEED + k))
            rng = np.random.default_rng(11 + k)
            Wp = np.zeros((n, n))
            for i in range(n):
                j = rng.choice(np.delete(np.arange(n), i), size=min(S1.TOPK, n - 1), replace=False)
                Wp[i, j] = 1.0 / len(j)
            fold = a[(a.date >= S1.TRAIN_START) & (a.date < tend)].copy()
            fold["g_market"] = FM.nb(fold, tickers, Wu); fold["g_corr"] = FM.nb(fold, tickers, Wc)
            fold["g_sector"] = FM.nb(fold, tickers, Ws); fold["g_plac"] = FM.nb(fold, tickers, Wp)
            fold["g_oracle"] = PM.nb_col(fold, tickers, Wc, "y")
            for c in ("g_market", "g_corr", "g_sector", "g_plac", "g_oracle"):
                fold[c] = fold[c].fillna(0.0)
            trf = fold[(fold.date >= S1.TRAIN_START) & (fold.date < ts - embargo)]
            tef = fold[(fold.date >= ts) & (fold.date < tend)]
            cols = {"GBM": OWN, "GBM+market": OWN + ["g_market"],
                    "GBM+corr": OWN + ["g_corr"], "GBM+sector": OWN + ["g_sector"],
                    "GBM+plac": OWN + ["g_plac"], "GBM+oracle": OWN + ["g_oracle"]}
            if has_earn:
                cols["GBM+earn"] = OWN + FM.EARN; cols["GBM+earn+corr"] = OWN + FM.EARN + ["g_corr"]
            preds["HAR"].append(FM._har_ols(trf, tef))
            for m, cc in cols.items():
                preds[m].append(np.mean([FM.gbm(trf, tef, cc, s) for s in FM.SEEDS], 0))
            yy.append(tef["y"].to_numpy(float)); dts.append(tef["date"].to_numpy())
        if not yy:
            continue
        y = np.concatenate(yy); dates = np.concatenate(dts)
        err = {m: M.per_obs_qlike(y, np.concatenate(preds[m]), floor=FL) for m in models}
        met = {m: PM.all_metrics(y, np.concatenate(preds[m])) for m in models}
        out[f"h{h}"] = {"n": int(len(y)), "metrics": met, "dm_qlike_matrix": dm_matrix(err, dates, h)}
        _checkpoint(out, out_path)
    return out

def _print(market, out):
    for h, r in out.items():
        print(f"\n=== {market} {h} (n={r['n']:,}) ===", flush=True)
        for m, mm in sorted(r["metrics"].items(), key=lambda kv: kv[1]["qlike"]):
            print(f"  {m:16s} QLIKE {mm['qlike']:.4f} R2 {mm['r2']:+.3f}", flush=True)

def main():
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    outp = REPO / "results" / "xgb" / f"full_compare_{market}.json"
    out = run(market, out_path=outp)
    _print(market, out)
    print(f"\nsaved {outp.relative_to(REPO)}", flush=True)

if __name__ == "__main__":
    main()
