import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eda"))
import full_matrix as FM
import vn_gbm_graph_stage1 as S1
sys.path.insert(0, str(REPO / "baselines" / "common"))
import metrics as M
import stats as ST

FL = FM.FL

def nb_col(fold, tickers, W, col):
    piv = fold.pivot_table(index="date", columns="ticker", values=col).reindex(columns=tickers).sort_index()
    V = piv.to_numpy(float)
    Vf = np.nan_to_num(np.where(np.isnan(V), np.nanmean(V, axis=1, keepdims=True), V))
    NB = Vf @ W.T
    dpos = {d: i for i, d in enumerate(piv.index)}
    cpos = {c: j for j, c in enumerate(tickers)}
    return NB[fold["date"].map(dpos).to_numpy(), fold["ticker"].map(cpos).to_numpy()]

def all_metrics(y, p):
    return {"mse": M.mse(y, p), "rmse": M.rmse(y, p), "mae": M.mae(y, p),
            "r2": M.r2(y, p), "qlike": float(np.mean(M.per_obs_qlike(y, p, floor=FL)))}

def adj_sector(tickers, sect):
    n = len(tickers)
    W = np.zeros((n, n))
    s = [sect.get(t, -1) for t in tickers]
    for i in range(n):
        js = [j for j in range(n) if j != i and s[j] == s[i]]
        for j in js:
            W[i, j] = 1.0 / len(js)
    return W

def main():
    market = sys.argv[1] if len(sys.argv) > 1 else "sp500"
    min_rows = 30000 if market == "sp500" else 3000
    frames, sect, edates = FM.load(market)
    if market != "sp500":
        _ep = REPO / "data" / "earnings" / "hose_earnings.parquet"
        if _ep.exists():
            _e = pd.read_parquet(_ep)
            edates = {tk: np.sort(g["earnings_date"].to_numpy()) for tk, g in _e.groupby("ticker")}
    has_earn = bool(edates)
    out = {}
    for h in (1, 5, 10, 22):
        a = FM.panel(frames, edates, h)
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        tickers = sorted(a["ticker"].unique())
        n = len(tickers)
        Wu, Ws = FM._uniform(n), adj_sector(tickers, sect)
        models = ["HAR", "HARQ", "GBM", "GBM+market", "GBM+corr", "GBM+sector", "GBM+plac", "GBM+oracle"]
        if has_earn:
            models = models[:7] + ["GBM+earn", "GBM+earn+corr"] + models[7:]
        preds = {m: [] for m in models}
        yy, dts = [], []
        per_fold = {m: [] for m in (["GBM", "GBM+market", "GBM+corr"] + (["GBM+earn"] if has_earn else []))}
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
            fold["g_market"] = FM.nb(fold, tickers, Wu)
            fold["g_corr"] = FM.nb(fold, tickers, Wc)
            fold["g_sector"] = FM.nb(fold, tickers, Ws)
            fold["g_plac"] = FM.nb(fold, tickers, Wp)
            fold["g_oracle"] = nb_col(fold, tickers, Wc, "y")
            for c in ("g_market", "g_corr", "g_sector", "g_plac", "g_oracle"):
                fold[c] = fold[c].fillna(0.0)
            trf = fold[(fold.date >= S1.TRAIN_START) & (fold.date < ts - embargo)]
            tef = fold[(fold.date >= ts) & (fold.date < tend)]
            cols = {"GBM": FM.OWN, "GBM+market": FM.OWN + ["g_market"], "GBM+corr": FM.OWN + ["g_corr"],
                    "GBM+sector": FM.OWN + ["g_sector"], "GBM+plac": FM.OWN + ["g_plac"],
                    "GBM+oracle": FM.OWN + ["g_oracle"]}
            if has_earn:
                cols["GBM+earn"] = FM.OWN + FM.EARN
                cols["GBM+earn+corr"] = FM.OWN + FM.EARN + ["g_corr"]
            preds["HAR"].append(FM._har_ols(trf, tef))
            preds["HARQ"].append(FM._harq_ols(trf, tef))
            ph = {}
            for m, cc in cols.items():
                p = np.mean([FM.gbm(trf, tef, cc, s) for s in FM.SEEDS], 0)
                preds[m].append(p)
                ph[m] = p
            yf = tef["y"].to_numpy(float)
            for m in per_fold:
                per_fold[m].append(float(np.mean(M.per_obs_qlike(yf, ph[m], floor=FL))))
            yy.append(yf)
            dts.append(tef["date"].to_numpy())
        y = np.concatenate(yy)
        dates = np.concatenate(dts)
        e = {m: M.per_obs_qlike(y, np.concatenate(preds[m]), floor=FL) for m in models}
        met = {m: all_metrics(y, np.concatenate(preds[m])) for m in models}
        dmr = {}
        pairs = [("GBM+corr", "GBM+market"), ("GBM+oracle", "GBM+corr")]
        if has_earn:
            pairs.append(("GBM+earn", "GBM"))
        for x, b in pairs:
            dmr[f"{x}_vs_{b}"] = ST.date_clustered_dm(e[x], e[b], dates, h)["p_value"]
        oracle_gain = (met["GBM+corr"]["qlike"] - met["GBM+oracle"]["qlike"]) / met["GBM+corr"]["qlike"] * 100
        out[f"h{h}"] = {"n": int(len(y)), "metrics": met, "dm_qlike": dmr, "oracle_gain_pct": oracle_gain,
                        "per_fold_qlike": per_fold}
        print(f"\n=== {market} h{h} (n={len(y):,}) ===", flush=True)
        for m in models:
            mm = met[m]
            print(f"  {m:16s} QLIKE {mm['qlike']:.4f} RMSE {mm['rmse']:.3e} MAE {mm['mae']:.3e} R2 {mm['r2']:.3f}",
                  flush=True)
        print(f"  oracle gain vs corr: {oracle_gain:+.2f}% (p={dmr['GBM+oracle_vs_GBM+corr']:.3f})", flush=True)
    Path(REPO / "results" / "xgb" / f"paper_metrics_{market}.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved results/xgb/paper_metrics_{market}.json", flush=True)

if __name__ == "__main__":
    main()
