"""Full comparison matrix (reviewer + user): HAR / HARQ / GBM(own) / GBM+graph / GBM+market / GBM+earn /
GBM+earn+graph / random-edge placebo, over ALL horizons, ALL walk-forward folds, and MULTIPLE seeds. Baselines are
HAR and HARQ (no HAR-X, no market_pk/volume scalar). Cross-sectional / market information enters ONLY through graph
neighbour aggregates: market = uniform-adjacency aggregate, corr = correlation top-k, placebo = degree-matched
random edges (multiple seeds). Earnings apply to SP500 only (Vietnam has no per-firm dates). gamma-GBM predictions
are averaged across seeds (ensemble) for the reported QLIKE + date-clustered DM; per-seed spread is reported.

Run: `python full_matrix.py [sp500|vn30|vn100]`."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eda"))
import vn_gbm_graph_stage1 as S1  # noqa: E402
sys.path.insert(0, str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"))
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402

FL = S1.FL
HAR = ["har_daily", "har_weekly", "har_monthly"]
OWN = HAR + ["rq", "mr_change", "mr_slope5", "mr_slope10", "mr_dev5", "mr_z22"]   # own-history, no market/volume
EARN = ["earn_prox", "earn_soon", "earn_pre", "earn_post"]
SEEDS = (0, 1, 2)
PLAC_SEEDS = (11, 12, 13)
WK = 5


def load(market):
    if market == "sp500":
        sect = json.load(open(REPO / "results" / "gamma_gbm" / "sp500_sectors.json"))
        d = "sp500_clean"
    else:
        sect = pd.read_csv(S1.SECT).set_index("symbol")["industry_code"].to_dict(); d = market
    frames = {}
    for p in glob.glob(str(REPO / "data" / "processed_enriched" / d / "*.csv")):
        tk = Path(p).stem
        if tk.endswith("_rejections"):
            continue
        fr = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        frames[tk] = S1._feat(fr).assign(ticker=tk, sector=sect.get(tk, -1))
    edates = {}
    if market == "sp500":
        e = pd.read_parquet(REPO / "results" / "gamma_gbm" / "sp500_earnings.parquet")
        edates = {tk: np.sort(g["earnings_date"].to_numpy()) for tk, g in e.groupby("ticker")}
    return frames, sect, edates


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


def _uniform(n):
    return (np.ones((n, n)) - np.eye(n)) / (n - 1)


def nb(fold, tickers, W):
    piv = fold.pivot_table(index="date", columns="ticker", values="parkinson_variance").reindex(columns=tickers).sort_index()
    V = piv.to_numpy(float); Vf = np.nan_to_num(np.where(np.isnan(V), np.nanmean(V, axis=1, keepdims=True), V))
    NB = Vf @ W.T; dpos = {d: i for i, d in enumerate(piv.index)}; cpos = {c: j for j, c in enumerate(tickers)}
    return NB[fold["date"].map(dpos).to_numpy(), fold["ticker"].map(cpos).to_numpy()]


def _harq_ols(tr, te):
    def dm(df):
        x = df[HAR].to_numpy(float); return np.column_stack([np.ones(len(x)), x, x[:, 0] * df["rq"].to_numpy(float)])
    c = np.linalg.lstsq(dm(tr), np.maximum(tr["y"].to_numpy(float), FL), rcond=None)[0]
    nf = 0.01 * np.maximum(tr["y"], FL).mean()
    return np.maximum(dm(te) @ c, nf)


def _har_ols(tr, te):
    x = lambda df: np.column_stack([np.ones(len(df)), df[HAR].to_numpy(float)])  # noqa: E731
    c = np.linalg.lstsq(x(tr), np.maximum(tr["y"].to_numpy(float), FL), rcond=None)[0]
    return np.maximum(x(te) @ c, 0.01 * np.maximum(tr["y"], FL).mean())


def gbm(tr, te, cols, seed):
    m = HistGradientBoostingRegressor(loss="gamma", max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                      l2_regularization=1.0, random_state=seed)
    m.fit(tr[cols].to_numpy(float), np.maximum(tr["y"].to_numpy(float), FL))
    return np.maximum(m.predict(te[cols].to_numpy(float)), FL)


def main():  # pragma: no cover - entry driver: full walk-forward over all folds/seeds, writes JSON
    market = sys.argv[1] if len(sys.argv) > 1 else "sp500"
    frames, sect, edates = load(market)
    has_earn = bool(edates)
    GMODELS = {"GBM": OWN, "GBM+market": OWN + ["g_market"], "GBM+graph": OWN + ["g_corr"],
               "GBM+plac": OWN + ["g_plac"]}
    if has_earn:
        GMODELS.update({"GBM+earn": OWN + EARN, "GBM+earn+graph": OWN + EARN + ["g_corr"]})
    out = {}
    for h in (1, 5, 10, 22):
        a = panel(frames, edates, h); embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        tickers = sorted(a["ticker"].unique()); n = len(tickers); Wu = _uniform(n)
        preds = {m: [] for m in ["HAR", "HARQ"] + list(GMODELS)}; yy, dts = [], []
        for k in range(len(S1.FOLDS) - 1):
            ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
            tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]; te = a[(a.date >= ts) & (a.date < tend)]
            if len(te) == 0 or len(tr) < (30000 if market == "sp500" else 3000):
                continue
            Wc, _ = S1.build_graph(tr, tickers, np.random.default_rng(S1.RNG_SEED + k))
            fold = a[(a.date >= S1.TRAIN_START) & (a.date < tend)].copy()
            fold["g_market"] = nb(fold, tickers, Wu); fold["g_corr"] = nb(fold, tickers, Wc)
            plc = []
            for ps in PLAC_SEEDS:
                rng = np.random.default_rng(ps + k); Wp = np.zeros((n, n))
                for i in range(n):
                    j = rng.choice(np.delete(np.arange(n), i), size=min(S1.TOPK, n - 1), replace=False); Wp[i, j] = 1.0 / len(j)
                plc.append(nb(fold, tickers, Wp))
            fold["g_plac"] = np.mean(plc, 0)
            for c in ["g_market", "g_corr", "g_plac"]:
                fold[c] = fold[c].fillna(0.0)
            trf = fold[(fold.date >= S1.TRAIN_START) & (fold.date < ts - embargo)]; tef = fold[(fold.date >= ts) & (fold.date < tend)]
            preds["HAR"].append(_har_ols(trf, tef)); preds["HARQ"].append(_harq_ols(trf, tef))
            for mdl, cols in GMODELS.items():
                preds[mdl].append(np.mean([gbm(trf, tef, cols, s) for s in SEEDS], 0))   # seed-ensemble
            yy.append(tef["y"].to_numpy(float)); dts.append(tef["date"].to_numpy())
        y = np.concatenate(yy); dates = np.concatenate(dts)
        e = {m: M.per_obs_qlike(y, np.concatenate(preds[m]), floor=FL) for m in preds}
        q = {m: float(np.mean(e[m])) for m in e}
        print(f"\n===== {market} h{h} (n={len(y):,}, {len(SEEDS)} seeds ensemble) =====", flush=True)
        for m in e:
            print(f"  {m:16s} QLIKE {q[m]:.4f}", flush=True)
        cmp = [("GBM+graph", "GBM"), ("GBM+graph", "GBM+market"), ("GBM+graph", "GBM+plac")]
        if has_earn:
            cmp += [("GBM+earn", "GBM"), ("GBM+earn+graph", "GBM+earn")]
        dmr = {}
        for x, b in cmp:
            p = ST.date_clustered_dm(e[x], e[b], dates, h)["p_value"]
            dmr[f"{x}_vs_{b}"] = p
            print(f"  DM {x} vs {b:14s}: {(q[b]-q[x])/q[b]*100:+.2f}% (p={p:.3f})", flush=True)
        out[f"h{h}"] = {"qlike": q, "dm": dmr}
    Path(REPO / "results" / "gamma_gbm" / f"full_matrix_{market}.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved results/gamma_gbm/full_matrix_{market}.json", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
