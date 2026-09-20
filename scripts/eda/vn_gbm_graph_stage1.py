"""Stage 1 of docs/experement_guide/gbm_graph_volatility_implementation_brief.md, executed faithfully on VN30 and
VN100: GBM + leakage-safe graph AGGREGATION features, with the decision controls the brief mandates.

Models (per brief section 6 / 9):
  M0  = GBM(stock-only own features)                       -- core nonlinear baseline
  M1  = M0 + market aggregate (market_pk) + sector aggregate (same-ICB-sector mean vol at t)
  M2  = M1 + graph-aggregation features from a volatility-correlation top-k graph (brief section 16 core set)
  M2p = M1 + the SAME graph features but from RANDOMISED edges of identical density (brief placebo, section 9)

Graph (section 5/16): per-fold static volatility-correlation graph built on TRAIN log-vol only, top_k neighbours,
positive-only weights. Neighbour aggregation features (section 6/16): weighted neighbour vol, weighted neighbour
vol-shock, neighbour vol max, neighbour vol dispersion, node-minus-neighbour vol, weighted neighbour return,
weighted neighbour volume-shock.

Validation (section 10): expanding walk-forward, per-fold graph from train only, target embargo at the train/test
boundary, single evaluation of pooled test. Metric QLIKE; DM aggregated by trading date (section 11). Decision
(section 12): a graph model is retained only if M2 beats BOTH M0 and, crucially, M1, AND beats the placebo.

Note on GBM: uses HistGradientBoostingRegressor(loss='gamma') -- the project's canonical gamma-loss GBM
(gamma deviance == QLIKE up to a constant), consistent with the delivered SP500/VN champion; satisfies the
brief's "nonlinear GBM" intent without adding a LightGBM/XGBoost dependency."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "submission" / "soict_lstm_gat"))
sys.path.insert(0, str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"))
import metrics as M  # noqa: E402
import pipeline_config as pc  # noqa: E402
import stats as ST  # noqa: E402

FL = pc.QLIKE_FLOOR
WK, MO = 5, 22
OWN = ["har_daily", "har_weekly", "har_monthly", "volume_zscore_22",
       "rq", "mr_change", "mr_slope5", "mr_slope10", "mr_dev5", "mr_z22"]
AGG = ["market_pk", "sect_mean"]
GRAPH = ["g_nb_vol", "g_nb_shock", "g_nb_max", "g_nb_disp", "g_node_minus_nb", "g_nb_ret", "g_nb_volshock"]
TRAIN_START = "2015-01-01"
FOLDS = ["2022-07-01", "2023-01-01", "2023-07-01", "2024-01-01", "2024-07-01", "2025-01-01",
         "2025-07-01", "2026-01-01", "2100-01-01"]
TOPK, RNG_SEED = 10, 20260910
SECT = REPO / "baselines" / "2026-08-29_sector_gat_ablation" / "vn_icb_sectors.csv"


def _feat(d):
    pk = d["parkinson_variance"].to_numpy(float); lpk = pd.Series(np.log(np.maximum(pk, FL)), index=d.index)
    d["logpk"] = lpk
    d["rq"] = np.sqrt(pd.Series(pk ** 2, index=d.index).rolling(WK, min_periods=1).mean())
    d["mr_change"] = lpk.diff(1); d["mr_slope5"] = (lpk - lpk.shift(WK)) / WK
    d["mr_slope10"] = (lpk - lpk.shift(2 * WK)) / (2 * WK); d["mr_dev5"] = lpk - lpk.rolling(WK).mean()
    d["mr_z22"] = (lpk - lpk.rolling(MO).mean()) / (lpk.rolling(MO).std() + FL)
    return d


def load(market):
    sect = pd.read_csv(SECT).set_index("symbol")["industry_code"].to_dict()
    frames = {}
    for p in glob.glob(str(REPO / "data" / "processed_enriched" / market / "*.csv")):
        tk = Path(p).stem
        if tk.endswith("_rejections"):
            continue
        d = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        d = _feat(d); d["ticker"] = tk; d["sector"] = sect.get(tk, -1)
        frames[tk] = d[OWN + ["date", "ticker", "sector", "parkinson_variance", "logpk", "market_pk",
                             "daily_return"]]
    return frames


def panel(frames, h):
    rows = []
    for tk, d in frames.items():
        e = d.copy(); e["y"] = e["parkinson_variance"].shift(-h)
        rows.append(e)
    a = pd.concat(rows, ignore_index=True)
    a["sect_mean"] = a.groupby(["date", "sector"])["parkinson_variance"].transform("mean")
    return a.dropna(subset=OWN + ["y"]).reset_index(drop=True)


def _std_cols(df):
    z = (df - df.mean()) / df.std().replace(0, np.nan)
    return z.fillna(0.0).to_numpy(float), z.notna().to_numpy(float)


def build_graph(train, tickers, rng):
    """per-fold static vol-correlation top-k graph on TRAIN logpk; returns real+placebo row-normalised weight
    matrices W[i,j] (neighbour j of node i), positive-only."""
    piv = train.pivot_table(index="date", columns="ticker", values="logpk").reindex(columns=tickers)
    X, Xm = _std_cols(piv)
    corr = (X.T @ X) / np.maximum(Xm.T @ Xm, 1.0); np.fill_diagonal(corr, -np.inf)
    n = len(tickers); W = np.zeros((n, n)); Wp = np.zeros((n, n))
    for i in range(n):
        top = np.argsort(corr[i])[::-1][:TOPK]
        W[i, top] = np.clip(corr[i, top], 0.0, None)
        rand = rng.choice(np.delete(np.arange(n), i), size=min(TOPK, n - 1), replace=False)
        Wp[i, rand] = 1.0
    W /= np.maximum(np.abs(W).sum(1, keepdims=True), 1e-12)
    Wp /= np.maximum(np.abs(Wp).sum(1, keepdims=True), 1e-12)
    return W, Wp


def graph_feats(fold, tickers, W, tag):
    """neighbour-aggregation features for every (date,ticker) row in `fold` using weight matrix W."""
    idx = {t: j for j, t in enumerate(tickers)}
    def mat(col):
        m = fold.pivot_table(index="date", columns="ticker", values=col).reindex(columns=tickers)
        return m.index, m.to_numpy(float)
    dts, PK = mat("parkinson_variance"); _, SH = mat("mr_change"); _, RT = mat("daily_return")
    _, VZ = mat("volume_zscore_22")
    pkf = np.where(np.isnan(PK), np.nanmean(PK, axis=1, keepdims=True), PK)
    shf = np.where(np.isnan(SH), 0.0, SH); rtf = np.where(np.isnan(RT), 0.0, RT)
    vzf = np.where(np.isnan(VZ), 0.0, VZ)
    nb_vol = pkf @ W.T; nb_shock = shf @ W.T; nb_ret = rtf @ W.T; nb_volshock = vzf @ W.T
    nb_max = np.full_like(pkf, np.nan); nb_disp = np.full_like(pkf, np.nan)
    for i in range(len(tickers)):
        js = np.where(W[i] > 0)[0]
        if js.size:
            sub = pkf[:, js]; nb_max[:, i] = sub.max(1); nb_disp[:, i] = sub.std(1)
    out = {"g_nb_vol": nb_vol, "g_nb_shock": nb_shock, "g_nb_max": nb_max, "g_nb_disp": nb_disp,
           "g_node_minus_nb": pkf - nb_vol, "g_nb_ret": nb_ret, "g_nb_volshock": nb_volshock}
    dpos = {d: k for k, d in enumerate(dts)}
    r = fold["date"].map(dpos).to_numpy(); c = fold["ticker"].map(idx).to_numpy()
    cols = {f"{k}{tag}": out[k][r, c] for k in GRAPH}
    return pd.DataFrame(cols, index=fold.index)


def _gbm(tr, te, cols):
    m = HistGradientBoostingRegressor(loss="gamma", max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                      l2_regularization=1.0, random_state=0)
    m.fit(tr[cols].to_numpy(float), np.maximum(tr["y"].to_numpy(float), FL))
    return np.maximum(m.predict(te[cols].to_numpy(float)), FL)


def run_market(market, results):  # pragma: no cover - driver: full walk-forward loop (helpers tested directly)
    frames = load(market)
    print(f"\n########## {market.upper()} ({len(frames)} tickers) ##########", flush=True)
    for h in (1, 5, 10, 22):
        a = panel(frames, h)
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        tickers = sorted(a["ticker"].unique())
        acc = {m: [] for m in ("M0", "M1", "M2", "M2p")}; dts = []
        for k in range(len(FOLDS) - 1):
            ts, tend = pd.Timestamp(FOLDS[k]), pd.Timestamp(FOLDS[k + 1])
            tr = a[(a.date >= TRAIN_START) & (a.date < ts - embargo)]
            te = a[(a.date >= ts) & (a.date < tend)]
            if len(te) == 0 or len(tr) < 3000:
                continue
            rng = np.random.default_rng(RNG_SEED + k)
            W, Wp = build_graph(tr, tickers, rng)
            fold = a[(a.date >= TRAIN_START) & (a.date < tend)]
            gf = graph_feats(fold, tickers, W, ""); gfp = graph_feats(fold, tickers, Wp, "p")
            fa = pd.concat([fold.reset_index(drop=True), gf.reset_index(drop=True),
                            gfp.reset_index(drop=True)], axis=1)
            gp = [c + "p" for c in GRAPH]
            fa[GRAPH + gp] = fa[GRAPH + gp].fillna(0.0)
            trf = fa[(fa.date >= TRAIN_START) & (fa.date < ts - embargo)]
            tef = fa[(fa.date >= ts) & (fa.date < tend)]
            y = tef["y"].to_numpy(float)
            preds = {"M0": _gbm(trf, tef, OWN), "M1": _gbm(trf, tef, OWN + AGG),
                     "M2": _gbm(trf, tef, OWN + AGG + GRAPH), "M2p": _gbm(trf, tef, OWN + AGG + gp)}
            for m, p in preds.items():
                acc[m].append(M.per_obs_qlike(y, p, floor=FL))
            dts.append(tef["date"].to_numpy())
        if not dts:
            continue
        e = {m: np.concatenate(v) for m, v in acc.items()}; dates = np.concatenate(dts)
        q = {m: float(np.mean(e[m])) for m in e}

        def dm(a_, b_):
            return ST.date_clustered_dm(e[a_], e[b_], dates, h)["p_value"]
        print(f"\n----- {market} h{h} (folds pooled n={len(dates)}) -----", flush=True)
        print(f"  M0 stock-only : {q['M0']:.4f}", flush=True)
        print(f"  M1 +mkt/sect  : {q['M1']:.4f}  vs M0 {(q['M0']-q['M1'])/q['M0']*100:+.2f}% (p={dm('M1','M0'):.3f})",
              flush=True)
        print(f"  M2 +graph     : {q['M2']:.4f}  vs M1 {(q['M1']-q['M2'])/q['M1']*100:+.2f}% (p={dm('M2','M1'):.3f})"
              f" | vs M0 {(q['M0']-q['M2'])/q['M0']*100:+.2f}% (p={dm('M2','M0'):.3f})", flush=True)
        print(f"  M2p placebo   : {q['M2p']:.4f}  vs M1 {(q['M1']-q['M2p'])/q['M1']*100:+.2f}%"
              f" | M2 vs M2p {(q['M2p']-q['M2'])/q['M2p']*100:+.2f}% (p={dm('M2','M2p'):.3f})", flush=True)
        results[f"{market}_h{h}"] = {"q": q, "p_M2_vs_M1": dm("M2", "M1"), "p_M2_vs_M0": dm("M2", "M0"),
                                     "p_M2_vs_M2p": dm("M2", "M2p"), "p_M1_vs_M0": dm("M1", "M0"), "n": len(dates)}


def main():  # pragma: no cover - entry driver: runs both markets, writes JSON
    results = {}
    for market in ("vn30", "vn100"):
        run_market(market, results)
    Path(REPO / "results" / "gamma_gbm" / "vn_gbm_graph_stage1.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
