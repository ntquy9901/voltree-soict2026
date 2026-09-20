"""FINAL paper-grade walk-forward for the leaf-cooccurrence-graph result: does smoothing the champion-matched
gamma-XGB's predictions over a graph built from the XGB's OWN trees (leaf-index cooccurrence) beat the un-smoothed
XGB on out-of-sample QLIKE — at 3 seeds, on HOSE + SP500, and does the effect survive a leave-one-out removal of
the earnings features?

Model set is {XGB base, XGB+leafgraph} ONLY. The champion HGBR ``GBME`` is deliberately NOT fit or reported here.

Per horizon h and outer fold k: fit XGB (plain XGBoost gamma, capacity-matched to the champion HGBR) and
XGB+leafgraph (the XGB base predictions smoothed over a per-day kNN leaf-cooccurrence graph, alpha fit on a val
slice). Two feature sets: ``full`` = OWN-8 + EARN-4, ``noearn`` = OWN-8 only (leave-one-out of the 4 EARN
features). Score train(streamed)/val/test, pool over folds, date-clustered DM (XGB+leafgraph vs XGB = isolate the
graph), verdict + HOSE per-fold QLIKE + regime-spike robustness, fitted alpha.

Memory: an earlier variant accumulated every fold's TRAIN predictions and OOM'd at h22 on SP500. Here train
metrics are streamed via sufficient statistics (``_Stream``) and the per-fold train arrays are freed immediately;
only the smaller val/test arrays are pooled (needed for DM). Run one (market, feature-set, horizon) per process.

Output: results/gamma_gbm/leaf_graph_paper_<market>_<full|noearn>_h<h>.json, one per horizon, atomic checkpoint.

Run: python run_leaf_graph_paper.py [hose|sp500] [--featureset full|noearn] [--horizon H] [--smoke]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"),
           str(REPO / "scripts" / "quality_gate"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import full_matrix as FM  # noqa: E402
import vn_gbm_graph_stage1 as S1  # noqa: E402
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402
import overfit_check as OF  # noqa: E402
import leaf_graph_paper_config as C  # noqa: E402
import leaf_graph_lib as LG  # noqa: E402

FL = FM.FL
XGB, XGBLG = "XGB", "XGB+leafgraph"
ORDER = [XGB, XGBLG]
FEATURE_SETS = ("full", "noearn")


def _own8():
    """OWN-8 own-history feature list, single-sourced from the paper_models config, loaded by path so it does
    NOT register a second bare ``config`` module (avoids the sys.modules collision)."""
    cfg_path = REPO / "baselines" / "2026-09-13_paper_models" / "code" / "config.py"
    spec = importlib.util.spec_from_file_location("paper_models_config", cfg_path)
    pmc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pmc)
    return pmc.own_set(FM.OWN)


OWN = _own8()


def _load_earn(market, edates):
    """Replace edates with the REAL crawled VN announcement dates if present (mirrors the sibling drivers);
    SP500 keeps its own earnings from FM.load."""
    if market == "sp500":
        return edates
    ep = REPO / "results" / "gamma_gbm" / "hose_earnings_combined.parquet"
    if ep.exists():
        e = pd.read_parquet(ep)
        return {tk: np.sort(g["earnings_date"].to_numpy()) for tk, g in e.groupby("ticker")}
    return edates


def resolve_cols(feature_set, has_earn):
    """Feature columns for a run arm. ``full`` = OWN-8 + EARN-4 (when earnings exist); ``noearn`` = OWN-8 only
    (leave-one-out removal of the 4 EARN features). Rows are identical across arms (panel dropna is on OWN+y),
    so full-vs-noearn on the same base isolates the earnings marginal effect."""
    use_earn = has_earn and feature_set == "full"
    return OWN + (FM.EARN if use_earn else []), use_earn


class _Stream:
    """Streaming pooled-metric accumulator so the large TRAIN arrays never all live in memory at once.

    Accumulates the sufficient statistics for the 5 reported metrics (mse/rmse/mae/r2/qlike) over folds; each
    fold's arrays are folded in then dropped. ``finalize`` reproduces ``mean_squared_error`` / ``r2_score`` /
    per-obs QLIKE exactly (pooled ybar, pooled SS_tot)."""

    def __init__(self):
        self.n = 0
        self.sse = 0.0      # sum (y-p)^2
        self.sae = 0.0      # sum |y-p|
        self.sy = 0.0       # sum y
        self.syy = 0.0      # sum y^2
        self.sq = 0.0       # sum per-obs qlike

    def update(self, y, p):
        y = np.asarray(y, float)
        p = np.asarray(p, float)
        r = y - p
        self.n += len(y)
        self.sse += float(np.dot(r, r))
        self.sae += float(np.abs(r).sum())
        self.sy += float(y.sum())
        self.syy += float(np.dot(y, y))
        self.sq += float(M.per_obs_qlike(y, p, floor=FL).sum())

    def finalize(self):
        n = self.n
        mse = self.sse / n
        ss_tot = self.syy - self.sy * self.sy / n
        r2 = 1.0 - self.sse / ss_tot if ss_tot > 0 else 0.0
        return {"mse": float(mse), "rmse": float(np.sqrt(mse)), "mae": float(self.sae / n),
                "r2": float(r2), "qlike": float(self.sq / n)}


def _metrics5(y, p):
    """The 5 reported metrics (DirAcc dropped per paper style) computed directly from pooled arrays."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    r = y - p
    mse = float(np.dot(r, r) / len(y))
    dy = y - y.mean()
    ss_tot = float(np.dot(dy, dy))
    r2 = 1.0 - float(np.dot(r, r)) / ss_tot if ss_tot > 0 else 0.0
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.abs(r).mean()),
            "r2": float(r2), "qlike": float(np.mean(M.per_obs_qlike(y, p, floor=FL)))}


def verdict(gain_pct, dm_p):
    """A horizon 'beats' iff the QLIKE gain over XGB is strictly positive AND the DM p-value clears DM_ALPHA."""
    return bool(gain_pct > C.GAIN_MIN * 100.0 and dm_p < C.DM_ALPHA)


def success(docs):
    """Pre-registered: XGB+leafgraph beats XGB at BOTH KILL_HORIZONS. Missing horizon = fail."""
    return bool(all(docs.get(h, {}).get("verdict", {}).get("beats") for h in C.KILL_HORIZONS))


def _safe_dm(err_a, err_b, dates, h):
    """Date-clustered DM, returning p=1.0 for the degenerate cases the reused DM cannot handle (identical loss
    series, or too few surviving dates for the HLN factor)."""
    degenerate = {"p_value": 1.0, "mean_diff": 0.0, "dm_hln": 0.0, "n_dates": int(np.unique(dates).size)}
    if np.allclose(err_a, err_b):
        return degenerate
    try:
        return ST.date_clustered_dm(err_a, err_b, dates, h)
    except ValueError:
        return degenerate


def _spike_mask(dates):
    """Boolean mask of test dates inside any configured regime-spike window (HOSE robustness)."""
    d = pd.to_datetime(dates)
    m = np.zeros(len(d), bool)
    for lo, hi in C.SPIKE_WINDOWS:
        m |= (d >= pd.Timestamp(lo)) & (d <= pd.Timestamp(hi))
    return m


def _fold_predictions(trf_e, vaf, tef, cols, seeds):
    """Both models' predictions for one fold, plus the fold's fitted alpha.

    XGB is the seed-ensembled base prediction over combo=[test, train, val]. XGB+leafgraph reuses the XGB base
    (isolating the graph), fits alpha on the val slice's leaf-graph, and applies the frozen alpha to smooth
    test/train/val. Returns dicts keyed by split ('te'|'tr'|'va') -> {model: array}, and alpha."""
    combo = pd.concat([tef, trf_e, vaf])
    n_te, n_tr = len(tef), len(trf_e)
    dates = combo["date"].to_numpy()
    p_xgb = LG.predict_xgb(trf_e, combo, cols, seeds)
    leaves = LG.leaf_matrix(LG.fit_booster(trf_e, cols, seeds[0]), combo[cols].to_numpy(float))

    def sl(arr, s):
        return {"te": arr[:n_te], "tr": arr[n_te:n_te + n_tr], "va": arr[n_te + n_tr:]}[s]

    alpha, _ = LG.fit_alpha(vaf["y"].to_numpy(float), sl(p_xgb, "va"), sl(leaves, "va"), sl(dates, "va"),
                            C.K_NEIGHBOURS, C.ALPHA_GRID, FL)
    out = {s: {} for s in ("te", "tr", "va")}
    for s in ("te", "tr", "va"):
        out[s][XGB] = sl(p_xgb, s)
        smoothed = LG.smooth_all(sl(p_xgb, s), sl(leaves, s), sl(dates, s), C.K_NEIGHBOURS, alpha)
        out[s][XGBLG] = np.clip(smoothed, FL, C.PRED_CAP)
    return out, float(alpha)


def _pool_doc(h, market, feature_set, cols, seeds, te_pred, va_pred, yy_te, yy_va, dts, train_metrics,
              alphas, per_fold, spike):
    """Per-horizon result doc: pooled 5-metric train(streamed)/val/test for both models, fit_diagnostics,
    date-clustered DM (XGB+leafgraph vs XGB = isolate the graph), gain, verdict, fitted alpha, plus HOSE
    per-fold/spike robustness."""
    y = np.concatenate(yy_te)
    dates = np.concatenate(dts)
    pooled = {m: np.concatenate(te_pred[m]) for m in ORDER}
    err = {m: M.per_obs_qlike(y, pooled[m], floor=FL) for m in ORDER}
    metrics = {m: _metrics5(y, pooled[m]) for m in ORDER}
    y_va = np.concatenate(yy_va)
    val_metrics = {m: _metrics5(y_va, np.concatenate(va_pred[m])) for m in ORDER}
    fit = {m: OF.classify_fit(train_metrics[m], val_metrics[m], metrics[m]) for m in ORDER}
    dm_xgb = _safe_dm(err[XGBLG], err[XGB], dates, h)
    gain = (metrics[XGB]["qlike"] - metrics[XGBLG]["qlike"]) / metrics[XGB]["qlike"] * 100.0
    doc = {"market": market, "feature_set": feature_set, "h": h, "n": int(len(y)),
           "n_folds": len(yy_te), "seeds": list(seeds), "features": list(cols),
           "metrics": metrics, "train_metrics": train_metrics, "val_metrics": val_metrics,
           "fit_diagnostics": fit,
           "alpha": {"per_fold": [float(a) for a in alphas], "mean": float(np.mean(alphas))},
           "dm": {f"{XGBLG}_vs_{XGB}": dm_xgb["p_value"]},
           "gain_pct": {f"{XGBLG}_vs_{XGB}": gain},
           "verdict": {"beats": verdict(gain, dm_xgb["p_value"]), "gain_pct": gain, "dm_p": dm_xgb["p_value"],
                       "criterion": "XGB+leafgraph beats XGB: gain>GAIN_MIN AND DM p<DM_ALPHA"}}
    if per_fold:
        doc["per_fold_qlike"] = {m: [float(np.mean(M.per_obs_qlike(yy_te[i], te_pred[m][i], floor=FL)))
                                     for i in range(len(yy_te))] for m in ORDER}
    if spike:
        sm = _spike_mask(dates)
        keep = ~sm
        if keep.any():
            dm_ex = _safe_dm(err[XGBLG][keep], err[XGB][keep], dates[keep], h)
            qx, qz = float(np.mean(err[XGB][keep])), float(np.mean(err[XGBLG][keep]))
            doc["spike_robustness"] = {
                "n_spike_obs": int(sm.sum()), "n_ex_spike_obs": int(keep.sum()),
                "qlike_ex_spike": {XGB: qx, XGBLG: qz}, "gain_pct_ex_spike": (qx - qz) / qx * 100.0,
                "dm_p_ex_spike": dm_ex["p_value"], "beats_ex_spike": verdict((qx - qz) / qx * 100.0,
                                                                             dm_ex["p_value"])}
    return doc


def _checkpoint(doc, out_path):
    """Atomically write the per-horizon doc (tmp + replace) so a disconnect keeps the completed horizon."""
    tmp = Path(str(out_path) + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(out_path)


def run(market, feature_set="full", load_fn=None, out_dir=None, smoke=False, horizons=None):
    """Walk-forward XGB vs XGB+leafgraph for a (market, feature_set). Writes one JSON per horizon and returns
    {h: doc}. Train metrics are streamed (no cross-fold train-array accumulation)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}")
    load_fn = load_fn or FM.load
    seeds = (C.SEEDS[0],) if smoke else C.SEEDS
    horizons = horizons or (C.HORIZONS_SMOKE if smoke else C.HORIZONS)
    fold_cap = 1 if smoke else None
    min_rows = C.MIN_ROWS.get(market, C.MIN_ROWS["default"])
    out_dir = Path(out_dir) if out_dir else (REPO / "results" / "gamma_gbm")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if smoke else ""

    frames, _sect, edates = load_fn(market)
    edates = _load_earn(market, edates)
    has_earn = bool(edates)
    cols, use_earn = resolve_cols(feature_set, has_earn)
    per_fold = market != "sp500"
    spike = market != "sp500"
    docs = {}
    for h in horizons:
        t0 = time.time()
        out_path = out_dir / f"leaf_graph_paper_{market}_{feature_set}{tag}_h{h}.json"
        a = FM.panel(frames, edates, h)
        embargo = pd.Timedelta(days=int(h * C.EMBARGO_MULT) + C.EMBARGO_BUFFER_DAYS)
        te_pred = {m: [] for m in ORDER}
        va_pred = {m: [] for m in ORDER}
        stream = {m: _Stream() for m in ORDER}
        yy_te, yy_va, dts, alphas = [], [], [], []
        n_done = 0
        for k in range(len(S1.FOLDS) - 1):
            ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
            trf = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
            tef = a[(a.date >= ts) & (a.date < tend)]
            if len(tef) == 0 or len(trf) < min_rows:
                continue
            if fold_cap is not None and n_done >= fold_cap:
                break
            n_done += 1
            val_dates = np.sort(trf["date"].unique())[-C.VALID_LEN:]
            is_val = trf["date"].isin(val_dates)
            trf_e, vaf = trf[~is_val], trf[is_val]
            fp, alpha = _fold_predictions(trf_e, vaf, tef, cols, seeds)
            y_tr = trf_e["y"].to_numpy(float)
            for m in ORDER:
                te_pred[m].append(fp["te"][m])
                va_pred[m].append(fp["va"][m])
                stream[m].update(y_tr, fp["tr"][m])                 # fold train arrays streamed then dropped
            yy_te.append(tef["y"].to_numpy(float))
            yy_va.append(vaf["y"].to_numpy(float))
            dts.append(tef["date"].to_numpy())
            alphas.append(alpha)
            del fp
            train_metrics = {m: stream[m].finalize() for m in ORDER}
            doc = _pool_doc(h, market, feature_set, cols, seeds, te_pred, va_pred, yy_te, yy_va, dts,
                            train_metrics, alphas, per_fold, spike)
            _checkpoint(doc, out_path)
            print(f"  {market}/{feature_set} h{h} fold {k} ({ts.date()}) alpha={alpha:.2f} done, "
                  f"{time.time()-t0:.0f}s (checkpointed)", flush=True)
        if not yy_te:  # pragma: no cover - defensive: a horizon with no eligible walk-forward fold
            continue
        train_metrics = {m: stream[m].finalize() for m in ORDER}
        doc = _pool_doc(h, market, feature_set, cols, seeds, te_pred, va_pred, yy_te, yy_va, dts,
                        train_metrics, alphas, per_fold, spike)
        _checkpoint(doc, out_path)
        v = doc["verdict"]
        print(f"\n== {market}/{feature_set} h{h} (n={doc['n']:,}, {doc['n_folds']} folds, use_earn={use_earn}, "
              f"alpha_mean={doc['alpha']['mean']:.2f}) ==", flush=True)
        for m in ORDER:
            print(f"  {m:14s} QLIKE {doc['metrics'][m]['qlike']:.4f}", flush=True)
        print(f"  DM {XGBLG} vs {XGB}: {v['gain_pct']:+.2f}% (p={v['dm_p']:.4f}) beats={v['beats']} | "
              f"saved {out_path.name}", flush=True)
        docs[h] = doc
    return docs


def main():  # pragma: no cover - entry driver: loads real data + full walk-forward over folds/seeds
    ap = argparse.ArgumentParser()
    ap.add_argument("market", nargs="?", choices=("sp500", "hose"), default="hose")
    ap.add_argument("--featureset", choices=FEATURE_SETS, default="full")
    ap.add_argument("--horizon", type=int, default=None, help="run a single horizon in a fresh process")
    ap.add_argument("--smoke", action="store_true", help="1 horizon, 1 fold, 1 seed")
    args = ap.parse_args()
    hz = (args.horizon,) if args.horizon else None
    docs = run(args.market, feature_set=args.featureset, smoke=args.smoke, horizons=hz)
    if set(C.KILL_HORIZONS) <= set(docs):        # only meaningful when both kill horizons ran in this process
        print(f"\nPRE-REGISTERED SUCCESS (h1 & h5 both beat XGB): {success(docs)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
