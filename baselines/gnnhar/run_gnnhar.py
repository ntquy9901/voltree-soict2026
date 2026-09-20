import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[1]
for _p in (str(REPO / "baselines" / "common"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import feature_panel as D
import metrics as M
import overfit_check as OF
import gnnhar_config as cfg
import gnnhar_model as G

FL = D.FL
DEVICE = G.DEVICE

def train_with_curves(X, Ys, Mt, adj, tr_idx, va_idx, in_f, n_gcn, seed, max_epochs, patience):
    torch.manual_seed(seed)
    model = G.GNNHAR(in_f, cfg.N_HID, n_gcn).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    tr = torch.as_tensor(tr_idx, device=DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    best_val, best_state, bad = float("inf"), None, 0
    curve = []
    for epoch in range(max_epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=gen, device=DEVICE)]
        tl_sum, n_b = None, 0
        for s in range(0, len(perm), cfg.BATCH_DATES):
            b = perm[s:s + cfg.BATCH_DATES]
            opt.zero_grad()
            loss = G._qlike_loss(model(X[b], adj), Ys[b], Mt[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            opt.step()
            tl_sum = loss.detach() if tl_sum is None else tl_sum + loss.detach()
            n_b += 1
        model.eval()
        with torch.no_grad():
            vl = G._qlike_loss(model(X[va_idx], adj), Ys[va_idx], Mt[va_idx]).item()
        tl = float((tl_sum / n_b).item())
        curve.append({"epoch": epoch, "train": tl, "val": vl})
        if vl < best_val - 1e-6:
            best_val, best_state, bad = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience and epoch >= cfg.MIN_EPOCHS - 1:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, curve

def fit(X, Ys, Mt, adj, tr_idx, va_idx, in_f, n_gcn, seed, max_epochs, patience):
    model, best_val, curve = train_with_curves(X, Ys, Mt, adj, tr_idx, va_idx, in_f, n_gcn,
                                                seed, max_epochs, patience)
    tries = 0
    while (not np.isfinite(best_val) or best_val > cfg.COLLAPSE_VAL) and tries < cfg.MAX_RESTARTS:
        tries += 1
        model, best_val, curve = train_with_curves(X, Ys, Mt, adj, tr_idx, va_idx, in_f, n_gcn,
                                                    seed + 1000 * tries, max_epochs, patience)
    return model, best_val, curve

def _metrics5(y, p):
    return {"mse": M.mse(y, p), "rmse": M.rmse(y, p), "mae": M.mae(y, p),
            "r2": M.r2(y, p), "qlike": M.qlike(y, p, floor=FL)}

def _fold_splits(fold, ts, tend, embargo):
    trf = fold[(fold.date >= D.TRAIN_START) & (fold.date < ts - embargo)]
    tef = fold[(fold.date >= ts) & (fold.date < tend)]
    val_dates = np.sort(trf["date"].unique())[-cfg.VALID_LEN:]
    is_val = trf["date"].isin(val_dates)
    return trf, trf[~is_val], trf[is_val], tef

def _gnn_maps(fold, tickers, tr_date_fn, ts, tend, trf_e, vaf, tef):
    Xn, _Yraw, Ys, mask, sc, dpos, cpos, dts = G.build_fold_tensors(fold, tickers, cfg.HAR3, tr_date_fn)
    Xt = torch.as_tensor(Xn, device=DEVICE)
    Yst = torch.as_tensor(Ys, device=DEVICE)
    Mt = torch.as_tensor(mask, device=DEVICE)
    all_tr = np.where(np.array([tr_date_fn(d) for d in dts]))[0]
    va_idx, tr_idx = all_tr[-cfg.VALID_LEN:], all_tr[:-cfg.VALID_LEN]
    te_idx = np.where((dts >= np.datetime64(ts)) & (dts < np.datetime64(tend)))[0]
    maps = {"te": (te_idx, G._row_in_te(tef["date"], dpos, te_idx), tef["ticker"].map(cpos).to_numpy()),
            "tr": (tr_idx, G._row_in_te(trf_e["date"], dpos, tr_idx), trf_e["ticker"].map(cpos).to_numpy()),
            "va": (va_idx, G._row_in_te(vaf["date"], dpos, va_idx), vaf["ticker"].map(cpos).to_numpy())}
    return Xt, Yst, Mt, sc, tr_idx, va_idx, maps

def _horizon(a, h, market, seeds, max_epochs, patience, fold_cap, min_train):
    embargo = pd.Timedelta(days=int(h * 1.6) + 5)
    tickers = sorted(a["ticker"].unique())
    preds = {sp: [] for sp in ("te", "tr", "va")}
    yy = {"te": [], "tr": [], "va": []}
    seed_q, curves, dts_te = [], [], []
    n_done = 0
    for k in range(len(D.FOLDS) - 1):
        ts, tend = pd.Timestamp(D.FOLDS[k]), pd.Timestamp(D.FOLDS[k + 1])
        tr = a[(a.date >= D.TRAIN_START) & (a.date < ts - embargo)]
        te = a[(a.date >= ts) & (a.date < tend)]
        if len(te) == 0 or len(tr) < min_train:
            continue
        if fold_cap is not None and n_done >= fold_cap:
            break
        n_done += 1
        Wc = G.build_graph(tr, tickers)
        keep = list(dict.fromkeys(cfg.HAR3 + ["date", "ticker", "y"]))
        fold = a.loc[(a.date >= D.TRAIN_START) & (a.date < tend), keep].copy()
        trf, trf_e, vaf, tef = _fold_splits(fold, ts, tend, embargo)
        y_te = tef["y"].to_numpy(float)
        tcut = ts - embargo
        tr_date_fn = (lambda d, tcut=tcut: d < np.datetime64(tcut))
        Xt, Yst, Mt, sc, tr_idx, va_idx, maps = _gnn_maps(fold, tickers, tr_date_fn, ts, tend, trf_e, vaf, tef)
        adj = torch.as_tensor(Wc, dtype=torch.float32, device=DEVICE)
        spreds = {"te": [], "tr": [], "va": []}
        for sd in seeds:
            model, _bv, curve = fit(Xt, Yst, Mt, adj, tr_idx, va_idx, len(cfg.HAR3), cfg.N_GCN,
                                    sd, max_epochs, patience)
            for sp in ("te", "tr", "va"):
                spreds[sp].append(G.row_preds(model, Xt, adj, *maps[sp], sc))
            seed_q.append(float(np.mean(M.per_obs_qlike(y_te, spreds["te"][-1], floor=FL))))
            if n_done == 1:
                curves.append(curve)
        for sp in ("te", "tr", "va"):
            preds[sp].append(np.mean(spreds[sp], 0))
        yy["te"].append(y_te)
        yy["tr"].append(trf_e["y"].to_numpy(float))
        yy["va"].append(vaf["y"].to_numpy(float))
        dts_te.append(tef["date"].to_numpy())
        del fold, Xt, Yst, Mt
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    if not dts_te:
        return None
    return _score(h, market, preds, yy, dts_te, seed_q, curves)

def _score(h, market, preds, yy, dts_te, seed_q, curves):
    y = np.concatenate(yy["te"])
    metrics = {"GNNHAR": _metrics5(y, np.concatenate(preds["te"]))}
    y_tr, y_va = np.concatenate(yy["tr"]), np.concatenate(yy["va"])
    train_metrics = {"GNNHAR": _metrics5(y_tr, np.concatenate(preds["tr"]))}
    val_metrics = {"GNNHAR": _metrics5(y_va, np.concatenate(preds["va"]))}
    fit_diag = {"GNNHAR": OF.classify_fit(train_metrics["GNNHAR"], val_metrics["GNNHAR"], metrics["GNNHAR"])}
    res = {"n": int(len(y)), "n_folds": len(dts_te), "metrics": metrics, "train_metrics": train_metrics,
           "val_metrics": val_metrics, "fit_diagnostics": fit_diag,
           "learning_curves": {"GNNHAR": curves},
           "qlike": {"GNNHAR": metrics["GNNHAR"]["qlike"]},
           "per_seed_qlike": {"GNNHAR": [round(v, 6) for v in seed_q]}}
    if market == "hose":
        res["per_fold"] = {"GNNHAR": [float(np.mean(M.per_obs_qlike(yy["te"][i], preds["te"][i], floor=FL)))
                                      for i in range(len(dts_te))]}
    return res

def _merge(out, res, h, market):
    for blk in ("metrics", "train_metrics", "val_metrics"):
        for m, v in res[blk].items():
            out[blk][f"{m}_h{h}"] = v
    for m, v in res["fit_diagnostics"].items():
        out["fit_diagnostics"][f"{m}_h{h}"] = v
    for m, v in res["learning_curves"].items():
        out["learning_curves"][f"{m}_h{h}"] = v
    for m, v in res["qlike"].items():
        out["qlike"][f"{m}_h{h}"] = v
    out["n"][f"h{h}"], out["n_folds"][f"h{h}"] = res["n"], res["n_folds"]
    out["per_seed_qlike"].update({f"{m}_h{h}": v for m, v in res["per_seed_qlike"].items()})
    if market == "hose":
        out["per_fold_qlike"].update({f"{m}_h{h}": v for m, v in res["per_fold"].items()})

def _checkpoint(out, out_path):
    if out_path is None:
        return
    tmp = Path(str(out_path) + ".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(out_path)

def run(market, load_fn=None, out_path=None, smoke=False, min_train=None, max_epochs=None, seeds=None):
    load_fn = load_fn or D.load
    frames, _sect, _edates = load_fn(market)
    seeds = seeds if seeds is not None else ((0,) if smoke else cfg.SEEDS)
    max_epochs = max_epochs if max_epochs is not None else (cfg.SMOKE_EPOCHS if smoke else cfg.MAX_EPOCHS)
    fold_cap = 1 if smoke else None
    if min_train is None:
        min_train = cfg.MIN_TRAIN_ROWS.get(market, cfg.MIN_TRAIN_ROWS["default"])
    horizons = (1,) if smoke else cfg.HORIZONS
    out = {"market": market, "horizons": list(horizons), "n": {}, "n_folds": {}, "metrics": {},
           "train_metrics": {}, "val_metrics": {}, "fit_diagnostics": {}, "learning_curves": {},
           "qlike": {}, "per_seed_qlike": {}}
    if market == "hose":
        out["per_fold_qlike"] = {}
    for h in horizons:
        t0 = time.time()
        a = D.panel(frames, {}, h)
        res = _horizon(a, h, market, seeds, max_epochs, cfg.PATIENCE, fold_cap, min_train)
        if res is not None:
            _merge(out, res, h, market)
            _checkpoint(out, out_path)
            print(f"  h{h}: GNNHAR QLIKE {res['qlike']['GNNHAR']:.4f} "
                  f"(n={res['n']:,}, {res['n_folds']} folds, {time.time() - t0:.0f}s)", flush=True)
        del a
        gc.collect()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("market", nargs="?", default="hose", choices=("hose", "sp500"))
    ap.add_argument("--smoke", action="store_true", help="1 horizon, 1 fold, 1 seed, few epochs")
    args = ap.parse_args()
    outp = REPO / "results" / "xgb" / f"gnnhar_{args.market}{'_smoke' if args.smoke else ''}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    print(f"loaded market={args.market} device={DEVICE}", flush=True)
    out = run(args.market, out_path=outp, smoke=args.smoke)
    outp.write_text(json.dumps(out, indent=2))
    print(f"saved {outp.relative_to(REPO)}", flush=True)

if __name__ == "__main__":
    main()
