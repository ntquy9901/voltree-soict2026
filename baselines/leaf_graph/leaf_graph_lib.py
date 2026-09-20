from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import xgboost as xgb

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[1]
for _p in (str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "common"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import feature_panel as D
import metrics as M
import leaf_graph_config as C

FL = D.FL

def _params(seed):
    return {"objective": "reg:gamma", "eta": C.XGB_LR, "max_leaves": C.XGB_MAX_LEAVES,
            "max_depth": C.XGB_MAX_DEPTH, "grow_policy": "lossguide", "lambda": C.XGB_L2,
            "min_child_weight": C.XGB_MIN_CHILD_WEIGHT, "seed": int(seed), "tree_method": "hist",
            "verbosity": 0}

def fit_booster(trf, cols, seed, floor=FL):
    y_tr = np.maximum(trf["y"].to_numpy(float), floor)
    d_tr = xgb.DMatrix(trf[cols].to_numpy(float), label=y_tr)
    return xgb.train(_params(seed), d_tr, num_boost_round=C.XGB_N_ESTIMATORS)

def predict_booster(bst, X, floor=FL):
    return np.clip(bst.predict(xgb.DMatrix(np.asarray(X, float))), floor, C.PRED_CAP)

def predict_xgb(trf, combo, cols, seeds, floor=FL):
    X = combo[cols].to_numpy(float)
    return np.mean([predict_booster(fit_booster(trf, cols, s, floor), X, floor) for s in seeds], 0)

def leaf_matrix(bst, X):
    return bst.predict(xgb.DMatrix(np.asarray(X, float)), pred_leaf=True).astype(np.int32)

def day_similarity(leaf_rows):
    leaf_rows = np.asarray(leaf_rows, np.int64)
    m, t = leaf_rows.shape
    off = int(leaf_rows.max()) + 1
    ids = (leaf_rows + np.arange(t, dtype=np.int64) * off).ravel()
    rows = np.repeat(np.arange(m), t)
    s = sp.csr_matrix((np.ones(m * t, np.float32), (rows, ids)), shape=(m, int(ids.max()) + 1))
    return (s @ s.T).toarray() / float(t)

def knn_neighbour_mean(pred, sim, k):
    pred = np.asarray(pred, float)
    m = len(pred)
    if m == 1:
        return pred.copy()
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    kk = min(int(k), m - 1)
    idx = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
    return pred[idx].mean(axis=1)

def smooth_day(pred, leaf_rows, k, alpha):
    pred = np.asarray(pred, float)
    if alpha == 0.0 or len(pred) == 1:
        return pred.copy()
    nbr = knn_neighbour_mean(pred, day_similarity(leaf_rows), k)
    return (1.0 - alpha) * pred + alpha * nbr

def smooth_all(pred, leaves, dates, k, alpha):
    pred = np.asarray(pred, float)
    out = pred.copy()
    if alpha == 0.0:
        return out
    dates = np.asarray(dates)
    for d in np.unique(dates):
        mask = dates == d
        out[mask] = smooth_day(pred[mask], leaves[mask], k, alpha)
    return out

def fit_alpha(y, pred, leaves, dates, k, grid, floor=FL):
    best_a, best_q = grid[0], float("inf")
    for a in grid:
        q = float(np.mean(M.per_obs_qlike(y, smooth_all(pred, leaves, dates, k, a), floor=floor)))
        if q < best_q:
            best_a, best_q = a, q
    return best_a, best_q
