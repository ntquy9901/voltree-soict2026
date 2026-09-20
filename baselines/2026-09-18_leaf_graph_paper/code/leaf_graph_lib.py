"""XGBoost gamma booster + a leaf-cooccurrence graph that smooths its predictions.

Copied verbatim (math unchanged) from the committed proof-of-concept
``baselines/2026-09-18_gbm_leaf_graph/code/leaf_graph.py`` (947296f1); the only edit is the config import, which
points to THIS baseline's single-source config so every tunable constant lives in one place. The committed
baseline is NOT modified.

Mechanism (falsification test): fit an XGBoost ``reg:gamma`` booster (capacity matched to the champion HGBR
GBME) on the causal train window. Each observation's *leaf-vector* is its row of ``pred_leaf`` — the integer
leaf index it lands in for each of the ``T`` trees. Two same-day stocks are neighbours if their leaf-vectors
overlap a lot (fraction of trees landing in the SAME leaf = Hamming similarity). A per-day kNN graph over that
similarity is used to smooth the base prediction:

    y_smooth_i = (1 - alpha) * y_i + alpha * mean_{j in kNN(i)} y_j

with ``alpha`` fit on a validation slice per fold, frozen for test.

Causality: the graph on any day uses only that day's cross-section (same-day leaf-vectors) and a booster fit on
the past train window. No future rows, no cross-day mixing (each date's cross-section is smoothed independently).

Reuses the champion floor from ``full_matrix`` (single source) and the tested per-obs QLIKE from ``metrics``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import xgboost as xgb

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import full_matrix as FM  # noqa: E402  (import first so it registers the submission `metrics` path)
import metrics as M  # noqa: E402
import leaf_graph_paper_config as C  # noqa: E402

FL = FM.FL


def _params(seed):
    """XGBoost ``reg:gamma`` params mirroring the champion HGBR gamma capacity (single-sourced from config)."""
    return {"objective": "reg:gamma", "eta": C.XGB_LR, "max_leaves": C.XGB_MAX_LEAVES,
            "max_depth": C.XGB_MAX_DEPTH, "grow_policy": "lossguide", "lambda": C.XGB_L2,
            "min_child_weight": C.XGB_MIN_CHILD_WEIGHT, "seed": int(seed), "tree_method": "hist",
            "verbosity": 0}


def fit_booster(trf, cols, seed, floor=FL):
    """Fit one XGBoost gamma booster on the causal train rows ``trf`` (floored target). Returns the booster so
    its per-tree leaf indices can be read for the cooccurrence graph."""
    y_tr = np.maximum(trf["y"].to_numpy(float), floor)
    d_tr = xgb.DMatrix(trf[cols].to_numpy(float), label=y_tr)
    return xgb.train(_params(seed), d_tr, num_boost_round=C.XGB_N_ESTIMATORS)


def predict_booster(bst, X, floor=FL):
    """Floored gamma prediction of one booster on feature matrix ``X`` (clipped to [floor, PRED_CAP])."""
    return np.clip(bst.predict(xgb.DMatrix(np.asarray(X, float))), floor, C.PRED_CAP)


def predict_xgb(trf, combo, cols, seeds, floor=FL):
    """Seed-ensembled plain XGBoost gamma base prediction over the ``combo`` rows (this is the SHARED base for
    both the XGB and the XGB+leafgraph model, so the graph effect is isolated)."""
    X = combo[cols].to_numpy(float)
    return np.mean([predict_booster(fit_booster(trf, cols, s, floor), X, floor) for s in seeds], 0)


def leaf_matrix(bst, X):
    """Integer (n_samples x n_trees) leaf-index matrix from a fitted booster (``pred_leaf=True``)."""
    return bst.predict(xgb.DMatrix(np.asarray(X, float)), pred_leaf=True).astype(np.int32)


def day_similarity(leaf_rows):
    """Pairwise leaf-Hamming similarity for one day's cross-section.

    ``leaf_rows`` is (m x T) integer leaf indices for ``m`` stocks over ``T`` trees. Returns an (m x m) matrix
    whose (i, j) entry is the fraction of trees where stock i and stock j land in the SAME leaf. Identical
    leaf-vectors -> 1.0; fully disjoint -> 0.0. Diagonal is 1.0.

    Implemented as a sparse one-hot inner product (S @ S.T) / T over globally-unique (tree, leaf) ids, which is
    exact and far cheaper than an m x m x T broadcast."""
    leaf_rows = np.asarray(leaf_rows, np.int64)
    m, t = leaf_rows.shape
    off = int(leaf_rows.max()) + 1                                 # per-tree id offset -> globally unique ids
    ids = (leaf_rows + np.arange(t, dtype=np.int64) * off).ravel()
    rows = np.repeat(np.arange(m), t)
    s = sp.csr_matrix((np.ones(m * t, np.float32), (rows, ids)), shape=(m, int(ids.max()) + 1))
    return (s @ s.T).toarray() / float(t)


def knn_neighbour_mean(pred, sim, k):
    """Mean base prediction over each stock's top-``k`` leaf-similar neighbours (self excluded).

    A singleton cross-section (m == 1) has no neighbours, so it returns the base prediction unchanged."""
    pred = np.asarray(pred, float)
    m = len(pred)
    if m == 1:
        return pred.copy()
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)                                    # never a neighbour of itself
    kk = min(int(k), m - 1)
    idx = np.argpartition(-s, kk - 1, axis=1)[:, :kk]              # top-kk by similarity (unordered is fine)
    return pred[idx].mean(axis=1)


def smooth_day(pred, leaf_rows, k, alpha):
    """Graph-smooth one day's base predictions: (1-alpha)*pred + alpha*neighbour_mean.

    ``alpha == 0`` (or a singleton day) returns the base unchanged; ``alpha == 1`` returns the neighbour-mean."""
    pred = np.asarray(pred, float)
    if alpha == 0.0 or len(pred) == 1:
        return pred.copy()
    nbr = knn_neighbour_mean(pred, day_similarity(leaf_rows), k)
    return (1.0 - alpha) * pred + alpha * nbr


def smooth_all(pred, leaves, dates, k, alpha):
    """Apply per-day leaf-graph smoothing to every row, grouping by ``dates`` (each date's cross-section is
    smoothed independently -> strictly causal, no cross-day mixing). ``alpha == 0`` is an identity fast path
    that skips building any graph (exact, and the expected NO-GO case)."""
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
    """Pick the smoothing weight in ``grid`` that minimises validation QLIKE (frozen for test). Returns
    ``(best_alpha, best_qlike)``; ties keep the smaller alpha (grid is ascending, strict ``<`` update)."""
    best_a, best_q = grid[0], float("inf")
    for a in grid:
        q = float(np.mean(M.per_obs_qlike(y, smooth_all(pred, leaves, dates, k, a), floor=floor)))
        if q < best_q:
            best_a, best_q = a, q
    return best_a, best_q
