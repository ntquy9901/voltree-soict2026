"""Tests for the FINAL paper-grade leaf-cooccurrence-graph baseline (model set = {XGB, XGB+leafgraph} ONLY, no
GBME; 3 seeds; full vs no-earn feature-set switch). Cover: leaf-Hamming similarity, kNN neighbour-mean,
alpha-smoothing (identity at 0, neighbour-mean at 1, monotone, causal per-day), booster plumbing + numerical
guard, the feature-set switch (noearn drops EARN), the streaming metric accumulator, 3-seed ensembling,
DM/verdict/spike logic, and a walk-forward run smoke carrying the gate-required over/under-fit evidence keys.
Synthetic panels keep the driver fast without real data."""
import numpy as np
import pandas as pd

import leaf_graph_paper_config as C
import leaf_graph_lib as LG
import run_leaf_graph_paper as R
import vn_gbm_graph_stage1 as S1


# --------------------------------------------------------------------------- synthetic data
def _ticker_frame(seed, start="2021-01-01", end="2023-06-30"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    pk = np.empty(n)
    pk[0] = 3e-4
    for t in range(1, n):
        pk[t] = max(1e-6, 0.85 * pk[t - 1] + 0.15 * 3e-4 + rng.normal(0, 3e-5))
    s = pd.Series(pk)
    base = pd.DataFrame({
        "date": dates, "parkinson_variance": pk,
        "har_daily": pk, "har_weekly": s.rolling(5, min_periods=1).mean().to_numpy(),
        "har_monthly": s.rolling(22, min_periods=1).mean().to_numpy(),
        "volume_zscore_22": rng.normal(0, 1, n), "market_pk": pk * 0.9,
        "daily_return": rng.normal(0, 0.01, n)})
    return S1._feat(base)


def _frames(n_tickers=4):
    return {f"TK{i}": _ticker_frame(seed=i).assign(ticker=f"TK{i}", sector=0) for i in range(n_tickers)}


def _fake_loader(market):
    return _frames(), {f"TK{i}": 0 for i in range(4)}, {}


def _tiny(monkeypatch):
    monkeypatch.setattr(C, "MIN_ROWS", {"sp500": 10, "default": 10})


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 2))
    y = np.exp(-8.0 + 0.5 * x[:, 0])                       # positive target, log-linear in x0
    df = pd.DataFrame({"f0": x[:, 0], "f1": x[:, 1], "y": y})
    return df.iloc[:300], df.iloc[300:]


# --------------------------------------------------------------------------- leaf-Hamming similarity
def test_day_similarity_identical_and_disjoint():
    ident = np.array([[3, 7, 1, 4], [3, 7, 1, 4]])          # same leaf in every tree -> similarity 1
    sim = LG.day_similarity(ident)
    assert sim.shape == (2, 2)
    assert np.allclose(sim, 1.0)                            # incl. diagonal
    disjoint = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])       # no shared leaf in any tree -> off-diagonal 0
    s2 = LG.day_similarity(disjoint)
    assert np.isclose(s2[0, 1], 0.0) and np.isclose(s2[1, 0], 0.0)
    assert np.allclose(np.diag(s2), 1.0)


def test_day_similarity_partial_overlap_is_fraction():
    # 2 of 4 trees match -> similarity 0.5 (exact Hamming fraction)
    L = np.array([[1, 2, 3, 4], [1, 2, 9, 8]])
    assert np.isclose(LG.day_similarity(L)[0, 1], 0.5)


# --------------------------------------------------------------------------- kNN neighbour-mean
def test_knn_neighbour_mean_picks_top_k_excluding_self():
    pred = np.array([10.0, 20.0, 30.0])
    # stock0 most similar to stock1, then stock2; k=1 -> neighbour is stock1 -> mean 20
    sim = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])
    nm = LG.knn_neighbour_mean(pred, sim, k=1)
    assert np.isclose(nm[0], 20.0)
    # k=2 for stock0 -> neighbours {1,2} -> mean 25
    assert np.isclose(LG.knn_neighbour_mean(pred, sim, k=2)[0], 25.0)


def test_knn_neighbour_mean_singleton_returns_base():
    assert np.allclose(LG.knn_neighbour_mean(np.array([5.0]), np.array([[1.0]]), k=3), [5.0])


# --------------------------------------------------------------------------- smoothing
def _one_day_leaves():
    # 3 stocks; stock0 & stock1 share all leaves, stock2 disjoint
    return np.array([[1, 2, 3, 4], [1, 2, 3, 4], [5, 6, 7, 8]])


def test_smooth_day_alpha0_is_base_alpha1_is_neighbour_mean():
    pred = np.array([10.0, 40.0, 100.0])
    leaves = _one_day_leaves()
    assert np.allclose(LG.smooth_day(pred, leaves, C.K_NEIGHBOURS, 0.0), pred)          # alpha=0 -> base
    # alpha=1 -> neighbour-mean; stock0's most-similar neighbour is stock1 (identical leaves)
    nm = LG.smooth_day(pred, leaves, k=1, alpha=1.0)
    assert np.isclose(nm[0], 40.0) and np.isclose(nm[1], 10.0)


def test_smooth_day_singleton_returns_base():
    assert np.allclose(LG.smooth_day(np.array([7.0]), np.array([[1, 2, 3]]), C.K_NEIGHBOURS, 0.5), [7.0])


def test_smooth_day_is_monotone_in_alpha():
    pred = np.array([10.0, 40.0, 100.0])
    leaves = _one_day_leaves()
    s_lo = LG.smooth_day(pred, leaves, k=1, alpha=0.2)[0]
    s_hi = LG.smooth_day(pred, leaves, k=1, alpha=0.8)[0]
    # stock0 pulled toward its neighbour (40 > 10): larger alpha -> larger smoothed value, both above base
    assert 10.0 < s_lo < s_hi < 40.0


def test_smooth_all_groups_by_date_and_alpha0_fastpath():
    pred = np.array([10.0, 40.0, 100.0, 200.0])
    # two dates; day A = stocks 0,1 (identical leaves), day B = stocks 2,3 (identical leaves)
    leaves = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]])
    dates = np.array(["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"])
    assert np.allclose(LG.smooth_all(pred, leaves, dates, C.K_NEIGHBOURS, 0.0), pred)   # identity fast path
    sm = LG.smooth_all(pred, leaves, dates, k=1, alpha=1.0)
    # cross-day never mixed: day A rows swap (10<->40), day B rows swap (100<->200)
    assert np.allclose(sm, [40.0, 10.0, 200.0, 100.0])


def test_fit_alpha_prefers_lower_qlike_and_returns_grid_member():
    # neighbour-mean is a WORSE forecast here (targets are per-stock persistent), so alpha=0 should win
    rng = np.random.default_rng(3)
    m = 6
    pred = np.array([1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4])
    y = pred.copy()                                          # base is exact -> any smoothing raises QLIKE
    leaves = np.tile(np.arange(m)[:, None], (1, 8)).astype(int) + rng.integers(0, 2, (m, 8))
    dates = np.array(["2021-01-01"] * m)
    a, q = LG.fit_alpha(y, pred, leaves, dates, C.K_NEIGHBOURS, C.ALPHA_GRID, LG.FL)
    assert a == 0.0 and a in C.ALPHA_GRID and q >= 0.0


def test_fit_alpha_can_select_positive_alpha_when_smoothing_helps():
    # construct a day where the neighbour-mean is a strictly better forecast than the (noisy) base
    truth = np.array([2e-4, 2e-4, 2e-4, 2e-4])
    pred = np.array([1e-4, 3e-4, 1e-4, 3e-4])               # base noisy around truth
    leaves = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])   # all mutually similar -> mean ~ truth
    dates = np.array(["2021-01-01"] * 4)
    a, _ = LG.fit_alpha(truth, pred, leaves, dates, C.K_NEIGHBOURS, C.ALPHA_GRID, LG.FL)
    assert a > 0.0


# --------------------------------------------------------------------------- booster plumbing + guard
def test_fit_predict_booster_positive_and_leaf_matrix_shape():
    tr, te = _toy()
    bst = LG.fit_booster(tr, ["f0", "f1"], seed=0)
    p = LG.predict_booster(bst, te[["f0", "f1"]].to_numpy(float))
    assert (p > 0).all() and np.isfinite(p).all() and (p <= C.PRED_CAP + 1e-12).all()
    L = LG.leaf_matrix(bst, te[["f0", "f1"]].to_numpy(float))
    assert L.shape == (len(te), C.XGB_N_ESTIMATORS) and L.dtype == np.int32


def test_predict_booster_clips_to_cap(monkeypatch):
    # a tiny cap forces the clip upper bound to bite -> guards against gamma exp-link overflow
    tr, te = _toy()
    bst = LG.fit_booster(tr, ["f0", "f1"], seed=0)
    monkeypatch.setattr(C, "PRED_CAP", 1e-4)
    p = LG.predict_booster(bst, te[["f0", "f1"]].to_numpy(float))
    assert (p <= 1e-4 + 1e-18).all() and (p >= LG.FL).all()


def test_predict_xgb_seed_ensemble_positive():
    tr, te = _toy()
    combo = pd.concat([te, tr])
    p = LG.predict_xgb(tr, combo, ["f0", "f1"], (0, 1, 2))
    assert p.shape == (len(combo),) and (p > 0).all() and np.isfinite(p).all()


# --------------------------------------------------------------------------- causality
def test_smoothing_is_per_day_causal():
    # a test day's smoothing must depend ONLY on that day's rows: perturbing another day's predictions
    # leaves the target day's smoothed output unchanged.
    pred = np.array([10.0, 40.0, 100.0, 200.0])
    leaves = np.array([[1, 1], [1, 1], [2, 2], [2, 2]])
    dates = np.array(["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"])
    base = LG.smooth_all(pred, leaves, dates, k=1, alpha=0.5)
    pred2 = pred.copy(); pred2[2:] = [999.0, 888.0]         # change ONLY day-2 rows
    other = LG.smooth_all(pred2, leaves, dates, k=1, alpha=0.5)
    assert np.allclose(base[:2], other[:2])                 # day-1 output invariant to day-2 changes


# --------------------------------------------------------------------------- feature-set switch (ablation)
def test_resolve_cols_full_includes_earn_when_present():
    cols, use_earn = R.resolve_cols("full", has_earn=True)
    assert use_earn and cols == R.OWN + R.FM.EARN and len(cols) == len(R.OWN) + 4


def test_resolve_cols_noearn_drops_earn():
    cols, use_earn = R.resolve_cols("noearn", has_earn=True)
    assert not use_earn and cols == R.OWN and all(c not in cols for c in R.FM.EARN)


def test_resolve_cols_full_without_earn_falls_back_to_own():
    cols, use_earn = R.resolve_cols("full", has_earn=False)     # no earnings available -> OWN only
    assert not use_earn and cols == R.OWN


# --------------------------------------------------------------------------- streaming accumulator
def test_stream_matches_direct_pooled_metrics():
    rng = np.random.default_rng(7)
    y = np.abs(rng.normal(3e-4, 1e-4, 500)) + 1e-5
    p = np.abs(rng.normal(3e-4, 1e-4, 500)) + 1e-5
    st = R._Stream()
    for lo in range(0, 500, 137):                              # fold-in ragged chunks then finalize
        st.update(y[lo:lo + 137], p[lo:lo + 137])
    got = st.finalize()
    want = R._metrics5(y, p)
    for kk in ("mse", "rmse", "mae", "r2", "qlike"):
        assert np.isclose(got[kk], want[kk], rtol=1e-9, atol=1e-12), kk


def test_stream_zero_variance_target_r2_zero():
    st = R._Stream()
    st.update(np.array([2e-4, 2e-4, 2e-4]), np.array([1e-4, 3e-4, 2e-4]))   # ss_tot == 0 -> r2 branch = 0.0
    assert st.finalize()["r2"] == 0.0


def test_metrics5_zero_variance_target_r2_zero():
    m = R._metrics5(np.array([2e-4, 2e-4, 2e-4]), np.array([1e-4, 3e-4, 2e-4]))
    assert m["r2"] == 0.0 and set(m) == {"mse", "rmse", "mae", "r2", "qlike"}


# --------------------------------------------------------------------------- pure driver logic
def test_verdict_and_success():
    assert R.verdict(0.5, 0.01) and not R.verdict(-0.1, 0.01) and not R.verdict(0.5, 0.20)
    assert R.success({1: {"verdict": {"beats": True}}, 5: {"verdict": {"beats": True}}})
    assert not R.success({1: {"verdict": {"beats": True}}})            # missing h5 -> fail


def test_safe_dm_degenerate_and_valueerror(monkeypatch):
    e = np.array([1.0, 2.0, 3.0, 4.0]); d = pd.to_datetime(["2021-01-01"] * 4).to_numpy()
    assert R._safe_dm(e, e.copy(), d, 1)["p_value"] == 1.0            # identical loss -> degenerate p=1
    monkeypatch.setattr(R.ST, "date_clustered_dm", lambda *a, **k: (_ for _ in ()).throw(ValueError("hln")))
    assert R._safe_dm(e, e + 1.0, d, 1)["p_value"] == 1.0             # DM raises -> degenerate


def test_safe_dm_delegates_when_non_degenerate(monkeypatch):
    monkeypatch.setattr(R.ST, "date_clustered_dm", lambda a, b, d, h: {"p_value": 0.033})
    e = np.array([1.0, 2.0, 3.0, 4.0]); d = pd.to_datetime(["2021-01-01"] * 4).to_numpy()
    assert R._safe_dm(e, e + 1.0, d, 1)["p_value"] == 0.033


def test_spike_mask_flags_windows():
    d = pd.to_datetime(["2019-06-01", "2020-03-15", "2022-06-01", "2025-04-10"]).to_numpy()
    assert list(R._spike_mask(d)) == [False, True, True, True]


def test_load_earn_sp500_passthrough_and_hose_missing(monkeypatch, tmp_path):
    assert R._load_earn("sp500", {"A": 1}) == {"A": 1}                # sp500 keeps its own edates
    monkeypatch.setattr(R, "REPO", tmp_path)                         # no crawled parquet under tmp
    assert R._load_earn("hose", {}) == {}                            # hose w/o parquet -> edates unchanged


def test_load_earn_hose_reads_real_parquet():
    ed = R._load_earn("hose", {})                                    # real crawled VN parquet present in repo
    assert isinstance(ed, dict) and len(ed) > 0                      # tickers -> announcement-date arrays


def test_run_rejects_unknown_feature_set():
    import pytest
    with pytest.raises(ValueError, match="feature_set must be"):
        R.run("hose", feature_set="bogus", load_fn=_fake_loader, smoke=True)


# --------------------------------------------------------------------------- run smoke
def test_run_hose_full_structure_evidence_perfold_spike(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    # non-overlapping spike window so the fold's test dates survive exclusion -> spike_robustness is computed
    monkeypatch.setattr(C, "SPIKE_WINDOWS", (("1990-01-01", "1990-01-02"),))
    docs = R.run("hose", feature_set="full", load_fn=_fake_loader, out_dir=tmp_path, smoke=True)
    assert set(docs) == {1}
    doc = docs[1]
    assert doc["feature_set"] == "full"
    assert (tmp_path / "leaf_graph_paper_hose_full_smoke_h1.json").exists()
    assert R.ORDER == ["XGB", "XGB+leafgraph"] and "GBME" not in doc["metrics"]
    for blk in ("metrics", "train_metrics", "val_metrics"):
        for m in R.ORDER:
            assert set(doc[blk][m]) == {"mse", "rmse", "mae", "r2", "qlike"}
    assert set(doc["dm"]) == {"XGB+leafgraph_vs_XGB"}
    assert set(doc["gain_pct"]) == {"XGB+leafgraph_vs_XGB"}
    assert "per_fold_qlike" in doc and "spike_robustness" in doc and "verdict" in doc
    assert "alpha" in doc and set(doc["alpha"]) == {"per_fold", "mean"}
    import overfit_check as OF
    _ok, probs = OF.check_result_evidence(doc)
    assert all("missing" not in p for p in probs), probs


def test_run_hose_noearn_excludes_earn_features(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    docs = R.run("hose", feature_set="noearn", load_fn=_fake_loader, out_dir=tmp_path, smoke=True)
    doc = docs[1]
    assert doc["feature_set"] == "noearn"
    assert all(c not in doc["features"] for c in R.FM.EARN)          # leave-one-out of EARN
    assert (tmp_path / "leaf_graph_paper_hose_noearn_smoke_h1.json").exists()


def test_fold_predictions_shares_base_and_alpha0_isolates_graph(monkeypatch):
    # graph-isolation invariant: XGBLG must reuse the SAME XGB base; when fit_alpha selects 0 the two arms are
    # bit-identical (only the graph, not a different base, can move XGBLG off XGB).
    monkeypatch.setattr(R.LG, "fit_alpha", lambda *a, **k: (0.0, 0.0))   # force alpha=0 (identity)
    tr, te = _toy(n=200)
    tr = tr.assign(date=pd.Timestamp("2021-01-01"))
    te = te.assign(date=pd.Timestamp("2021-02-01"))
    trf_e, vaf = tr.iloc[:150], tr.iloc[150:]
    out, alpha = R._fold_predictions(trf_e, vaf, te, ["f0", "f1"], (0, 1, 2))
    assert alpha == 0.0
    for s in ("te", "tr", "va"):
        assert np.allclose(out[s][R.XGB], out[s][R.XGBLG])              # alpha=0 -> XGBLG == XGB base exactly


def test_run_sp500_no_spike_block(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    docs = R.run("sp500", feature_set="full", load_fn=_fake_loader, out_dir=tmp_path, smoke=True)
    doc = docs[1]
    assert "per_fold_qlike" not in doc and "spike_robustness" not in doc
    assert (tmp_path / "leaf_graph_paper_sp500_full_smoke_h1.json").exists()


def test_run_skips_empty_folds_multi(monkeypatch, tmp_path):
    # non-smoke: synthetic data ends mid-2023, so later S1.FOLDS have empty test windows -> the skip `continue`
    # fires while early folds run (multi-fold pooling).
    _tiny(monkeypatch)
    monkeypatch.setattr(C, "HORIZONS", (1,))
    docs = R.run("sp500", feature_set="full", load_fn=_fake_loader, out_dir=tmp_path, smoke=False)
    assert docs[1]["n_folds"] >= 1


def test_run_omits_robustness_when_all_test_in_spike(monkeypatch, tmp_path):
    # default 2022 spike window covers all of fold0's test dates -> keep empty -> spike_robustness omitted
    _tiny(monkeypatch)
    docs = R.run("hose", feature_set="full", load_fn=_fake_loader, out_dir=tmp_path, smoke=True)
    assert "spike_robustness" not in docs[1] and "per_fold_qlike" in docs[1]
