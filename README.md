# Stock Volatility Forecasting — Gradient-Boosted Trees with Leaf-Graph Smoothing

A pooled stock-day model for forecasting daily range-based (Parkinson) variance. A gamma-loss
gradient-boosted tree ensemble (XGBoost) is trained on endogenous own-history features; two optional
components can be added on top: a block of earnings-calendar features, and a leaf-cooccurrence graph
that smooths each trading day's cross-section of forecasts. Forecast horizons are 1, 5, 10, and 22
trading days; two equity universes are supported (a large U.S. universe and a Vietnamese exchange).

## Requirements

Python 3.10+ with:

    pip install numpy pandas scikit-learn xgboost pyarrow

## Input data

Compact input panels are provided in `data/panels/`. Expand them into the per-ticker layout the code
reads:

    python rebuild_data.py

This writes `data/processed_enriched/<universe>/<ticker>.csv`. Every model feature is derived from these
inputs at run time.

## Running the models

Each command runs the full walk-forward evaluation for one universe.

Endogenous model with the leaf-cooccurrence graph (no earnings block):

    python baselines/leaf_graph/run_leaf_graph.py hose  --featureset noearn
    python baselines/leaf_graph/run_leaf_graph.py sp500 --featureset noearn

Adding the earnings block with the cadence-predicted (origin-time) release schedule:

    python baselines/cadence_earnings/run_expected.py

Running `run_leaf_graph.py <universe> --featureset full` instead adds the earnings block using the
*realized* (ex-post) release dates — a diagnostic upper bound on the earnings signal.

Outputs are written under `results/xgb/` (the cadence-earnings variant under
`results/xgb/cadence_earnings/`) as one JSON per universe and horizon, each holding QLIKE and
squared/absolute error metrics, a Diebold-Mariano p-value, per-regime robustness, and the fitted graph
weight. The endogenous and cadence-earnings outputs are shipped for direct inspection; the realized-date
variant is regenerated on demand by the command above.

## Code map

| Component | File |
|---|---|
| Gradient-boosted model, walk-forward evaluation, leaf-graph smoothing | `baselines/leaf_graph/run_leaf_graph.py` |
| Leaf-cooccurrence graph (kNN over shared tree leaves) | `baselines/leaf_graph/leaf_graph_lib.py` |
| Model configuration (windows, embargo, grids, thresholds) | `baselines/leaf_graph/leaf_graph_config.py` |
| Cadence-predicted earnings schedule | `baselines/cadence_earnings/expected_schedule.py` |
| Feature panel (endogenous + earnings features) | `scripts/eda/full_matrix.py` |
| Graph neural network over HAR features | `scripts/eda/vn_gbm_graph_stage1.py` |
| Linear (HAR) benchmark | `baselines/har_baseline/full_compare.py` |
| Earnings event study | `scripts/eda/earnings_event_study.py` |
| Diebold-Mariano test; shared statistics/helpers | `baselines/common/` |

## Data sources

Daily open-high-low-close prices: Yahoo Finance (U.S. universe) and the `vnstock` feed (Vietnamese
universe). Earnings-announcement dates come from public securities-disclosure feeds. All inputs live
under `data/`: compact price panels in `data/panels/` and earnings dates in `data/earnings/`.
