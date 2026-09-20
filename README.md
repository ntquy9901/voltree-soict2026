# VolTree: Earnings-Aware Gradient Boosting with Leaf-Graph Smoothing for Stock Volatility Forecasting

> Anonymous repository accompanying a paper under double-blind review. Author, citation, and license
> information will be added upon acceptance.

VolTree is a pooled stock-day model for forecasting daily range-based (Parkinson) variance. A gamma-loss
gradient-boosted tree ensemble (XGBoost) is trained on endogenous own-history features, and we ask when
additional information provides incremental value. The endogenous model outperforms the HAR linear
benchmark, per-stock GARCH, and a learned graph neural network (GNNHAR) on QLIKE at every horizon (1, 5,
10, and 22 trading days). We then examine two optional components across two structurally different
markets: a cadence-estimated earnings block, and a leaf-cooccurrence graph that smooths each trading
day's cross-section of forecasts using the boosted model's own tree leaves, with a validation-fit
smoothing weight. On the S&P 500, the earnings feature reliably lowers QLIKE by a further 2.7–3.2% at
every horizon; on Vietnam's Ho Chi Minh Stock Exchange (HOSE), it gives no measurable improvement.
Conversely, the leaf graph lowers HOSE QLIKE at the one- and five-day horizons, while its learned weight
collapses toward zero on the S&P 500. The two components thus provide incremental gains in different
markets rather than universally additive improvements.

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

`run_leaf_graph.py <universe> [--featureset full|noearn] [--realized]` runs the full walk-forward
evaluation for one universe (`sp500` or `hose`). `--featureset` takes two values:

`noearn` — endogenous own-history features only, with the leaf-cooccurrence graph:

    python baselines/leaf_graph/run_leaf_graph.py hose  --featureset noearn
    python baselines/leaf_graph/run_leaf_graph.py sp500 --featureset noearn

`full` — adds the earnings block. By default it uses the cadence-predicted (origin-time, leakage-safe)
release schedule; this is the main earnings model:

    python baselines/leaf_graph/run_leaf_graph.py hose  --featureset full
    python baselines/leaf_graph/run_leaf_graph.py sp500 --featureset full

Adding `--realized` uses realized (ex-post) release dates instead — a diagnostic upper bound on the
earnings signal:

    python baselines/leaf_graph/run_leaf_graph.py sp500 --featureset full --realized

Outputs are written under `results/xgb/`: the endogenous (`noearn`) variants at the top level, the
cadence-earnings (`full`) variant under `results/xgb/cadence_earnings/`, and the realized-date variant
under `results/xgb/realized_earnings/`. Each JSON (one per universe and horizon) holds QLIKE and
squared/absolute error metrics, a Diebold-Mariano p-value, per-regime robustness, and the fitted graph
weight. All are shipped for direct inspection.

## Model variants inside each result JSON

**Every run writes two arms in the same JSON: the un-smoothed base model, and its leaf-graph-smoothed
version** (the leaf graph is always fitted on top of the base, with its weight selected on validation).
The earnings block is chosen by `--featureset`. The four ablation variants therefore come from two runs,
two keys each:

| Variant | `--featureset` | JSON key |
|---|---|---|
| Endogenous only (XGB) | `noearn` | `metrics["XGB"]` |
| Endogenous + leaf graph (XGB+LG) | `noearn` | `metrics["XGB+leafgraph"]` |
| Endogenous + earnings (XGB+E) | `full` | `metrics["XGB"]` |
| Endogenous + earnings + leaf graph (XGB+E+LG, the full model) | `full` | `metrics["XGB+leafgraph"]` |

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
