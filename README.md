# VolTree — Source code and reproduction data

Source code and reproduction data for the paper **VolTree: Earnings-Aware Gradient Boosting with
Leaf-Graph Smoothing for Stock Volatility Forecasting**.

The tree keeps the same relative path layout the code expects, so the `sys.path` bootstraps in each
script resolve correctly when run in place. This repository is self-contained: it ships the compact
reproduction panels needed to regenerate the paper's numbers end-to-end.

## Reproducing the paper's numbers

Requirements: Python 3.10+ and `pip install numpy pandas scikit-learn xgboost pyarrow pytest`.

1. Rebuild the per-ticker price panels the model reads, from the compact sharded panels in `data/repro/`:

       python rebuild_data.py

   This writes `data/processed_enriched/{sp500_clean,hose}/<ticker>.csv`. These are the exact model
   inputs — verified: the 8 endogenous (OWN) features are bit-for-bit identical to the full enriched
   data; every other feature (log-volatility momentum, earnings, leaf graph) is computed by the code.

2. Reproduce the model results (all horizons, expanding-window walk-forward):

   - **Headline leak-free earnings** — the XGB+E and VolTree rows in Tables 1-2, using the
     cadence-predicted (origin-time) release schedule:

         python baselines/2026-09-19_expected_schedule/code/run_expected.py

     writes `results/gamma_gbm/expected_schedule/leaf_graph_paper_<market>_full_h<h>.json`.

   - **Endogenous and leaf-graph variants** — the XGB and XGB+leaf-graph rows (Table 2 `noearn`), plus
     the realized-date upper bound:

         python baselines/2026-09-18_leaf_graph_paper/code/run_leaf_graph_paper.py hose
         python baselines/2026-09-18_leaf_graph_paper/code/run_leaf_graph_paper.py sp500

     writes `results/gamma_gbm/leaf_graph_paper_<market>_<full|noearn>_h<h>.json`. (The `full` output
     here uses *realized* dates = the optimistic upper bound; the headline uses the expected schedule
     above.)

   Each JSON carries the DM p-value, QLIKE gain, spike-robustness, and per-fold weight alpha.

3. Pre-computed result JSONs are shipped for direct verification (no re-run needed), under
   `results/gamma_gbm/` and `results/gamma_gbm/expected_schedule/`: `leaf_graph_paper_*` (VolTree +
   ablations), `expected_schedule/leaf_graph_paper_*_full` (leak-free headline), `earnings_dm_sp500`
   (SP500 earnings DM), `garch_*`, `gnnhar_*`, `full_compare_*` (HAR). Every result JSON is slimmed to
   the models the paper reports (HAR, GARCH, GNNHAR, XGB, XGB+E, VolTree).

## Code map (paper component -> file)

| Paper component | File |
|---|---|
| **VolTree core** — walk-forward gamma-XGBoost vs XGB+leaf-graph, DM, spike-robustness, alpha fit (Table 2) | `baselines/2026-09-18_leaf_graph_paper/code/run_leaf_graph_paper.py` |
| Leaf-cooccurrence graph (kNN on leaf-Hamming similarity; smooths the forecast) | `baselines/2026-09-18_leaf_graph_paper/code/leaf_graph_lib.py` |
| Config constants (walk-forward, embargo, MIN_ROWS, alpha grid, spike windows) | `baselines/2026-09-18_leaf_graph_paper/code/leaf_graph_paper_config.py` |
| **Leak-free earnings schedule** (release dates predicted at the forecast origin) | `baselines/2026-09-19_expected_schedule/code/expected_schedule.py` |
| Earnings Diebold-Mariano audit | `baselines/2026-09-19_expected_schedule/code/earnings_dm.py` |
| Feature panel + OWN-8 + earnings features | `scripts/eda/full_matrix.py` |
| Train-only correlation graph (GNNHAR) + fold boundaries | `scripts/eda/vn_gbm_graph_stage1.py` |
| HAR baseline (Table 1, HAR row) | `baselines/2026-09-13_paper_models/code/full_compare.py` (shipped `full_compare_*.json` slimmed to the HAR row) |
| Earnings event study (Fig 2) | `scripts/eda/earnings_event_study.py` |
| Cadence diagnostics | `scripts/eda/earnings_pit_cadence.py` |
| Diebold-Mariano test (date-clustered) | `submission/soict_lstm_gat/metrics.py` |
| Over/under-fit evidence check | `scripts/quality_gate/overfit_check.py` |
| OWN-8 feature list (single source) | `baselines/2026-09-13_paper_models/code/config.py` |
| Helper statistics | `baselines/2026-08-21_har_anchored_residual/code/stats.py` |

The GARCH and GNNHAR baseline results are shipped as `results/gamma_gbm/garch_*.json` and
`gnnhar_*.json` for verification.

## Tests

Run each test directory separately (running two baselines together triggers a duplicate-`conftest`
name clash — a pytest limitation, not a code bug):

    python -m pytest baselines/2026-09-19_expected_schedule/test -q     # leak-free schedule + earnings DM
    python -m pytest baselines/2026-09-18_leaf_graph_paper/test  -q     # leaf-graph walk-forward
    python -m pytest scripts/eda/test_earnings_event_study.py    -q     # event study (Fig 2)

Most tests monkeypatch the data loader (synthetic data) to check the architecture and leakage-safety;
a few read the small real samples shipped here (`data/raw/vn_earnings/hose_disclosures.csv`,
`results/gamma_gbm/*.parquet`).

## Data provenance

Prices: Yahoo Finance (S&P 500), vnstock (HOSE). Earnings dates: State Securities Commission portal
plus a vnstock feed (`data/raw/vn_earnings/`, `results/gamma_gbm/*earnings*.parquet`). Sectors:
`results/gamma_gbm/sp500_sectors.json`, `baselines/2026-08-29_sector_gat_ablation/vn_icb_sectors.csv`.
