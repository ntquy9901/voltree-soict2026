"""Re-run the paper's XGB+E / XGB+E+LG models with EXPECTED-SCHEDULE earnings (strictly leakage-safe)
instead of realized (actual) dates, for both markets, writing NEW result JSONs (the actual-date baseline
is left untouched).

Reuses the 2026-09-18 leaf-graph runner's full machinery (walk-forward folds, 3-seed ensembles, leaf-graph
smoothing, DM, spike-robustness, over/under-fit evidence) by injecting an expected-schedule earnings loader
in place of the module's ``_load_earn`` (a runtime override in THIS process; the sibling baseline's file is
not modified). Output: ``results/xgb/expected_schedule/leaf_graph_paper_<market>_full_h*.json``.

The XGB (no-earn) column is earnings-independent, so it is NOT re-run here -- it reuses the committed
``leaf_graph_paper_<market>_noearn_h*.json`` unchanged.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parent
for _p in (str(REPO), str(_CODE),
           str(REPO / "baselines" / "2026-09-18_leaf_graph_paper" / "code"),
           str(REPO / "baselines" / "2026-09-18_gbm_leaf_graph" / "code"),
           str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "common" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from expected_schedule import expected_schedule, hose_quarterly_dates  # noqa: E402
# NB: run_leaf_graph_paper is imported lazily inside main() (heavy deps) so this module stays cheap to
# import for unit-testing expected_load_earn.

OUT_DIR = REPO / "results" / "xgb" / "expected_schedule"


def expected_load_earn(market, edates):
    """Replace realized earnings dates with the strictly-causal expected schedule.

    SP500: predict from the yfinance dates' own cadence. HOSE: first reduce to one clean quarterly event
    per (ticker, fiscal-year, quarter) (earliest across scopes), then predict from that cadence."""
    if market == "sp500":
        return expected_schedule(edates)
    return expected_schedule(hose_quarterly_dates())


def main(smoke=False, markets=("sp500", "hose")):  # pragma: no cover - data-driven driver (full re-run)
    import run_leaf_graph_paper as R                        # lazy: heavy deps, only needed for the real run
    R._load_earn = expected_load_earn                      # inject expected-schedule earnings
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for market in markets:
        R.run(market, feature_set="full", out_dir=OUT_DIR, smoke=smoke)


if __name__ == "__main__":  # pragma: no cover
    main(smoke="--smoke" in sys.argv)
