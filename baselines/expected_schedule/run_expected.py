import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_CODE = Path(__file__).resolve().parent
for _p in (str(REPO), str(_CODE),
           str(REPO / "baselines" / "leaf_graph"),
           str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from expected_schedule import expected_schedule, hose_quarterly_dates

OUT_DIR = REPO / "results" / "xgb" / "expected_schedule"

def expected_load_earn(market, edates):
    if market == "sp500":
        return expected_schedule(edates)
    return expected_schedule(hose_quarterly_dates())

def main(smoke=False, markets=("sp500", "hose")):
    import run_leaf_graph as R
    R._load_earn = expected_load_earn
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for market in markets:
        R.run(market, feature_set="full", out_dir=OUT_DIR, smoke=smoke)

if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
