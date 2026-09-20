"""Proper audit-trail DM for the leak-free EARNINGS increment on the S&P 500 headline pipeline.

The leaf-graph runner only records the leaf-graph DM (XGB+leafgraph vs XGB) per feature set; it never
DM-tests the earnings marginal (XGB+E vs XGB-noearn) because ``full`` and ``noearn`` are separate runs. This
script recomputes both arms' seed-ensembled XGB base predictions on the IDENTICAL walk-forward test rows
(same folds/embargo/train-minus-val/seeds as ``run_leaf_graph_paper``), using the same leak-free EXPECTED
earnings schedule as the headline, and runs the date-clustered Diebold-Mariano on the per-observation QLIKE
losses. Output: results/xgb/expected_schedule/earnings_dm_sp500.json (gain% + DM p per horizon),
giving the earnings significance an artifact that matches the headline QLIKE table exactly.

Run: .venv_gpu_encode/Scripts/python.exe baselines/2026-09-19_expected_schedule/code/earnings_dm.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "common" / "code"),
           str(REPO / "baselines" / "2026-09-18_leaf_graph_paper" / "code"),
           str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import full_matrix as FM  # noqa: E402
import vn_gbm_graph_stage1 as S1  # noqa: E402
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402
import leaf_graph_paper_config as C  # noqa: E402
import leaf_graph_lib as LG  # noqa: E402
import run_leaf_graph_paper as R  # noqa: E402  (reuse its OWN-8 list)
from expected_schedule import expected_schedule  # noqa: E402

OWN = R.OWN
FL = FM.FL


def main():  # pragma: no cover - data-driven driver (full S&P 500 walk-forward)
    market = "sp500"
    frames, _sect, edates = FM.load(market)
    earn = expected_schedule(edates)                       # leak-free expected schedule (headline construction)
    seeds = C.SEEDS
    full_cols = OWN + FM.EARN
    out = {}
    for h in (1, 5, 10, 22):
        a = FM.panel(frames, earn, h)
        embargo = pd.Timedelta(days=int(h * C.EMBARGO_MULT) + C.EMBARGO_BUFFER_DAYS)
        yy, dts, p_full, p_noearn = [], [], [], []
        for k in range(len(S1.FOLDS) - 1):
            ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
            trf = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
            tef = a[(a.date >= ts) & (a.date < tend)]
            if len(tef) == 0 or len(trf) < C.MIN_ROWS.get(market, C.MIN_ROWS["default"]):
                continue
            val_dates = np.sort(trf["date"].unique())[-C.VALID_LEN:]
            trf_e = trf[~trf["date"].isin(val_dates)]      # train minus val, as in the headline runner
            p_full.append(LG.predict_xgb(trf_e, tef, full_cols, seeds))
            p_noearn.append(LG.predict_xgb(trf_e, tef, OWN, seeds))
            yy.append(tef["y"].to_numpy(float))
            dts.append(tef["date"].to_numpy())
            print(f"  h{h} fold {k} done", flush=True)
        y = np.concatenate(yy)
        dates = np.concatenate(dts)
        err_full = M.per_obs_qlike(y, np.concatenate(p_full), floor=FL)
        err_noearn = M.per_obs_qlike(y, np.concatenate(p_noearn), floor=FL)
        q_full, q_noearn = float(err_full.mean()), float(err_noearn.mean())
        gain = (q_noearn - q_full) / q_noearn * 100.0
        dm = ST.date_clustered_dm(err_full, err_noearn, dates, h)   # <0 mean_diff favours XGB+E (earnings helps)
        out[f"h{h}"] = {"qlike_noearn_xgb": q_noearn, "qlike_xgbe": q_full, "earn_gain_pct": gain,
                        "dm_p": float(dm["p_value"]), "dm_mean_diff": float(dm["mean_diff"]),
                        "n": int(len(y)), "n_folds": len(yy), "n_dates": int(dm["n_dates"])}
        print(f"h{h}: earn gain {gain:+.2f}%  DM p={dm['p_value']:.3e}  "
              f"(q_noearn {q_noearn:.5f} -> q_XGB+E {q_full:.5f})", flush=True)
    outp = REPO / "results" / "xgb" / "expected_schedule" / "earnings_dm_sp500.json"
    outp.write_text(json.dumps(out, indent=2))
    print("saved", outp, flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
