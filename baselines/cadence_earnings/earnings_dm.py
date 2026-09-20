import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "common"),
           str(REPO / "baselines" / "leaf_graph"),
           str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import feature_panel as D
import metrics as M
import stats as ST
import leaf_graph_config as C
import leaf_graph_lib as LG
import run_leaf_graph as R
from expected_schedule import expected_schedule

OWN = R.OWN
FL = D.FL

def main():
    market = "sp500"
    frames, _sect, edates = D.load(market)
    earn = expected_schedule(edates)
    seeds = C.SEEDS
    full_cols = OWN + D.EARN
    out = {}
    for h in (1, 5, 10, 22):
        a = D.panel(frames, earn, h)
        embargo = pd.Timedelta(days=int(h * C.EMBARGO_MULT) + C.EMBARGO_BUFFER_DAYS)
        yy, dts, p_full, p_noearn = [], [], [], []
        for k in range(len(D.FOLDS) - 1):
            ts, tend = pd.Timestamp(D.FOLDS[k]), pd.Timestamp(D.FOLDS[k + 1])
            trf = a[(a.date >= D.TRAIN_START) & (a.date < ts - embargo)]
            tef = a[(a.date >= ts) & (a.date < tend)]
            if len(tef) == 0 or len(trf) < C.MIN_ROWS.get(market, C.MIN_ROWS["default"]):
                continue
            val_dates = np.sort(trf["date"].unique())[-C.VALID_LEN:]
            trf_e = trf[~trf["date"].isin(val_dates)]
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
        dm = ST.date_clustered_dm(err_full, err_noearn, dates, h)
        out[f"h{h}"] = {"qlike_noearn_xgb": q_noearn, "qlike_xgbe": q_full, "earn_gain_pct": gain,
                        "dm_p": float(dm["p_value"]), "dm_mean_diff": float(dm["mean_diff"]),
                        "n": int(len(y)), "n_folds": len(yy), "n_dates": int(dm["n_dates"])}
        print(f"h{h}: earn gain {gain:+.2f}%  DM p={dm['p_value']:.3e}  "
              f"(q_noearn {q_noearn:.5f} -> q_XGB+E {q_full:.5f})", flush=True)
    outp = REPO / "results" / "xgb" / "cadence_earnings" / "earnings_dm_sp500.json"
    outp.write_text(json.dumps(out, indent=2))
    print("saved", outp, flush=True)

if __name__ == "__main__":
    main()
