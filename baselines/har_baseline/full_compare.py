import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[1]
for _p in (str(REPO / "baselines" / "common"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config
import feature_panel as D

def _checkpoint(out, out_path):
    if out_path is None:
        return
    tmp = Path(str(out_path) + ".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(out_path)
    print(f"[checkpoint] wrote {out_path.name} ({len(out)} horizons)", flush=True)

def run(market, load_fn=None, out_path=None):
    load_fn = load_fn or D.load
    min_rows = config.MIN_ROWS.get(market, config.MIN_ROWS["default"])
    frames, _sect, edates = load_fn(market)
    out = {}
    for h in config.HORIZONS:
        a = D.panel(frames, edates, h)
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        preds, yy = [], []
        for k in range(len(D.FOLDS) - 1):
            ts, tend = pd.Timestamp(D.FOLDS[k]), pd.Timestamp(D.FOLDS[k + 1])
            trf = a[(a.date >= D.TRAIN_START) & (a.date < ts - embargo)]
            tef = a[(a.date >= ts) & (a.date < tend)]
            if len(tef) == 0 or len(trf) < min_rows:
                continue
            preds.append(D._har_ols(trf, tef))
            yy.append(tef["y"].to_numpy(float))
        if not yy:
            continue
        y = np.concatenate(yy)
        out[f"h{h}"] = {"n": int(len(y)), "metrics": {"HAR": D.all_metrics(y, np.concatenate(preds))}}
        _checkpoint(out, out_path)
    return out

def _print(market, out):
    for h, r in out.items():
        mm = r["metrics"]["HAR"]
        print(f"  {market} {h} (n={r['n']:,})  HAR QLIKE {mm['qlike']:.4f} R2 {mm['r2']:+.3f}", flush=True)

def main():
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    outp = REPO / "results" / "xgb" / f"full_compare_{market}.json"
    out = run(market, out_path=outp)
    _print(market, out)
    print(f"\nsaved {outp.relative_to(REPO)}", flush=True)

if __name__ == "__main__":
    main()
