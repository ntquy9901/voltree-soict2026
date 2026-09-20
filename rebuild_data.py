import glob
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
MARKETS = {"sp500": "sp500_clean", "hose": "hose"}

for market, dirname in MARKETS.items():
    parts = sorted(glob.glob(str(REPO / "data" / "panels" / f"{market}_panel_*.parquet")))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    out_dir = REPO / "data" / "processed_enriched" / dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    for ticker, g in df.groupby("ticker"):
        g.drop(columns=["ticker"]).to_csv(out_dir / f"{ticker}.csv", index=False)
    print(f"{market}: {len(parts)} shards -> {df['ticker'].nunique()} CSVs -> {out_dir}")
print("done. Next: python baselines/leaf_graph/run_leaf_graph.py hose --featureset noearn")
