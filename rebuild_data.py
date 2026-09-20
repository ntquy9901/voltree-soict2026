"""Rebuild the per-ticker enriched price CSVs the model code reads, from the compact reproduction
panels shipped in data/repro/. Run this ONCE before running any model script:

    python rebuild_data.py

It writes data/processed_enriched/sp500_clean/<ticker>.csv and data/processed_enriched/hose/<ticker>.csv.
The panels are sharded (<95 MB/file for GitHub) and carry exactly the columns the pipeline consumes
(date, parkinson_variance, the three HAR lags, and auxiliary price columns); every other feature
(log-vol momentum, earnings, leaf graph) is computed by the code, so the rebuilt CSVs reproduce the
paper's model inputs bit-for-bit (verified: OWN-8 features identical to the full enriched data).
"""
import glob
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
MARKETS = {"sp500": "sp500_clean", "hose": "hose"}

for market, dirname in MARKETS.items():
    parts = sorted(glob.glob(str(REPO / "data" / "repro" / f"{market}_panel_*.parquet")))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    out_dir = REPO / "data" / "processed_enriched" / dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    for ticker, g in df.groupby("ticker"):
        g.drop(columns=["ticker"]).to_csv(out_dir / f"{ticker}.csv", index=False)
    print(f"{market}: {len(parts)} shards -> {df['ticker'].nunique()} CSVs -> {out_dir}")
print("done. Next: python baselines/leaf_graph_paper/code/run_leaf_graph_paper.py hose")
