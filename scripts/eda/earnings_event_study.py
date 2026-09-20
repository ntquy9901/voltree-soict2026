import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "baselines" / "common"))
import feature_panel as D

OFFS = list(range(-10, 16))

def event_curve(market):
    frames, sect, edates = D.load(market)
    if market == "hose":
        e = pd.read_parquet(REPO / "data" / "earnings" / "hose_earnings.parquet")
        edates = {tk: np.sort(g["earnings_date"].to_numpy()) for tk, g in e.groupby("ticker")}
    acc = {d: [] for d in OFFS}
    n_events = 0
    for tk, df in frames.items():
        ed = edates.get(tk)
        if ed is None or len(ed) == 0:
            continue
        pk = df["parkinson_variance"].to_numpy(float)
        dates = df["date"].to_numpy()
        base = np.nanmedian(pk)
        if not np.isfinite(base) or base <= 0:
            continue
        for d0 in ed:
            i = int(np.searchsorted(dates, d0, side="left"))
            if i >= len(dates):
                continue
            n_events += 1
            for off in OFFS:
                j = i + off
                if 0 <= j < len(pk) and np.isfinite(pk[j]):
                    acc[off].append(pk[j] / base)
    if n_events == 0:
        raise ValueError(f"no earnings events aligned for {market} (check earnings-date coverage/dtype)")
    mean = np.array([np.mean(acc[o]) if acc[o] else np.nan for o in OFFS])
    med = np.array([np.median(acc[o]) if acc[o] else np.nan for o in OFFS])
    return mean, med, n_events

def main():
    sp_mean, sp_med, sp_n = event_curve("sp500")
    ho_mean, ho_med, ho_n = event_curve("hose")
    x = np.array(OFFS)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (mean, med, n, name) in zip(axs, [(sp_mean, sp_med, sp_n, "S&P 500"), (ho_mean, ho_med, ho_n, "HOSE")]):
        ax.plot(x, mean, "-o", ms=3, color="tab:red", label="mean")
        ax.plot(x, med, "-o", ms=3, color="tab:blue", label="median")
        ax.axvline(0, color="0.3", lw=1.0)
        for xv in (-5, 10):
            ax.axvline(xv, color="green", lw=0.8, ls=":")
        ax.set_title(f"{name}: realized pk vs days from earnings (n={n:,} events)")
        ax.set_xlabel("trading days relative to announcement (0 = release)")
        ax.set_ylabel("pk / ticker-median (>1 = elevated)")
        ax.legend(fontsize=7); ax.grid(alpha=0.25)
    plt.tight_layout()
    outp = REPO / "results" / "xgb" / "earnings_event_study.png"
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"SP500 peak median {np.nanmax(sp_med):.2f} (n={sp_n:,}) | "
          f"HOSE peak median {np.nanmax(ho_med):.2f} (n={ho_n:,})", flush=True)
    print("saved", outp.relative_to(REPO), flush=True)

if __name__ == "__main__":
    main()
