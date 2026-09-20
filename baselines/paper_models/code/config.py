"""Constants for the principled-HAR feature baseline (option A: fixed Corsi windows).

See spec docs/superpowers/specs/2026-09-13-principled-har-features-design.md. No data-driven lag selection:
the HAR windows are the standard Corsi (2009) 1/5/22 (already precomputed as har_daily/weekly/monthly).
"""
HAR_WINDOWS = (1, 5, 22)     # Corsi (2009) daily/weekly/monthly (informational; features are precomputed cols)
SEMI_WINDOW = 5              # trailing window (days) for realized semivariance (weekly scale, matches har_weekly)
SEMI_MIN_PERIODS = 3         # min observations before a trailing semivariance value is defined
HORIZONS = (1, 5, 10, 22)   # forecast horizons
MIN_ROWS = {"sp500": 30000, "default": 3000}   # min causal train rows per fold (mirrors run_gbm's gate)
RANGE_SHORT = 5             # short window for the range-compression (squeeze) feature (weekly, matches Corsi 5)
RANGE_LONG = 22             # long window for range compression / expansion (monthly, matches Corsi 22)

# --- SINGLE SOURCE OF TRUTH for the paper's own-history baseline (relative to the shared FM.OWN) ---
# To drop or add a feature later, edit ONLY these two tuples; all runners derive their baseline from own_set().
OWN_DROP = ("rq",)          # features removed from FM.OWN (rq proxy dropped 2026-09-13: ~0.1% effect, mislabel)
OWN_ADD = ()                # extra cited features appended to the baseline (e.g. add "semi_neg" once confirmed)


def own_set(fm_own):
    """The paper own-history feature list = FM.OWN minus OWN_DROP plus OWN_ADD."""
    return [f for f in fm_own if f not in OWN_DROP] + list(OWN_ADD)
