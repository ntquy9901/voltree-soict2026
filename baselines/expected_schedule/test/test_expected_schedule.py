"""Unit + real-data tests for the expected-schedule earnings dates + HOSE quarterly reduction."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from expected_schedule import (
    MIN_HISTORY, expected_schedule, expected_vs_actual, hose_quarterly_dates, predict_schedule, summarize,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eda"))


def _quarterly(n, start="2018-01-31", step=91):
    d0 = np.datetime64(start)
    return np.array([d0 + np.timedelta64(step * i, "D") for i in range(n)], dtype="datetime64[D]")


def test_short_series_exposes_nothing_leaksafe():
    # <= MIN_HISTORY releases -> no causally-predicted date exists, so NOTHING is exposed to the model
    # (the actual anchors must not be emitted, or they would leak a future realized date at an earlier origin).
    dates = _quarterly(MIN_HISTORY)
    out = expected_schedule({"AAA": dates})
    assert out["AAA"].dtype == np.dtype("datetime64[ns]")
    assert len(out["AAA"]) == 0


def test_expected_schedule_drops_actual_anchors():
    # 5 quarterly dates -> only the 2 causally-predicted dates (index >= MIN_HISTORY) are exposed;
    # the first MIN_HISTORY ACTUAL dates never appear in the exposed schedule.
    dates = _quarterly(5, step=91)
    out = expected_schedule({"AAA": dates})["AAA"].astype("datetime64[D]")
    assert len(out) == 5 - MIN_HISTORY
    anchors = np.sort(dates.astype("datetime64[D]"))[:MIN_HISTORY]
    assert not np.isin(anchors, out).any()                       # no actual anchor leaks into the schedule


def test_regular_cadence_predicts_actual_exactly():
    dates = _quarterly(8, step=91)
    pred = predict_schedule(dates)
    assert np.array_equal(pred[:MIN_HISTORY], dates[:MIN_HISTORY])
    assert np.array_equal(pred[MIN_HISTORY:], dates[MIN_HISTORY:])


def test_prediction_is_causal_prior_gaps_only():
    # gaps between the first releases are [30, 90, 90]. For event i=MIN_HISTORY the CAUSAL median uses only
    # gaps[:i-1]=[30,90] -> 60 days; a LEAKY median(gaps[:i])=[30,90,90] -> 90 days would require the gap TO
    # event i (the future date d[i]). This case DISTINGUISHES the two (regular-cadence cases cannot).
    base = np.datetime64("2020-01-01")
    d = np.array([base, base + np.timedelta64(30, "D"), base + np.timedelta64(120, "D"),
                  base + np.timedelta64(210, "D")], dtype="datetime64[D]")   # gaps 30, 90, 90
    pred = predict_schedule(d)
    causal = d[MIN_HISTORY - 1] + np.timedelta64(60, "D")    # d[2] + median([30,90]) = +60 (strictly prior)
    leaky = d[MIN_HISTORY - 1] + np.timedelta64(90, "D")     # d[2] + median([30,90,90]) = +90 (leaks d[3])
    assert pred[MIN_HISTORY] == causal
    assert pred[MIN_HISTORY] != leaky                        # guards against the off-by-one look-ahead leak


def test_expected_vs_actual_and_summarize():
    disc = expected_vs_actual({"AAA": _quarterly(6, step=91), "SHORT": _quarterly(2)})
    assert list(disc.columns) == ["ticker", "event_index", "actual_date", "predicted_date",
                                  "abs_err_days", "signed_err_days"]
    assert set(disc["ticker"]) == {"AAA"} and (disc["abs_err_days"] == 0).all()
    s = summarize(disc)
    assert s["n_events"] == 6 - MIN_HISTORY and s["within_1d_pct"] == 100.0
    assert summarize(expected_vs_actual({"S": _quarterly(2)})) == {"n_events": 0, "n_tickers": 0}


def test_hose_quarterly_earliest_across_scope(tmp_path):
    rows = []
    for scope, day in (("hop-nhat", "2025-04-24"), ("rieng", "2025-04-20")):   # same quarter, two scopes
        rows.append({"ticker": "AAA", "fiscal_year": 2025, "quarter": "Q1", "scope": scope,
                     "announcement_date": day})
    rows.append({"ticker": "AAA", "fiscal_year": 2025, "quarter": "FY", "scope": "hop-nhat",
                 "announcement_date": "2025-03-30"})                            # annual -> dropped
    csv = tmp_path / "disc.csv"; pd.DataFrame(rows).to_csv(csv, index=False)
    q = hose_quarterly_dates(csv)
    assert list(q) == ["AAA"]
    assert len(q["AAA"]) == 1                                                   # one quarterly event
    assert q["AAA"][0].astype("datetime64[D]") == np.datetime64("2025-04-20")   # earliest across scopes; FY dropped


def test_expected_load_earn_dispatch(monkeypatch):
    import run_expected
    monkeypatch.setattr(run_expected, "expected_schedule", lambda ed: ("ES", ed))
    monkeypatch.setattr(run_expected, "hose_quarterly_dates", lambda: {"HQ": [1]})
    # SP500 forwards the given (yfinance) edates; HOSE ignores them and builds the quarterly schedule
    assert run_expected.expected_load_earn("sp500", {"A": [1]}) == ("ES", {"A": [1]})
    assert run_expected.expected_load_earn("hose", {"ignored": 1}) == ("ES", {"HQ": [1]})


def _min_frame(dates):
    """One ticker's featurised frame with constant OWN columns (enough for full_matrix.panel to run)."""
    import full_matrix as FM
    fr = pd.DataFrame({"date": pd.DatetimeIndex(dates)})
    for c in FM.OWN:
        fr[c] = 1.0
    fr["parkinson_variance"] = 1e-4
    return fr


def test_panel_no_future_actual_date_leaks_before_history():
    # INTEGRATION-level guard (the missing test the review asked for): a forecast target that lands BEFORE
    # a ticker has MIN_HISTORY observed releases must see NO upcoming-earnings signal. The leak-safe schedule
    # exposes only causally-predicted dates, so the first actual release (an anchor) is never used as the
    # "next earnings" for an earlier origin.
    import full_matrix as FM
    bdays = pd.bdate_range("2025-01-01", "2025-12-31")
    frames = {"AAA": _min_frame(bdays)}
    actual = {"AAA": np.array(["2025-01-24", "2025-04-29", "2025-07-30", "2025-10-30", "2026-01-28"],
                              dtype="datetime64[D]")}
    anchor = np.datetime64("2025-01-24")                                # first ACTUAL release (an anchor)
    sched = expected_schedule(actual)
    exposed = sched["AAA"].astype("datetime64[D]")
    assert anchor not in exposed                                        # anchor never exposed
    assert exposed.min() > anchor + np.timedelta64(30, "D")             # earliest exposed date is far later

    # rows in the anchor's month: their forecast target sits at/near the actual anchor date
    def near_anchor(p):
        dd = p["date"].to_numpy("datetime64[D]")
        return p[(dd >= np.datetime64("2025-01-01")) & (dd <= anchor)]

    p_safe = near_anchor(FM.panel(frames, sched, h=1))
    assert len(p_safe) > 0
    assert (p_safe["earn_prox"] == 0).all() and (p_safe["earn_soon"] == 0).all() and (p_safe["earn_pre"] == 0).all()

    # Contrast: feeding the RAW actual dates (the leaky construction) lights up earn_prox at those same rows
    # -- proving this test distinguishes the fix from the leak.
    raw = {"AAA": np.sort(actual["AAA"]).astype("datetime64[ns]")}
    p_leak = near_anchor(FM.panel(frames, raw, h=1))
    assert (p_leak["earn_prox"] > 0).any()


@pytest.mark.parametrize("_", [0])
def test_real_hose_disclosures_reduce_to_quarterly(_):
    q = hose_quarterly_dates()                                                  # real SSC archive
    assert len(q) > 100
    gaps = []
    for d in q.values():
        d = np.sort(d)
        if len(d) >= 2:
            gaps.extend(np.diff(d).astype("timedelta64[D]").astype(int).tolist())
    g = np.asarray(gaps, float)
    assert float(np.mean((g >= 60) & (g <= 130)) * 100) > 75.0                  # quarterly-regular after reduction
