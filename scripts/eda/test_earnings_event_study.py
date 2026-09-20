"""Fast smoke test for the earnings event-study curve (Fig 2), with a monkeypatched loader + earnings.

Locks the normalization identity (flat series -> ratio 1.0), the offset-0 alignment (a spike on the
release day lands at offset 0), and the fail-loud guard when no event aligns. No real market data.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # self-contained import (no reliance on pytest prepend)
import earnings_event_study as E  # noqa: E402


def _frames(pk):
    dates = pd.bdate_range("2023-01-02", periods=len(pk))
    df = pd.DataFrame({"date": dates.to_numpy(), "parkinson_variance": np.asarray(pk, float)})
    return {"AAA": df}, {}, {}


def _patch(monkeypatch, pk, earnings):
    monkeypatch.setattr(E.FM, "load", lambda market: _frames(pk))
    monkeypatch.setattr(E.pd, "read_parquet",
                        lambda *a, **k: pd.DataFrame({"ticker": ["AAA"], "earnings_date": earnings}))


def test_flat_series_normalizes_to_one(monkeypatch):
    pk = np.full(200, 3e-4)
    ann = pd.bdate_range("2023-01-02", periods=200)[100].to_datetime64()
    _patch(monkeypatch, pk, [ann])
    mean, med, n = E.event_curve("hose")
    assert len(mean) == len(med) == len(E.OFFS) == 26
    assert n == 1
    assert np.allclose(med, 1.0)                                         # baseline = ticker median


def test_spike_lands_on_offset_zero(monkeypatch):
    pk = np.full(200, 3e-4)
    pk[100] = 6e-4                                                       # exactly 2x median on the release day
    ann = pd.bdate_range("2023-01-02", periods=200)[100].to_datetime64()
    _patch(monkeypatch, pk, [ann])
    _mean, med, _n = E.event_curve("hose")
    assert med[E.OFFS.index(0)] == pytest.approx(2.0)                    # searchsorted-left lands offset 0 on release


def test_no_aligned_event_fails_loud(monkeypatch):
    pk = np.full(50, 3e-4)
    _patch(monkeypatch, pk, [np.datetime64("2099-01-01")])              # past the last date -> dropped
    with pytest.raises(ValueError):
        E.event_curve("hose")


def test_offsets_out_of_range_are_skipped(monkeypatch):
    pk = np.full(40, 3e-4)
    ann = pd.bdate_range("2023-01-02", periods=40)[2].to_datetime64()   # near start: offset -10 lands before index 0
    _patch(monkeypatch, pk, [ann])
    _mean, med, n = E.event_curve("hose")
    assert n == 1
    assert np.isnan(med[0])                                             # offset -10 -> j<0, no sample -> nan


def test_skips_ticker_without_earnings_and_invalid_base(monkeypatch):
    dates = pd.bdate_range("2023-01-02", periods=60)
    ann = dates[30].to_datetime64()
    frames = {
        "AAA": pd.DataFrame({"date": dates.to_numpy(), "parkinson_variance": np.full(60, 3e-4)}),
        "NOE": pd.DataFrame({"date": dates.to_numpy(), "parkinson_variance": np.full(60, 3e-4)}),  # no earnings
        "BAD": pd.DataFrame({"date": dates.to_numpy(), "parkinson_variance": np.zeros(60)}),        # base<=0
    }
    monkeypatch.setattr(E.FM, "load", lambda market: (frames, {}, {}))
    monkeypatch.setattr(E.pd, "read_parquet",
                        lambda *a, **k: pd.DataFrame({"ticker": ["AAA", "BAD"], "earnings_date": [ann, ann]}))
    _mean, med, n = E.event_curve("hose")
    assert n == 1                                                       # NOE skipped (no earnings), BAD skipped (base<=0)
    assert med[E.OFFS.index(0)] == pytest.approx(1.0)
