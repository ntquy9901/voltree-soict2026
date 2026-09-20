"""Import/contract smoke + numerical regression for the earnings-DM audit script.

Importing the module covers its bootstrap + OWN-8 wiring; the heavy walk-forward lives in main().
The numerical test guards the committed audit artifact so the paper's earnings-DM headline
(gain 3.24/3.13/3.00/2.68%, p<1e-3 all horizons) cannot silently drift from the JSON it cites."""

import json
from pathlib import Path

_ARTIFACT = (Path(__file__).resolve().parents[3]
             / "results" / "xgb" / "expected_schedule" / "earnings_dm_sp500.json")

# Paper-cited headline (docs/paper/soict_2026-09-19_*.tex, Section 5.1): earnings gain % per horizon.
_HEADLINE_GAIN = {"h1": 3.24, "h5": 3.13, "h10": 3.00, "h22": 2.68}


def test_earnings_dm_module_contract():
    import earnings_dm as E
    assert isinstance(E.OWN, list) and len(E.OWN) == 8 and "rq" not in E.OWN
    assert E.FL > 0
    assert hasattr(E, "main")


def test_earnings_dm_artifact_matches_paper_headline():
    assert _ARTIFACT.exists(), f"missing earnings-DM audit artifact: {_ARTIFACT}"
    d = json.loads(_ARTIFACT.read_text())
    assert set(d) == set(_HEADLINE_GAIN), "artifact horizons must be exactly h1/h5/h10/h22"
    for h, exp_gain in _HEADLINE_GAIN.items():
        row = d[h]
        assert row["n_folds"] == 8, f"{h}: audit must span all 8 walk-forward folds"
        # earnings must lower QLIKE (XGB+E below XGB-noearn) and the DM diff sign must agree
        assert row["qlike_xgbe"] < row["qlike_noearn_xgb"], f"{h}: earnings did not lower QLIKE"
        assert row["dm_mean_diff"] < 0, f"{h}: DM mean loss differential must favour XGB+E"
        # gain % recomputed from the two QLIKE values must match the stored field and the paper
        recomputed = 100.0 * (row["qlike_noearn_xgb"] - row["qlike_xgbe"]) / row["qlike_noearn_xgb"]
        assert abs(recomputed - row["earn_gain_pct"]) < 1e-6, f"{h}: stored gain != recomputed"
        assert abs(row["earn_gain_pct"] - exp_gain) < 0.05, f"{h}: gain drifted from paper headline"
        # significance: paper claims p<0.001 for the emphasized earnings comparison
        assert 0.0 < row["dm_p"] < 1e-3, f"{h}: DM p-value must stay below 1e-3"
