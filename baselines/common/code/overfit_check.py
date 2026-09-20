"""Over/under-fit confirmation from train/val/test metrics — the shared "fit verdict" logic.

Used by (a) the training path (run_masked_rich.run) to stamp a per-model ``fit_diagnostics`` block into each
result.json, and (b) the pre-push gate (check_overfit_evidence.py) to BLOCK a pushed result.json that either
lacks the train/val/test evidence or shows an over/under-fit model.

Rationale for the metric choices:
- QLIKE is scale-invariant, so the RELATIVE val->test degradation ``(test_qlike - val_qlike)/|val_qlike|`` is a
  clean overfit signal (a model that generalises has test ~ val). Volatility MSE is ~1e-7, so an ABSOLUTE 0.05
  gap (the old CLAUDE.md wording) is meaningless — we use relative gaps + R^2 instead.
- R^2 drop ``train_r2 - test_r2`` is the classic overfit gap on the explained-variance scale.
- Underfit = the model cannot even fit the TRAIN set (train_r2 below a floor): high bias, not variance.
"""
from __future__ import annotations

import math

_REQUIRED_METRIC_KEYS = ("qlike", "r2")
LEARNED = ("LSTM", "LSTM_wGAT_vol2pk")   # default learned models (masked_rich); other drivers auto-detected by name
# Name fragments that mark a LEARNED (capacity-to-overfit) model in any driver's metrics keys. Deterministic
# baselines (HAR / HAR-X / GARCH / RW / EWMA) do not match and are exempt from the fit-evidence requirement.
_LEARNED_PATTERNS = ("lstm", "volga", "gat", "gnn", "timesfm", "timesnet", "transformer", "patchtst", "mamba",
                     "xgb", "catboost", "lightgbm", "dae", "autoencoder")


def looks_learned(name: str) -> bool:
    """True if a metrics key names a learned model (any driver) that must carry over/under-fit evidence."""
    n = name.lower()
    return any(p in n for p in _LEARNED_PATTERNS)


def learned_models(res: dict) -> tuple:
    """The learned models a result must carry evidence for: those in its ``metrics`` whose name looks learned
    (works across drivers: LSTM/VolGA/VolGA_hm, LSTM/LSTM_wGAT_vol2pk, LSTM5/VolGA6, ...). Falls back to the
    masked_rich default so a partial artifact that dropped its learned block still fails rather than passes."""
    te = res.get("metrics") if isinstance(res, dict) else None
    found = tuple(m for m in te if looks_learned(m)) if isinstance(te, dict) else ()
    return found or LEARNED


def classify_fit(train: dict, val: dict, test: dict, *,
                 overfit_gap_rel: float = 0.25, overfit_r2_drop: float = 0.20,
                 underfit_r2: float = 0.0) -> dict:
    """Return a fit verdict for one model from its train/val/test metric dicts.

    ``status`` is one of ``"ok" | "overfit" | "underfit" | "unknown"``. Underfit takes precedence (a model that
    cannot fit train is high-bias regardless of the gap). Thresholds are relative/​R^2-based and configurable.
    """
    for name, m in (("train", train), ("val", val), ("test", test)):
        if not isinstance(m, dict) or any(k not in m for k in _REQUIRED_METRIC_KEYS):
            return {"status": "unknown", "reasons": [f"missing {name} metrics {_REQUIRED_METRIC_KEYS}"]}
        # F-03 (v3): a NaN/inf metric would make every threshold comparison False -> fail-OPEN ('ok'). Reject
        # non-finite evidence as 'unknown' (the gate treats non-ok as a failure) instead of silently passing.
        for k in _REQUIRED_METRIC_KEYS:
            if not math.isfinite(float(m[k])):
                return {"status": "unknown", "reasons": [f"{name} {k} is not finite ({m[k]})"]}

    vq, tq = float(val["qlike"]), float(test["qlike"])
    val_test_gap_rel = (tq - vq) / abs(vq) if vq != 0 else float("inf")
    r2_drop = float(train["r2"]) - float(test["r2"])
    reasons = []
    status = "ok"
    if float(train["r2"]) < underfit_r2 and float(test["r2"]) < underfit_r2:
        status = "underfit"
        reasons.append(f"train_r2={train['r2']:.3f} & test_r2={test['r2']:.3f} both < {underfit_r2}")
    elif val_test_gap_rel > overfit_gap_rel:
        status = "overfit"
        reasons.append(f"val->test QLIKE degraded {val_test_gap_rel:.1%} > {overfit_gap_rel:.0%}")
    elif r2_drop > overfit_r2_drop:
        status = "overfit"
        reasons.append(f"train->test R2 drop {r2_drop:.3f} > {overfit_r2_drop}")
    return {"status": status,
            "val_test_qlike_gap_rel": round(val_test_gap_rel, 4),
            "train_test_r2_drop": round(r2_drop, 4),
            "reasons": reasons}


def check_result_evidence(res: dict, *, learned=None, **thresholds) -> tuple[bool, list]:
    """Validate a result.json dict CARRIES the over/under-fit evidence and every learned model is 'ok'.

    Returns ``(passed, problems)``. A result is acceptable iff it has ``train_metrics`` + ``val_metrics`` +
    ``metrics`` (test) for each learned model AND each model's recomputed verdict is ``ok``. Deterministic
    models (HAR/HAR-X/GARCH) are exempt (no capacity to overfit in the variance sense) unless present in
    ``train_metrics``. When ``learned`` is not given it is auto-detected from the result's metrics keys
    (``learned_models``), so the check works for any driver (masked_rich, edge_hmatched, contemp, ...).
    """
    if learned is None:
        learned = learned_models(res)
    problems = []
    tr, va, te = res.get("train_metrics"), res.get("val_metrics"), res.get("metrics")
    if not isinstance(tr, dict) or not isinstance(va, dict) or not isinstance(te, dict):
        return False, ["result.json missing train_metrics / val_metrics / metrics blocks (no fit evidence)"]
    for m in learned:
        if m not in tr or m not in va or m not in te:
            problems.append(f"{m}: missing train/val/test metrics")
            continue
        v = classify_fit(tr[m], va[m], te[m], **thresholds)
        if v["status"] != "ok":
            problems.append(f"{m}: {v['status']} ({'; '.join(v['reasons'])})")
    return (len(problems) == 0), problems
