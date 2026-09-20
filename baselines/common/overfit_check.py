from __future__ import annotations

import math

_REQUIRED_METRIC_KEYS = ("qlike", "r2")
LEARNED = ("LSTM", "LSTM_wGAT_vol2pk")

_LEARNED_PATTERNS = ("lstm", "volga", "gat", "gnn", "timesfm", "timesnet", "transformer", "patchtst", "mamba",
                     "xgb", "catboost", "lightgbm", "dae", "autoencoder")

def looks_learned(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in _LEARNED_PATTERNS)

def learned_models(res: dict) -> tuple:
    te = res.get("metrics") if isinstance(res, dict) else None
    found = tuple(m for m in te if looks_learned(m)) if isinstance(te, dict) else ()
    return found or LEARNED

def classify_fit(train: dict, val: dict, test: dict, *,
                 overfit_gap_rel: float = 0.25, overfit_r2_drop: float = 0.20,
                 underfit_r2: float = 0.0) -> dict:
    for name, m in (("train", train), ("val", val), ("test", test)):
        if not isinstance(m, dict) or any(k not in m for k in _REQUIRED_METRIC_KEYS):
            return {"status": "unknown", "reasons": [f"missing {name} metrics {_REQUIRED_METRIC_KEYS}"]}

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
