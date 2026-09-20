from __future__ import annotations

from dataclasses import dataclass

import arch
import numpy as np

import config

@dataclass(frozen=True)
class Params:
    mu: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    ok: bool
    fallback_var: float

def reversion_persistence(p: Params, variant: str) -> float:
    phi = p.alpha + p.beta
    if variant == "gjr":
        phi += p.gamma / 2.0
    return phi

def multistep(s_next_scaled: np.ndarray, p: Params, variant: str, h: int) -> np.ndarray:
    phi = reversion_persistence(p, variant)
    uncond = p.omega / (1.0 - phi)
    s_next_scaled = np.asarray(s_next_scaled, dtype=float)
    return uncond + phi ** (h - 1) * (s_next_scaled - uncond)

def one_step_next(returns_scaled: np.ndarray, p: Params, variant: str) -> np.ndarray:
    eps = np.asarray(returns_scaled, dtype=float) - p.mu
    n = eps.shape[0]
    neg = (eps < 0.0).astype(float)
    e2 = eps ** 2
    sig = np.empty(n, dtype=float)
    sig[0] = p.omega / (1.0 - reversion_persistence(p, variant))
    lev = p.gamma if variant == "gjr" else 0.0
    for i in range(1, n):
        sig[i] = p.omega + (p.alpha + lev * neg[i - 1]) * e2[i - 1] + p.beta * sig[i - 1]
    return p.omega + (p.alpha + lev * neg) * e2 + p.beta * sig

def fit_params(train_returns: np.ndarray, variant: str) -> Params:
    train_returns = np.asarray(train_returns, dtype=float)
    fallback_var = float(np.var(train_returns, ddof=1)) if train_returns.size >= 2 else 0.0
    bad = Params(0.0, 0.0, 0.0, 0.0, 0.0, ok=False, fallback_var=fallback_var)
    if train_returns.size < config.MIN_TRAIN_OBS:
        return bad
    y = train_returns * config.SCALE
    kw = dict(mean=config.MEAN, vol="GARCH", p=1, q=1, dist=config.DIST)
    if variant == "gjr":
        kw["o"] = 1
    try:
        res = arch.arch_model(y, **kw).fit(disp="off", show_warning=False)
        pr = res.params
        p = Params(mu=float(pr.get("mu", 0.0)), omega=float(pr["omega"]), alpha=float(pr["alpha[1]"]),
                   gamma=float(pr.get("gamma[1]", 0.0)) if variant == "gjr" else 0.0,
                   beta=float(pr["beta[1]"]), ok=True, fallback_var=fallback_var)
    except Exception:
        return bad
    phi = reversion_persistence(p, variant)
    if not (p.omega > 0.0 and config.PERSIST_LO < phi < config.PERSIST_HI):
        return bad
    uncond_orig = (p.omega / (1.0 - phi)) / (config.SCALE ** 2)
    if not (fallback_var > 0.0
            and fallback_var / config.VAR_RATIO_CAP <= uncond_orig <= fallback_var * config.VAR_RATIO_CAP):
        return bad
    return p

def forecast(returns: np.ndarray, p: Params, variant: str, test_idx: np.ndarray, h: int) -> np.ndarray:
    test_idx = np.asarray(test_idx, dtype=int)
    if not p.ok:
        return np.full(test_idx.shape[0], p.fallback_var, dtype=float)
    s_next = one_step_next(np.asarray(returns, dtype=float) * config.SCALE, p, variant)
    f = multistep(s_next[test_idx], p, variant, h) / (config.SCALE ** 2)
    return np.clip(f, p.fallback_var / config.VAR_RATIO_CAP, p.fallback_var * config.VAR_RATIO_CAP)
