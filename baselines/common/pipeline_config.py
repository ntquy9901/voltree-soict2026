from __future__ import annotations

# QLIKE positivity floor shared by the HAR benchmark and the XGBoost model:
# predictions and realized variance are clipped to this value before scoring.
QLIKE_FLOOR: float = 1e-8
