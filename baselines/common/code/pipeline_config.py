from __future__ import annotations

LOOKBACK: int = 10
EPOCHS: int = 20
PATIENCE: int = 3
MIN_EPOCHS: int = 5
HIDDEN: int = 64
HEADS: int = 4
DROPOUT: float = 0.2
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-5
GRAD_CLIP: float = 1.0
BATCH_SIZE: int = 512

SEEDS: tuple = (42, 123, 2026, 7, 2024)
HORIZONS: tuple = (1, 5, 10, 22)

FIRST_VALID: int = 21
HAR_WEEKLY_WINDOW: int = 5
HAR_MONTHLY_WINDOW: int = 22

VOLUME_ZSCORE_WINDOW: int = 22
VOL_OF_VOL_WINDOW: int = 22

TRAIN_FRAC: float = 0.80
VAL_FRAC: float = 0.10
MIN_ROWS: int = 200
MIN_ANCHORS: int = 60
MIN_TRAIN: int = 30
MIN_VAL: int = 5
MIN_TEST: int = 5
MIN_VALID_NODES: int = 8
MIN_TRAIN_ROWS: int = 252
MIN_COMMON_DATES: int = 300

N_NODE_FEATURES: int = 5
EDGE_TOP_K: int = 5
EDGE_MIN_OVERLAP: int = 100
EDGE_MIN_PAIRS_DIRECTED: int = 30
MIN_VOL_COVERAGE: float = 0.5
EMPTY_VOL_COVERAGE: float = 0.05

QLIKE_FLOOR: float = 1e-8
PRED_FLOOR_FRAC: float = 1e-3
POS_FLOOR_FRAC: float = 1e-2
POS_FLOOR_EPS: float = 1e-12
SCALER_EPS: float = 1e-8
RESIDUAL_EPS: float = 1e-8
CROSSFIT_FOLDS: int = 5

WF_RETRAIN_K: int = 66
WF_VAL_TAIL: int = 66
WF_TEST_FRAC: float = 0.90
WF_HORIZON: int = 1
