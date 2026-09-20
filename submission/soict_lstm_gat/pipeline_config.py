"""Single canonical configuration — the ONE source of truth for every tunable pipeline constant.

Per CLAUDE.md "Single-source-of-truth app config (ENFORCED)": all tunable constants
(hyperparameters, windows, thresholds/floors, edge params, horizons) live HERE and are imported by
the pipeline modules. Editing a value = editing exactly one place; no module drifts. Root cause this
fixes: ``volume_zscore`` used a hardcoded window=20 in ``masked_rich`` while the project monthly
convention is 22 (``har_monthly``) and ``screen_features`` already used 22 — a silent drift.

Consumers: ``config.Config`` / ``run_walkforward.WFConfig`` source their field defaults from here;
``masked_rich`` / ``masked_snapshots`` / ``data_utils`` / ``experts`` / ``run_masked_rich`` /
``run_experiment`` / ``screen_features`` import the named constants. This module lives beside
``config.py`` (already on every consumer's ``sys.path``) and has no external dependencies.
"""
from __future__ import annotations

# ============================ TRAINING HYPERPARAMETERS ============================
LOOKBACK: int = 10                 # main sequence lookback (SEQ); 22 = variation
EPOCHS: int = 20                   # max training epochs
PATIENCE: int = 3                  # early-stop patience on val MSE
MIN_EPOCHS: int = 5                # minimum epochs before early stop may fire
HIDDEN: int = 64                   # LSTM / GAT hidden size
HEADS: int = 4                     # GAT attention heads
DROPOUT: float = 0.2               # dropout rate
LR: float = 1e-3                   # Adam learning rate
WEIGHT_DECAY: float = 1e-5         # Adam L2 weight decay
GRAD_CLIP: float = 1.0             # gradient-norm clip
BATCH_SIZE: int = 512              # training batch size

# ============================ SEEDS + HORIZONS ============================
SEEDS: tuple = (42, 123, 2026, 7, 2024)   # 5 seeds for multi-seed ensembling
HORIZONS: tuple = (1, 5, 10, 22)          # canonical forecast horizons (CLI may override)

# ============================ DATA / FEATURE WINDOWS ============================
FIRST_VALID: int = 21              # monthly HAR (22-obs rolling) is first valid at index 21
HAR_WEEKLY_WINDOW: int = 5         # weekly HAR rolling window
HAR_MONTHLY_WINDOW: int = 22       # monthly HAR rolling window (the "monthly = 22 trading days" convention)
# volume_zscore trailing rolling window. HISTORICAL NOTE: the delivered/paper result JSONs
# (results/masked_rich_floor1e2/...) were computed with 20; the committed canonical value is now 22 to
# match the project monthly convention (HAR_MONTHLY_WINDOW=22, screen_features vol shock). This is a
# config KNOB — reproducing the delivered results requires temporarily setting it back to 20. The
# 2-day widening changes the feature slightly, so delivered result JSONs are not reproduced at 22.
VOLUME_ZSCORE_WINDOW: int = 22
VOL_OF_VOL_WINDOW: int = 22        # rolling std of pk (vol-of-vol); distinct concept from the volume window

# ============================ SPLIT / DROP THRESHOLDS ============================
TRAIN_FRAC: float = 0.80           # chronological train fraction
VAL_FRAC: float = 0.10             # chronological val fraction (test = 1 - train - val)
MIN_ROWS: int = 200                # drop a ticker with fewer than this many rows
MIN_ANCHORS: int = 60              # drop a ticker with fewer than this many valid anchors
MIN_TRAIN: int = 30                # drop if TRAIN split smaller than this
MIN_VAL: int = 5                   # drop if VAL split smaller than this
MIN_TEST: int = 5                  # drop if TEST split smaller than this
MIN_VALID_NODES: int = 8           # min valid nodes for a masked-panel anchor to be kept
MIN_TRAIN_ROWS: int = 252          # drop a node with fewer valid TRAIN targets (degenerate scaler)
MIN_COMMON_DATES: int = 300        # min common dates for the common-date snapshot node set

# ============================ GRAPH / EDGE PARAMS ============================
N_NODE_FEATURES: int = 5           # rich node feature count [pk, har_w, har_m, market_pk, volume_zscore]
EDGE_TOP_K: int = 5                # Top-K graph neighbours per node (correlation / vol->PK / glasso / vshock)
EDGE_MIN_OVERLAP: int = 100        # symmetric correlation edge: min overlapping days for corr()
EDGE_MIN_PAIRS_DIRECTED: int = 30  # directed volume->PK edge: min overlapping (t, t+1) pairs
MIN_VOL_COVERAGE: float = 0.5      # warn if a present ohlcv file covers < this fraction of a ticker's dates
EMPTY_VOL_COVERAGE: float = 0.05   # <= this fraction == present-but-empty == semantically missing (fail-loud cap)

# ============================ FLOORS / EPSILONS ============================
QLIKE_FLOOR: float = 1e-8          # metric floor; identical across all compared models
PRED_FLOOR_FRAC: float = 1e-3      # additive-residual reconstruction floor: PRED_FLOOR_FRAC * train mean
POS_FLOOR_FRAC: float = 1e-2       # shared per-node positivity floor: POS_FLOOR_FRAC * train mean + POS_FLOOR_EPS
POS_FLOOR_EPS: float = 1e-12       # positivity-floor additive epsilon
SCALER_EPS: float = 1e-8           # per-node scaler std floor (avoid divide-by-zero); distinct role from QLIKE_FLOOR
RESIDUAL_EPS: float = 1e-8         # log-residual epsilon in the multiplicative expert
CROSSFIT_FOLDS: int = 5            # expanding-window folds for cross-fitted OOS HAR residual targets

# ============================ WALK-FORWARD ============================
WF_RETRAIN_K: int = 66             # expanding-window retrain cadence (days per fold)
WF_VAL_TAIL: int = 66              # validation tail length per fold
WF_TEST_FRAC: float = 0.90         # anchor fraction where the OOS walk-forward region begins
WF_HORIZON: int = 1                # walk-forward default horizon
