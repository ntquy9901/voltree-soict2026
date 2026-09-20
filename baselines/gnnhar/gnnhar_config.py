import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[1]
if str(REPO / "baselines" / "common") not in sys.path:
    sys.path.insert(0, str(REPO / "baselines" / "common"))
import feature_panel as D

HAR3 = list(D.HAR)                # 3 HAR lags = GNNHAR's node inputs (paper)
SEEDS = (0, 1, 2)                 # multi-seed ensemble
N_HID = 9                         # paper default hidden width
N_GCN = 2                         # GNNHAR2L: 2 graph-conv layers (paper's best variant)
VALID_LEN = 22                    # trailing validation dates for early stopping
LR = 1e-3                         # Adam learning rate
WEIGHT_DECAY = 1e-5               # Adam L2
BATCH_DATES = 256                 # dates per batched GPU forward
MAX_EPOCHS = 120                  # upper bound; early stopping normally stops sooner
MIN_EPOCHS = 20                   # do not early-stop before this many epochs
PATIENCE = 15                     # early-stopping patience on validation QLIKE
GRAD_CLIP = 1.0                   # gradient-norm clip
COLLAPSE_VAL = 1.6                # scaled val-QLIKE(+1) above this => dead-ReLU -> restart
MAX_RESTARTS = 4                  # restart-with-new-seed attempts on collapse
HORIZONS = (1, 5, 10, 22)         # forecast horizons
MIN_TRAIN_ROWS = {"sp500": 30000, "default": 3000}   # min pooled train rows per fold
SMOKE_EPOCHS = 40                 # --smoke: 1 fold, 1 seed, few epochs
