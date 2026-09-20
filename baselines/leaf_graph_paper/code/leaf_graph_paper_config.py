"""Single source of truth for the FINAL paper-grade (3-seed + earnings-ablation) leaf-cooccurrence-graph baseline.

Every tunable constant lives here (not scattered as magic numbers in the pipeline) per CLAUDE.md
"Single-source-of-truth app config". Inline comments are intentional: they document intent and keep the
config-hardcode scanner's bare ``NAME = <num>$`` rule from firing on legitimate config constants.

Model set is deliberately {XGB base, XGB+leafgraph} ONLY — the champion HGBR ``GBME`` is NOT fit or reported here
(paper scope). Constants mirror the committed proof-of-concept
``(prior)/leaf_graph_config.py`` (same method, same champion-matched XGBoost gamma
capacity). Two run arms share this one config via a driver ``--featureset`` switch (full = OWN-8 + EARN-4 vs
noearn = OWN-8), so the feature set is NOT a constant here.
"""
# --- seed ensemble (paper knob: the 3 canonical seeds) ---
N_SEEDS = 3                        # number of seeds averaged for the XGB base predictions (paper-grade)
SEEDS = (0, 1, 2)                  # the 3 seeds (len == N_SEEDS); XGB base averaged over these

# --- XGBoost gamma booster (capacity mirrors the champion HGBR: 300 trees, lr 0.05, ~31 leaves, l2=1) ---
XGB_N_ESTIMATORS = 300      # boosting rounds = number of trees = leaf-vector length (match champion max_iter)
XGB_LR = 0.05               # learning rate (match champion learning_rate)
XGB_MAX_LEAVES = 31         # leaves per tree under lossguide (match champion max_leaf_nodes)
XGB_MAX_DEPTH = 0           # 0 = depth unbounded; leaf count is the capacity control under grow_policy=lossguide
XGB_L2 = 1.0                # L2 regularisation (match champion l2_regularization)
XGB_MIN_CHILD_WEIGHT = 20   # min hessian per leaf (analog of the champion HGBR min_samples_leaf=20; regularises)
PRED_CAP = 1.0              # variance ceiling (sigma=100%/day) — numerical guard so an XGB gamma exp-link on an
# extreme row cannot overflow to +inf; far above any real daily variance

# --- leaf-cooccurrence graph + prediction smoothing ---
K_NEIGHBOURS = 10           # per-day kNN degree over leaf-Hamming similarity (matched to the S1 graph TOPK=10)
ALPHA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
              0.8, 0.9, 1.0)   # smoothing weight grid; alpha fit on the val slice per fold (0 => graph inert)

# --- walk-forward / evaluation (mirror the sibling GBME battery) ---
EMBARGO_MULT = 1.6                              # embargo = int(h*EMBARGO_MULT)+EMBARGO_BUFFER_DAYS calendar days
EMBARGO_BUFFER_DAYS = 5                         # fixed buffer added to the horizon-scaled embargo (purge overlap)
VALID_LEN = 22                                  # trailing train dates held out as a true val slice (fit alpha)
HORIZONS = (1, 5, 10, 22)                       # forecast horizons (match the comparison table)
HORIZONS_SMOKE = (1,)                           # smoke: single horizon
MIN_ROWS = {"sp500": 30000, "default": 3000}    # min causal train rows per fold (mirrors the sibling gate)

# --- pre-registered kill criterion (isolate the graph: XGB+leafgraph vs XGB) ---
GAIN_MIN = 0.0             # XGB+leafgraph must beat XGB by strictly more than this QLIKE gain fraction
DM_ALPHA = 0.05            # date-clustered DM p-value must be below this
KILL_HORIZONS = (1, 5)     # BOTH must pass (gain>GAIN_MIN AND p<DM_ALPHA) for pre-registered success

# --- HOSE regime-spike robustness (project rule) ---
SPIKE_WINDOWS = (("2020-02-01", "2020-04-30"),   # COVID crash
                 ("2022-01-01", "2022-12-31"),   # 2022 VN drawdown
                 ("2025-04-01", "2025-04-30"))   # Apr-2025 tariff shock
