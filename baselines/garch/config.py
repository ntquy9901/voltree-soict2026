SCALE = 100.0                # returns are scaled x100 for arch's numerical conditioning; forecasts /SCALE**2
MEAN = "Constant"            # arch mean model (epsilon = r - mu)
DIST = "normal"             # innovation distribution
HORIZONS = (1, 5, 10, 22)   # forecast horizons
MIN_ROWS = {"sp500": 30000, "default": 3000}   # min pooled train rows per fold
MIN_TRAIN_OBS = 250         # min per-ticker train returns to attempt an ML fit, else fallback
PERSIST_LO = 0.0            # reversion persistence must satisfy PERSIST_LO < phi < PERSIST_HI
PERSIST_HI = 1.0
VAR_RATIO_CAP = 1000.0      # the fit-implied unconditional variance AND every forecast must stay within this
