HAR_WINDOWS = (1, 5, 22)
SEMI_WINDOW = 5
SEMI_MIN_PERIODS = 3
HORIZONS = (1, 5, 10, 22)
MIN_ROWS = {"sp500": 30000, "default": 3000}
RANGE_SHORT = 5
RANGE_LONG = 22

OWN_DROP = ("rq",)
OWN_ADD = ()

def own_set(fm_own):
    return [f for f in fm_own if f not in OWN_DROP] + list(OWN_ADD)
