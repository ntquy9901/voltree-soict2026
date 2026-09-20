HORIZONS = (1, 5, 10, 22)
MIN_ROWS = {"sp500": 30000, "default": 3000}

OWN_DROP = ("rq",)
OWN_ADD = ()

def own_set(fm_own):
    return [f for f in fm_own if f not in OWN_DROP] + list(OWN_ADD)
