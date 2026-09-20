"""Bootstrap sys.path so the baseline's code modules and their read-only shared deps import in tests
(folder names contain dashes, so `python -m` / package imports are not usable)."""
import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1] / "code"
REPO = _CODE.parents[2]
for _p in (str(REPO), str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"),
           str(REPO / "scripts" / "quality_gate"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
