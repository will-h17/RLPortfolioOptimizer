import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"

# Put repo root and src/ at the front of sys.path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
    
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def tiny_market():
    """
    6 business days, 2 assets. Returns designed so we can compute exact PnL:
      Day 0: baseline
      Day 1..5: simple patterns
    """
    idx = pd.bdate_range("2020-01-01", periods=6)
    # Construct prices so pct_change yields handy returns:
    # r = [0,  1%,  2%, -1%, 0%, 3%] on A
    # r = [0, -1%,  0%,  1%, 2%, 0%] on B
    RA = np.array([0.00, 0.01, 0.02,-0.01, 0.00, 0.03])
    RB = np.array([0.00,-0.01, 0.00, 0.01, 0.02, 0.00])
    A = 100 * np.cumprod(1 + RA)
    B =  50 * np.cumprod(1 + RB)
    prices = pd.DataFrame({"A": A, "B": B}, index=idx)

    # Features: past 1-day returns, *shifted by 1* so there is no look-ahead.
    r = prices.pct_change().fillna(0.0)
    feats = pd.DataFrame({
        "A_r1": r["A"].shift(1).fillna(0.0),
        "B_r1": r["B"].shift(1).fillna(0.0),
    }, index=idx)
    return prices, feats
