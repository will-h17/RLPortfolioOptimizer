import numpy as np
from src.utils.metrics import turnover as avg_turnover
from src.utils.metrics import max_drawdown as max_dd, sharpe_ratio as sharpe

def test_turnover_l1_mean():
    W = np.array([
        [0.5, 0.5],
        [0.6, 0.4],  # L1 change = 0.2
        [0.6, 0.4],  # L1 change = 0.0
        [0.3, 0.7],  # L1 change = 0.6
    ])
    # mean over (0.2, 0.0, 0.6) = 0.266666...
    assert np.isclose(avg_turnover([w for w in W]), (0.2 + 0.0 + 0.6) / 3, atol=1e-12)

def test_metrics_sharpe_and_mdd():
    # simple equity: up 1%, flat, down 2%, up 1% → check Sharpe sign and MDD magnitude
    rets = np.array([0.01, 0.0, -0.02, 0.01], dtype=float)
    eq = (1 + rets).cumprod()
    s = sharpe(rets)
    mdd = max_dd(eq)
    # For this series mean≈0 → Sharpe≈0. Just ensure finite and mdd is negative and modest.
    assert np.isfinite(s)
    assert mdd <= 0.0 and mdd >= -0.03
