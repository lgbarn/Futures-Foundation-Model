"""Rolling walk-forward splitting for the RL pipeline.

3-month train / 1-month OOS test, stride 1 month, UNANCHORED (drop the oldest
month each step). ~12 months data -> ~8-9 independent OOS months. A model is
only credible if EVERY OOS month is profitable (monthly PF > 1). Splits are
strictly temporal — never shuffle.

Trading-research evidence: short 3-month training windows generalize far
better than 6/18-month for intraday futures (regimes shift fast). 3:1 is the
validated ratio.
"""
import numpy as np
import pandas as pd


def walk_forward_windows(index: pd.DatetimeIndex,
                         train_months: int = 3,
                         test_months: int = 1):
    """Yield (train_mask, test_mask) boolean arrays — month-aligned rolling
    unanchored windows. Stride = test_months (retrain monthly)."""
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    # drop tz explicitly before to_period (month bucketing is tz-immaterial;
    # silences the "will drop timezone" warning, deterministic on wall-clock)
    _idx = idx.tz_localize(None) if idx.tz is not None else idx
    periods = _idx.to_period('M')
    months = periods.unique().sort_values()
    step = test_months
    s = 0
    while s + train_months + test_months <= len(months):
        tr = months[s:s + train_months]
        te = months[s + train_months:s + train_months + test_months]
        train_mask = np.asarray(periods.isin(tr))
        test_mask = np.asarray(periods.isin(te))
        if train_mask.any() and test_mask.any():
            yield train_mask, test_mask
        s += step
