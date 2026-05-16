"""Rolling walk-forward splitting (spec section 5).

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
    periods = idx.to_period('M')
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


def optuna_holdout(train_mask: np.ndarray, frac: float = 0.15):
    """Within a training window, carve the most-recent `frac` as the Optuna
    validation fold (strictly temporal — the tail, never shuffled).

    Returns (fit_mask, val_mask) over the SAME full-length axis as train_mask.
    """
    pos = np.flatnonzero(train_mask)
    if len(pos) < 10:
        return train_mask.copy(), np.zeros_like(train_mask)
    cut = int(len(pos) * (1.0 - frac))
    fit_mask = np.zeros_like(train_mask)
    val_mask = np.zeros_like(train_mask)
    fit_mask[pos[:cut]] = True
    val_mask[pos[cut:]] = True
    return fit_mask, val_mask
