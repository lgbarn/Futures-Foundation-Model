"""Keltner Channel + SuperTrend indicators.

Ported from trading-research/Python/lib/indicators (channels.py, core.py) so the
labeler reproduces the mechanical strategy exactly. Vectorised to numpy arrays;
the EMA / SuperTrend recurrences are kept bit-for-bit identical to the source.
"""

import numpy as np


def calc_ema(values: np.ndarray, period: int) -> np.ndarray:
    """EMA seeded with the first value (matches trading-research calc_ema)."""
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    if len(values) == 0:
        return out
    k = 2.0 / (period + 1.0)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1.0 - k)
    return out


def calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """ATR = EMA of True Range. TR[0] = high-low (matches trading-research)."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(high)
    tr = np.empty(n)
    if n == 0:
        return tr
    tr[0] = high[0] - low[0]
    if n > 1:
        prev_close = close[:-1]
        tr[1:] = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - prev_close),
            np.abs(low[1:] - prev_close),
        ])
    return calc_ema(tr, period)


def calc_keltner(high, low, close, ema_period: int, atr_period: int, atr_mult: float):
    """Keltner Channel: EMA(close) middle, +/- ATR*mult bands.

    Returns (middle, upper, lower) numpy arrays.
    """
    close = np.asarray(close, dtype=np.float64)
    middle = calc_ema(close, ema_period)
    atr = calc_atr(high, low, close, atr_period)
    upper = middle + atr * atr_mult
    lower = middle - atr * atr_mult
    return middle, upper, lower


def calc_supertrend(high, low, close, period: int = 10, multiplier: float = 3.0):
    """SuperTrend over ATR bands around HL2.

    Returns (value, direction) numpy arrays. direction is +1 (up) or -1 (down).
    Recurrence is a faithful port of trading-research calc_supertrend.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(high)
    value = np.zeros(n)
    direction = np.zeros(n, dtype=np.int8)
    if n == 0:
        return value, direction

    atr = calc_atr(high, low, close, period)
    prev_upper = prev_lower = prev_st = 0.0

    for i in range(n):
        hl2 = (high[i] + low[i]) / 2.0
        basic_upper = hl2 + multiplier * atr[i]
        basic_lower = hl2 - multiplier * atr[i]

        if i == 0:
            final_upper = basic_upper
            final_lower = basic_lower
        else:
            prev_close = close[i - 1]
            final_upper = basic_upper if (basic_upper < prev_upper or prev_close > prev_upper) else prev_upper
            final_lower = basic_lower if (basic_lower > prev_lower or prev_close < prev_lower) else prev_lower

        if i == 0:
            d = 1
            st = final_lower
        else:
            if prev_st == prev_upper:
                d = 1 if close[i] > final_upper else -1
            else:
                d = -1 if close[i] < final_lower else 1
            st = final_lower if d == 1 else final_upper

        value[i] = st
        direction[i] = d
        prev_upper, prev_lower, prev_st = final_upper, final_lower, st

    return value, direction
